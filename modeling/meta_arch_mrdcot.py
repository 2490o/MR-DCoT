"""MR-DCoT experimental branch.

A compact implementation inspired by the uploaded MR-DCoT paper:
1) keep the original SE-COT textual style evolution;
2) add a visual chain with local structural perturbation;
3) add prototype-anchored manifold regression for off-manifold features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

from .meta_arch_newCOT import ClipRCNNSECOT, convblock


class VisualChainPerturbation(nn.Module):
    def __init__(self, channels, noise_std=0.08, blur_kernel=3, blur_prob=0.5):
        super().__init__()
        self.noise_std = noise_std
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.blur_prob = blur_prob
        self.restore = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def _blur(self, x):
        if self.blur_kernel <= 1 or np.random.rand() > self.blur_prob:
            return x
        pad = self.blur_kernel // 2
        return F.avg_pool2d(x, kernel_size=self.blur_kernel, stride=1, padding=pad)

    def forward(self, content_feat):
        noisy = content_feat + torch.randn_like(content_feat) * self.noise_std
        diffused = self._blur(noisy)
        restored = self.restore(diffused) + content_feat
        return diffused, restored


class GlobalManifoldRegressor(nn.Module):
    def __init__(self, in_channels, proto_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.mapper = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, proto_dim),
            nn.LayerNorm(proto_dim),
            nn.ReLU(inplace=False),
            nn.Linear(proto_dim, proto_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))

    def map_feature(self, feat):
        return F.normalize(self.mapper(feat), dim=1)

    def forward(self, src_feat, off_feat, labels, temperature=0.1, local_weight=0.5):
        z_src = self.map_feature(src_feat).detach()
        z_off = self.map_feature(off_feat)
        protos = F.normalize(self.prototypes, dim=1)
        labels = labels.clamp(min=0, max=self.num_classes - 1)
        logits = torch.matmul(z_off, protos.t()) / temperature
        loss_global = F.cross_entropy(logits, labels)
        loss_local = F.mse_loss(z_off, z_src)
        return loss_global + local_weight * loss_local, loss_global, loss_local


@META_ARCH_REGISTRY.register()
class ClipRCNNMRDCOT(ClipRCNNSECOT):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.w_mr_align = cfg.MODEL.MR_LOSS_WEIGHT_ALIGN
        self.w_mr_diff = cfg.MODEL.MR_LOSS_WEIGHT_DIFF
        self.w_mr_reg = cfg.MODEL.MR_LOSS_WEIGHT_REG
        self.w_mr_local = cfg.MODEL.MR_LOSS_WEIGHT_LOCAL
        self.mr_tau = cfg.MODEL.MR_TEMPERATURE
        self.visual_chain = VisualChainPerturbation(
            self.style_c,
            noise_std=cfg.MODEL.MR_VISUAL_NOISE_STD,
            blur_kernel=cfg.MODEL.MR_BLUR_KERNEL,
            blur_prob=cfg.MODEL.MR_BLUR_PROB,
        )
        self.off_merge_conv = convblock(self.style_c * 3, self.style_c)
        self.manifold_regressor = GlobalManifoldRegressor(1024, self.style_c, self.num_classes)

    def _pooled_cosine_loss(self, a, b):
        a = F.adaptive_avg_pool2d(a, 1).flatten(1)
        b = F.adaptive_avg_pool2d(b, 1).flatten(1)
        return 1.0 - (F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(dim=1).mean()

    def _image_labels(self, gt_instances):
        labels = []
        for inst in gt_instances:
            if len(inst.gt_classes) == 0:
                labels.append(torch.zeros((), dtype=torch.long, device=self.device))
            else:
                labels.append(torch.mode(inst.gt_classes.to(self.device))[0])
        return torch.stack(labels, 0)

    def _build_offmanifold_features(self, f1, fs, fc):
        mean, std = self._pick_style_params(f1.shape[0])
        text_style = self._adain(fs, mean, std)
        diffused_content, visual_content = self.visual_chain(fc)
        fused = self.off_merge_conv(torch.cat([text_style, visual_content, diffused_content], dim=1)) + f1
        off_features = self.backbone.forward_l2_to_res4(fused)
        off_features["res4"] = self.cpcm(off_features["res4"])
        loss_align = self._pooled_cosine_loss(text_style, fs)
        loss_diff = F.mse_loss(visual_content, fc.detach())
        return off_features, loss_align, loss_diff

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        features, f1, fs, fc = self._build_features(images.tensor, apply_style_transfer=True)
        off_features, loss_align, loss_diff = self._build_offmanifold_features(f1, fs, fc)

        image_labels = self._image_labels(gt_instances)
        loss_reg, _, _ = self.manifold_regressor(
            features["res4"], off_features["res4"], image_labels, self.mr_tau, self.w_mr_local
        )
        losses = self._disentangle_losses(f1, fs, fc, features["res4"])
        losses.update({
            "loss_mr_align": loss_align * self.w_mr_align,
            "loss_mr_diff": loss_diff * self.w_mr_diff,
            "loss_mr_reg": loss_reg * self.w_mr_reg,
        })

        if self.proposal_generator is not None:
            _, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        try:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def _safe_fast_rcnn_inference(self, predictions, proposals):
        scores, proposal_deltas = predictions
        boxes = self.roi_heads.box_predictor.predict_boxes((scores, proposal_deltas), proposals)
        scores = self.roi_heads.box_predictor.predict_probs((scores, proposal_deltas), proposals)
        image_shapes = [x.image_size for x in proposals]
        score_thresh = self.roi_heads.box_predictor.test_score_thresh
        nms_thresh = self.roi_heads.box_predictor.test_nms_thresh
        topk = self.roi_heads.box_predictor.test_topk_per_image
        results = []
        start = 0
        for boxes_per_image, scores_per_image, image_shape in zip(boxes, scores, image_shapes):
            num_boxes = boxes_per_image.shape[0]
            scores_fg = scores_per_image[:, :-1]
            num_bbox_reg_classes = boxes_per_image.shape[1] // 4
            boxes_per_image = boxes_per_image.reshape(num_boxes, num_bbox_reg_classes, 4)
            if num_bbox_reg_classes == 1:
                boxes_for_classes = boxes_per_image.expand(num_boxes, scores_fg.shape[1], 4)
            else:
                boxes_for_classes = boxes_per_image[:, :scores_fg.shape[1], :]
            filter_mask = scores_fg > score_thresh
            filter_inds = filter_mask.nonzero()
            if filter_inds.numel() == 0:
                device = scores_per_image.device
                inst = Instances(image_shape)
                inst.pred_boxes = Boxes(torch.empty((0, 4), device=device))
                inst.scores = torch.empty((0,), device=device)
                inst.pred_classes = torch.empty((0,), dtype=torch.long, device=device)
                results.append(inst)
                start += num_boxes
                continue
            selected_boxes = boxes_for_classes[filter_inds[:, 0], filter_inds[:, 1]]
            selected_scores = scores_fg[filter_mask]
            selected_classes = filter_inds[:, 1]
            keep = batched_nms(selected_boxes, selected_scores, selected_classes, nms_thresh)
            if topk >= 0:
                keep = keep[:topk]
            inst = Instances(image_shape)
            inst.pred_boxes = Boxes(selected_boxes[keep])
            inst.scores = selected_scores[keep]
            inst.pred_classes = selected_classes[keep]
            results.append(inst)
            start += num_boxes
        return results

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features, _, _, _ = self._build_features(images.tensor, apply_style_transfer=False)
        if detected_instances is None:
            if self.proposal_generator is not None:
                _, proposals, _ = self.proposal_generator(images, features, None)
            else:
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            self.roi_heads.fwdres5 = self.backbone.forward_res5
            proposal_boxes = [x.proposal_boxes for x in proposals]
            box_features = self.roi_heads._shared_roi_transform(
                [features[f] for f in self.roi_heads.in_features], proposal_boxes
            )
            attn_feat = self.backbone.attention_global_pool(box_features)
            predictions = self.roi_heads.box_predictor([attn_feat, box_features.mean(dim=(2, 3))])
            results = self._safe_fast_rcnn_inference(predictions, proposals)
            results = self.roi_heads.forward_with_given_boxes(features, results)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results


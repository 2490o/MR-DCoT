"""Guarded MR-DCoT meta-architecture for the DATAC baseline.

This file intentionally does not reuse previous experimental MR-DCoT variants.
It keeps the successful DATAC source path intact and adds a zero-initialized,
warmup-gated residual manifold-regression branch.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

from .meta_arch_COT3_30 import (
    ClipRCNNWithClipBackboneWithOffsetGenTrainable,
    calc_mean_std,
)


class ZeroInitResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden = max(out_channels // 2, 128)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, 1, 0),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, out_channels, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class PrototypeFeatureResidual(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.assign = nn.Conv2d(channels, num_classes, 1)
        self.prototypes = nn.Parameter(torch.randn(num_classes, channels) * 0.02)
        self.fuse = ZeroInitResidualConv(channels * 2, channels)

    def forward(self, x):
        theta = torch.softmax(self.assign(x), dim=1)
        proto = F.normalize(self.prototypes, dim=1)
        proto_map = torch.einsum("bkhw,kc->bchw", theta, proto)
        return self.fuse(torch.cat([x, proto_map], dim=1))


class GuardedVisualChain(nn.Module):
    def __init__(self, channels, noise_std=0.03, blur_kernel=3, steps=2, drop_prob=0.0):
        super().__init__()
        self.noise_std = float(noise_std)
        self.blur_kernel = int(blur_kernel) if int(blur_kernel) % 2 == 1 else int(blur_kernel) + 1
        self.steps = max(1, int(steps))
        self.drop_prob = float(drop_prob)
        self.restore = ZeroInitResidualConv(channels, channels)

    def _blur(self, x):
        if self.blur_kernel <= 1:
            return x
        pad = self.blur_kernel // 2
        return F.avg_pool2d(x, self.blur_kernel, stride=1, padding=pad)

    def forward(self, content):
        shifted = content
        for _ in range(self.steps):
            local_std = shifted.flatten(2).std(dim=2, keepdim=True).view(
                shifted.shape[0], shifted.shape[1], 1, 1
            )
            noise = torch.randn_like(shifted) * self.noise_std * (1.0 + local_std.detach())
            shifted = self._blur(shifted + noise)
        if self.training and self.drop_prob > 0:
            shifted = F.dropout2d(shifted, p=self.drop_prob, training=True)
        restored = shifted + self.restore(shifted)
        return shifted, restored


class ROIPrototypeRegressor(nn.Module):
    def __init__(self, in_dim, proto_dim, num_classes):
        super().__init__()
        self.num_classes = int(num_classes)
        self.mapper = nn.Sequential(
            nn.Linear(in_dim, proto_dim),
            nn.LayerNorm(proto_dim),
            nn.ReLU(inplace=False),
            nn.Linear(proto_dim, proto_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, proto_dim) * 0.02)

    def map(self, x):
        return F.normalize(self.mapper(x), dim=1)

    def forward(self, source_feat, off_feat, labels, tau, local_weight, source_weight):
        labels = labels.clamp(min=0, max=self.num_classes - 1)
        z_source = self.map(source_feat)
        z_off = self.map(off_feat)
        prototypes = F.normalize(self.prototypes, dim=1)

        logits_off = torch.matmul(z_off, prototypes.t()) / tau
        loss_global = F.cross_entropy(logits_off, labels)
        loss_local = F.mse_loss(z_off, z_source.detach())

        if source_weight > 0:
            logits_source = torch.matmul(z_source, prototypes.t()) / tau
            loss_source = F.cross_entropy(logits_source, labels)
        else:
            loss_source = z_source.sum() * 0.0

        loss = loss_global + local_weight * loss_local + source_weight * loss_source
        return loss, loss_global.detach(), loss_local.detach(), loss_source.detach()


@META_ARCH_REGISTRY.register()
class ClipRCNNDATACMRDCOTGuard(ClipRCNNWithClipBackboneWithOffsetGenTrainable):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.mr_enable = bool(cfg.MODEL.MR_GUARD_ENABLE)
        self.mr_force_baseline = bool(cfg.MODEL.MR_GUARD_FORCE_BASELINE)
        self.mr_start_iter = int(cfg.MODEL.MR_GUARD_START_ITER)
        self.mr_warmup_iters = int(cfg.MODEL.MR_GUARD_WARMUP_ITERS)
        self.mr_max_alpha = float(cfg.MODEL.MR_GUARD_MAX_ALPHA)
        self.mr_tau = float(cfg.MODEL.MR_GUARD_TEMPERATURE)
        self.mr_local_weight = float(cfg.MODEL.MR_GUARD_LOCAL_WEIGHT)
        self.mr_source_proto_weight = float(cfg.MODEL.MR_GUARD_SOURCE_PROTO_WEIGHT)
        self.mr_loss_align_weight = float(cfg.MODEL.MR_GUARD_LOSS_ALIGN)
        self.mr_loss_diff_weight = float(cfg.MODEL.MR_GUARD_LOSS_DIFF)
        self.mr_loss_reg_weight = float(cfg.MODEL.MR_GUARD_LOSS_REG)
        self.mr_loss_off_det_weight = float(cfg.MODEL.MR_GUARD_LOSS_OFF_DET)
        self.mr_off_det_max_images = int(cfg.MODEL.MR_GUARD_OFF_DET_MAX_IMAGES)
        self.mr_max_gt_rois = int(cfg.MODEL.MR_GUARD_MAX_GT_ROIS)
        self.mr_style_mean_scale = float(cfg.MODEL.MR_GUARD_STYLE_MEAN_SCALE)
        self.mr_style_std_scale = float(cfg.MODEL.MR_GUARD_STYLE_STD_SCALE)

        channels = 1024
        num_classes = int(cfg.MODEL.NUM_CLASSES)
        clip_dim = self._clip_embed_dim()

        self.mr_gate = nn.Parameter(torch.tensor(float(cfg.MODEL.MR_GUARD_GATE_INIT)))
        self.mr_feature_residual = PrototypeFeatureResidual(channels, num_classes)
        self.mr_visual_chain = GuardedVisualChain(
            channels,
            noise_std=cfg.MODEL.MR_GUARD_VISUAL_NOISE_STD,
            blur_kernel=cfg.MODEL.MR_GUARD_BLUR_KERNEL,
            steps=cfg.MODEL.MR_GUARD_VISUAL_STEPS,
            drop_prob=cfg.MODEL.MR_GUARD_DROP_PROB,
        )
        self.mr_off_fuse = ZeroInitResidualConv(channels * 3, channels)
        self.mr_style_to_clip = nn.Linear(channels, clip_dim)
        self.mr_regressor = ROIPrototypeRegressor(
            clip_dim, int(cfg.MODEL.MR_GUARD_PROTO_DIM), num_classes
        )
        self._mr_text_bank = None
        self._mr_text_bank_device = None

    def _clip_embed_dim(self):
        clip_model = self.roi_heads.box_predictor.cls_score.model
        if hasattr(clip_model, "text_projection"):
            return int(clip_model.text_projection.shape[1])
        return 1024

    def _current_mr_scale(self):
        if not self.mr_enable or self.mr_force_baseline:
            return 0.0
        if not self.training:
            return 1.0
        try:
            cur_iter = int(get_event_storage().iter)
        except Exception:
            cur_iter = self.mr_start_iter + self.mr_warmup_iters
        if cur_iter < self.mr_start_iter:
            return 0.0
        if self.mr_warmup_iters <= 0:
            return 1.0
        return min(1.0, float(cur_iter - self.mr_start_iter + 1) / float(self.mr_warmup_iters))

    def _residual_alpha(self, scale):
        if scale <= 0:
            return self.mr_gate.new_tensor(0.0)
        return self.mr_max_alpha * torch.sigmoid(self.mr_gate) * scale

    def _encode_text_bank(self):
        device = self.stylemean.device
        if self._mr_text_bank is not None and self._mr_text_bank_device == device:
            return self._mr_text_bank

        clip_model = self.roi_heads.box_predictor.cls_score.model
        embeds = []
        with torch.no_grad():
            for tokens in self.cot_tokens:
                ft1 = clip_model.encode_text(tokens["l1"].to(device))
                ft2 = clip_model.encode_text(tokens["l2"].to(device)) + ft1
                ft3 = clip_model.encode_text(tokens["l3"].to(device)) + ft2
                embeds.append(F.normalize(ft3, dim=-1).squeeze(0))
        self._mr_text_bank = torch.stack(embeds, dim=0)
        self._mr_text_bank_device = device
        return self._mr_text_bank

    def _sample_text_style(self, batch_size, device):
        indices = torch.randint(len(self.cot_tokens), (batch_size,), device=device)
        mean = self.stylemean[indices].to(device)
        raw_std = self.stylestd[indices].to(device)
        mean = self.mr_style_mean_scale * torch.tanh(mean)
        std = 1.0 + self.mr_style_std_scale * torch.tanh(raw_std - 1.0)
        text_embed = self._encode_text_bank()[indices]
        return mean, std, text_embed

    @staticmethod
    def _adain(feat, mean, std):
        feat_mean, feat_std = calc_mean_std(feat)
        return ((feat - feat_mean) / (feat_std + 1e-6)) * std + mean

    def _build_source_features(self, raw_res4, apply_style_aug=True):
        base_di = self.di(raw_res4)
        base_ds = self.ds(raw_res4)
        proto = self.pro(raw_res4)

        loss_dp = self.consistency_loss(proto, base_di)
        zero_loss_di_p = F.normalize(base_di) * F.normalize(raw_res4)
        zero_loss_di_n = F.normalize(base_ds) * F.normalize(raw_res4)
        zero_loss_di_p = torch.exp(torch.sum(zero_loss_di_p, dim=1))
        zero_loss_di_n = torch.exp(torch.sum(zero_loss_di_n, dim=1))
        zero_loss_di = torch.mean(
            torch.log(zero_loss_di_p / (zero_loss_di_p + zero_loss_di_n)) * -1.0
        )

        if apply_style_aug and np.random.rand(1) > self.apply_aug:
            batch_size = raw_res4.shape[0]
            style_ids = np.random.choice(np.arange(len(self.stylemean)), batch_size)
            mean = torch.cat(
                [
                    self.stylemean[style_id : style_id + 1]
                    .to(raw_res4.device)
                    .mean(dim=(2, 3), keepdims=True)
                    for style_id in style_ids
                ],
                0,
            )
            std = torch.cat(
                [
                    self.stylestd[style_id : style_id + 1]
                    .to(raw_res4.device)
                    .mean(dim=(2, 3), keepdims=True)
                    for style_id in style_ids
                ],
                0,
            )
            base_ds = base_ds * std.expand(base_ds.size()) + mean.expand(base_ds.size())

        source_res4 = self.conv_out(torch.cat((base_di, base_ds), dim=1)) + raw_res4
        source_res4 = self.pro(source_res4)
        losses = {"loss_dp": loss_dp, "zero_loss_di": zero_loss_di}
        return {"res4": source_res4}, base_di, base_ds, losses

    def _apply_guard_residual(self, source_res4, scale):
        alpha = self._residual_alpha(scale)
        if float(alpha.detach().cpu()) == 0.0:
            return source_res4
        return source_res4 + alpha * self.mr_feature_residual(source_res4)

    def _build_off_manifold_features(self, source_res4, base_di, base_ds, scale):
        mean, std, text_embed = self._sample_text_style(base_ds.shape[0], base_ds.device)
        text_style = self._adain(base_ds, mean, std)
        visual_shifted, visual_restored = self.mr_visual_chain(base_di)
        off_delta = self.mr_off_fuse(torch.cat([text_style, visual_restored, visual_shifted], dim=1))
        off_res4 = source_res4 + scale * off_delta

        style_vec = F.adaptive_avg_pool2d(text_style, 1).flatten(1)
        style_vec = F.normalize(self.mr_style_to_clip(style_vec), dim=1)
        text_embed = F.normalize(text_embed, dim=1)
        loss_align = 1.0 - (style_vec * text_embed).sum(dim=1).mean()
        loss_diff = F.mse_loss(visual_restored, base_di.detach())
        return {"res4": off_res4}, loss_align, loss_diff

    def _detector_losses(self, images, features, gt_instances):
        if self.proposal_generator is not None:
            _, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            raise ValueError("ClipRCNNDATACMRDCOTGuard expects a proposal generator.")
        try:
            _, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances, None, self.backbone
            )
        except Exception:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None)
        return detector_losses, proposal_losses

    def _collect_gt_boxes(self, gt_instances):
        boxes, labels = [], []
        remaining = self.mr_max_gt_rois
        for inst in gt_instances:
            if len(inst) == 0 or remaining <= 0:
                boxes.append(Boxes(torch.empty((0, 4), device=self.device)))
                continue
            cur_boxes = inst.gt_boxes.tensor
            cur_labels = inst.gt_classes
            if len(cur_boxes) > remaining:
                perm = torch.randperm(len(cur_boxes), device=cur_boxes.device)[:remaining]
                cur_boxes = cur_boxes[perm]
                cur_labels = cur_labels[perm]
            boxes.append(Boxes(cur_boxes))
            labels.append(cur_labels)
            remaining -= len(cur_boxes)
        if not labels:
            return boxes, None
        return boxes, torch.cat(labels, dim=0)

    def _roi_embedding(self, features, boxes):
        self.roi_heads.fwdres5 = self.backbone.forward_res5
        roi_feat = self.roi_heads._shared_roi_transform(
            [features[f] for f in self.roi_heads.in_features], boxes
        )
        return self.backbone.attention_global_pool(roi_feat)

    def _manifold_losses(self, source_features, off_features, gt_instances):
        boxes, labels = self._collect_gt_boxes(gt_instances)
        if labels is None or labels.numel() == 0:
            zero = source_features["res4"].sum() * 0.0
            return {
                "loss_mr_guard_reg": zero,
                "loss_mr_guard_global": zero,
                "loss_mr_guard_local": zero,
                "loss_mr_guard_src_proto": zero,
            }

        source_embed = self._roi_embedding(source_features, boxes)
        off_embed = self._roi_embedding(off_features, boxes)
        loss, loss_global, loss_local, loss_source = self.mr_regressor(
            source_embed,
            off_embed,
            labels.to(source_embed.device),
            tau=self.mr_tau,
            local_weight=self.mr_local_weight,
            source_weight=self.mr_source_proto_weight,
        )
        return {
            "loss_mr_guard_reg": loss * self.mr_loss_reg_weight,
            "loss_mr_guard_global": loss_global,
            "loss_mr_guard_local": loss_local,
            "loss_mr_guard_src_proto": loss_source,
        }

    @staticmethod
    def _weighted_prefixed_losses(losses, prefix, weight):
        return {f"{prefix}_{name}": value * weight for name, value in losses.items()}

    def _slice_images(self, images, count):
        return type(images)(images.tensor[:count], images.image_sizes[:count])

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        raw_features = self.backbone(images.tensor)

        source_features, base_di, base_ds, losses = self._build_source_features(
            raw_features["res4"], apply_style_aug=True
        )
        mr_scale = self._current_mr_scale()
        det_features = {"res4": self._apply_guard_residual(source_features["res4"], mr_scale)}
        detector_losses, proposal_losses = self._detector_losses(images, det_features, gt_instances)
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if mr_scale > 0:
            off_features, loss_align, loss_diff = self._build_off_manifold_features(
                source_features["res4"], base_di, base_ds, mr_scale
            )
            losses["loss_mr_guard_align"] = loss_align * self.mr_loss_align_weight * mr_scale
            losses["loss_mr_guard_diff"] = loss_diff * self.mr_loss_diff_weight * mr_scale

            mr_losses = self._manifold_losses(source_features, off_features, gt_instances)
            for name, value in mr_losses.items():
                losses[name] = value * mr_scale if name == "loss_mr_guard_reg" else value

            if self.mr_loss_off_det_weight > 0 and self.mr_off_det_max_images > 0:
                count = min(self.mr_off_det_max_images, len(gt_instances))
                off_images = self._slice_images(images, count)
                off_subset = {"res4": off_features["res4"][:count]}
                off_det_losses, off_prop_losses = self._detector_losses(
                    off_images, off_subset, gt_instances[:count]
                )
                weight = self.mr_loss_off_det_weight * mr_scale
                losses.update(self._weighted_prefixed_losses(off_det_losses, "mr_off", weight))
                losses.update(self._weighted_prefixed_losses(off_prop_losses, "mr_off", weight))

        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        base_di = self.di(features["res4"])
        base_ds = self.ds(features["res4"])
        source_res4 = self.conv_out(torch.cat((base_di, base_ds), dim=1)) + features["res4"]
        source_res4 = self.pro(source_res4)
        features["res4"] = self._apply_guard_residual(source_res4, self._current_mr_scale())

        if detected_instances is None:
            if self.proposal_generator is not None:
                _, proposals, _ = self.proposal_generator(images, features, None)
            else:
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            try:
                results, _ = self.roi_heads(images, features, proposals, None, None, self.backbone)
            except Exception:
                results, _ = self.roi_heads(images, features, proposals, None, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

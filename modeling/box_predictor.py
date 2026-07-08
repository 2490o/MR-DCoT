from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn  # <--- [修改1] 必须导入 nn

from detectron2.layers import cat, cross_entropy
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from .clip import ClipPredictor


class ClipFastRCNNOutputLayers(FastRCNNOutputLayers):

    def __init__(self, cfg, input_shape, clsnames) -> None:
        super().__init__(cfg, input_shape)
        pretrained = getattr(cfg.MODEL, "CLIP_PRETRAINED_WEIGHTS", None)

        # [修改2] 定义 LayerNorm，用于稳定回归分支的特征分布
        self.bbox_pred_ln = nn.LayerNorm(input_shape.channels)

        self.cls_score = ClipPredictor(
            cfg.MODEL.CLIP_IMAGE_ENCODER_NAME,
            input_shape.channels,
            cfg.MODEL.DEVICE,
            clsnames,
            pretrained_weights=pretrained
        )

    def forward(self, x, gfeat=None):
        ## for features from clip model
        if isinstance(x, list):
            # x[0] 是 attention pooled 特征 (用于分类)
            # x[1] 是 mean pooled 特征 (用于回归)
            scores = self.cls_score(x[0], gfeat)

            # [修改3] 在送入 bbox_pred 之前，先过 LayerNorm
            reg_feat = x[1]
            if reg_feat.dim() == 2:  # 确保是 (Batch, Channels)
                reg_feat = self.bbox_pred_ln(reg_feat)

            proposal_deltas = self.bbox_pred(reg_feat)
        else:
            # 兼容非 list 输入的情况
            scores = self.cls_score(x, gfeat)

            # 同样应用 LayerNorm (如果 x 是特征本身)
            reg_feat = x
            if reg_feat.dim() == 2:
                reg_feat = self.bbox_pred_ln(reg_feat)

            proposal_deltas = self.bbox_pred(reg_feat)

        return scores, proposal_deltas
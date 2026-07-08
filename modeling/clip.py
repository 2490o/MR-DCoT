# clip.py
import clip
import torch
import torch.nn as nn
import numpy as np
import copy
import open_clip
import os


class ClipPredictor(nn.Module):
    def __init__(self, clip_encoder_name, inshape, device, clsnames, pretrained_weights=None):
        super().__init__()

        # 1. 初始化投影层
        # CLIP 本身是 feat @ proj 的形式，这里用 bias=False 更贴近原始实现
        self.projection = nn.Linear(inshape, 512, bias=False)

        # 2. 加载模型
        if "swin" in clip_encoder_name.lower() or "convnext" in clip_encoder_name.lower():
            print(f"Loading OpenCLIP model: {clip_encoder_name}")

            # 创建空模型
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_encoder_name, pretrained=None, device=device
            )

            # 手动加载权重
            if pretrained_weights and os.path.exists(pretrained_weights):
                print(f"Manual loading weights from: {pretrained_weights}")
                checkpoint = torch.load(pretrained_weights, map_location=device)
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                # 加载主干权重
                incompatible = self.model.load_state_dict(new_state_dict, strict=False)
                print(f"Weights loaded. Missing: {len(incompatible.missing_keys)}")

                # --- 复制投影层权重 (关键步骤) ---
                # OpenCLIP 的 Swin/ConvNeXt 通常在 visual.head 的最后一层做 (width -> 512) 投影。
                # 这里把预训练投影权重拷到本模块的 projection，避免随机投影导致分类塌陷。
                if hasattr(self.model.visual, 'head'):
                    pretrained_proj = self.model.visual.head[-1]
                    if isinstance(pretrained_proj, nn.Linear):
                        print(f"Copying pretrained projection weights: {pretrained_proj.in_features} -> {pretrained_proj.out_features}")
                        with torch.no_grad():
                            if (pretrained_proj.out_features == 512) and (pretrained_proj.in_features == inshape):
                                # Linear.weight: [out_features, in_features]
                                self.projection.weight.copy_(pretrained_proj.weight)
                            else:
                                print('[Warning] Projection shape mismatch; skip copying pretrained projection.')
            else:
                if pretrained_weights:
                    raise FileNotFoundError(f"Weights not found: {pretrained_weights}")

            self.tokenizer = open_clip.get_tokenizer(clip_encoder_name)
        else:
            # 兼容 OpenAI CLIP
            self.model, self.preprocess = clip.load(clip_encoder_name, device=device)
            self.tokenizer = clip.tokenize

            # --- Copy OpenAI CLIP ViT visual projection (proj) ---
            # For OpenAI CLIP ViT, visual.proj has shape [width, 512] and maps visual features
            # into the CLIP semantic embedding space. Without copying it, the ROI classifier
            # often collapses (many classes AP=0) because foreground logits are suppressed
            # by the fixed background logit (=0).
            if hasattr(self.model.visual, 'proj') and self.model.visual.proj is not None:
                with torch.no_grad():
                    proj = self.model.visual.proj
                    if (proj.shape[0] == inshape) and (proj.shape[1] == 512):
                        # self.projection is Linear(inshape -> 512, bias=False); weight is [512, inshape]
                        self.projection.weight.copy_(proj.T)
                    else:
                        print('[Warning] OpenAI CLIP visual.proj shape mismatch; skip copying projection.')
            # ----------------------------------------------------------

        self.model.float()
        for p in self.model.parameters():
            p.requires_grad = False

        # --- [本次修复] 补回缺失的 visual_enc 定义 ---
        # meta_arch 需要用它来初始化 backbone
        self.frozen_clip_model = copy.deepcopy(self.model)
        self.visual_enc = self.model.visual
        # ----------------------------------------

        # 生成文本特征
        prompt = 'a photo of a {}'
        print(clsnames)
        with torch.no_grad():
            texts = [prompt.format(cls) for cls in clsnames]
            if "swin" in clip_encoder_name.lower() or "convnext" in clip_encoder_name.lower():
                text_inputs = self.tokenizer(texts).to(device)
            else:
                text_inputs = torch.cat([self.tokenizer(t) for t in texts]).to(device)
            self.text_features = self.model.encode_text(text_inputs).float()

        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.projection_global = nn.Linear(inshape, 512, bias=False)

    def forward(self, feat, gfeat=None):
        if feat.shape[-1] > 512:
            feat = self.projection(feat)

        feat = feat / feat.norm(dim=-1, keepdim=True)

        if gfeat is not None:
            if gfeat.shape[-1] > 512:
                gfeat = self.projection_global(gfeat)
            feat = feat - gfeat
            feat = feat / feat.norm(dim=-1, keepdim=True)

        scores = (100.0 * torch.matmul(feat, self.text_features.detach().T))
        scores = torch.cat([scores, torch.zeros(scores.shape[0], 1, device=scores.device)], 1)
        return scores

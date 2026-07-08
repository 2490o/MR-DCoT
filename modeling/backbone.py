import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as T

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class ClipRN101(Backbone):
    def __init__(self, cfg, clip_visual):
        super().__init__()
        self.enc = None
        self.unfreeze = cfg.MODEL.BACKBONE.UNFREEZE 
        self.proj = nn.Linear(512,512)
        self.global_proj = nn.Linear(512,512)
        self.use_proj = cfg.MODEL.USE_PROJ 
        

    def set_backbone_model(self,model):
        self.enc = model
        for name,val in self.enc.named_parameters():
            head = name.split('.')[0]
            if head not in self.unfreeze:
                val.requires_grad = False
            else:
                val.requires_grad = True
        
        self.backbone_unchanged = nn.Sequential(*self.enc.layer3[:19])

    def forward(self, image):
        x = image
        x = self.enc.relu1(self.enc.bn1(self.enc.conv1(x)))
        x = self.enc.relu2(self.enc.bn2(self.enc.conv2(x)))
        x = self.enc.relu3(self.enc.bn3(self.enc.conv3(x)))
        x = self.enc.avgpool(x)
        
        x = self.enc.layer1(x)
        x = self.enc.layer2(x)
        x = self.enc.layer3(x)
        return {"res4": x}


    def forward_stem(self, image):
        x = image
        x = self.enc.relu1(self.enc.bn1(self.enc.conv1(x)))
        x = self.enc.relu2(self.enc.bn2(self.enc.conv2(x)))
        x = self.enc.relu3(self.enc.bn3(self.enc.conv3(x)))
        x = self.enc.avgpool(x)
        return x

    def forward_l1(self, image):
        """First-layer feature map F1 (256 channels) for style disentanglement."""
        x = self.forward_stem(image)
        return self.enc.layer1(x)

    def forward_l2_to_res4(self, x):
        """Continue from merged L1 features through layer2-3."""
        x = self.enc.layer2(x)
        x = self.enc.layer3(x)
        return {"res4": x}

    def forward_l12(self, image):
        x = self.forward_stem(image)
        x = self.enc.layer1(x)
        x = self.enc.layer2(x)
        return x
     
    def forward_l3(self, x):
        
        x = self.enc.layer3(x)
        return {"res4": x}
 
    def output_shape(self):
        return {"res4": ShapeSpec(channels=1024, stride=16)}
    
    def forward_res5(self,x):
        #detectron used last resnet layer for roi heads
        return self.enc.layer4(x)

    def attention_global_pool(self,input):
        x = input
        x = self.enc.attnpool(x)
        return x


@BACKBONE_REGISTRY.register()
class ClipViT(Backbone):
    """
    注意：Detectron2 build_backbone 会以 (cfg, input_shape) 调用注册的 backbone。
    因此 __init__ 的第二个参数实际是 input_shape，而不是 CLIP visual encoder。
    CLIP visual encoder 会在 meta-arch 中通过 set_backbone_model() 绑定：
        self.backbone.set_backbone_model(self.roi_heads.box_predictor.cls_score.visual_enc)
    """
    def __init__(self, cfg, input_shape=None):
        super().__init__()
        self.enc = None
        self.unfreeze = cfg.MODEL.BACKBONE.UNFREEZE

        # build 阶段 RPN 需要 output_shape()，这里必须给出非 None 的 channels/stride
        # 你在 yaml 里已经把 RES2_OUT_CHANNELS 调成 96，使得 96*8=768 对应 ViT-B/16 width
        self.out_channels = int(cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * 8)  # ViT-B/16: 768

        # stride（patch size）根据 encoder name 推断：ViT-B/16 取 16，ViT-B/32 取 32
        enc_name = str(getattr(cfg.MODEL, "CLIP_IMAGE_ENCODER_NAME", "ViT-B/16"))
        self.patch_size = 16 if "16" in enc_name else 32

    def set_backbone_model(self, model):
        """在 meta-arch 初始化后调用，用 CLIP visual encoder 绑定实际 backbone。"""
        self.enc = model

        # 冻结/解冻逻辑：保持你原来的 substring match 方案
        for name, val in self.enc.named_parameters():
            val.requires_grad = False
            for freeze_key in self.unfreeze:
                if freeze_key in name:
                    val.requires_grad = True
                    break

        # 绑定后，若模型里能读到 conv1 的 stride/out_channels，则用真实值覆盖 cfg 推断值
        #（这一步不是必须，但更稳健，适配 ViT-L/14 等）
        if hasattr(self.enc, "conv1"):
            if hasattr(self.enc.conv1, "out_channels"):
                self.out_channels = int(self.enc.conv1.out_channels)
            if hasattr(self.enc.conv1, "stride"):
                stride = self.enc.conv1.stride
                self.patch_size = int(stride[0] if isinstance(stride, (tuple, list)) else stride)

    def forward(self, image):
        assert self.enc is not None, "ClipViT.enc is None. Did you forget to call set_backbone_model()?"

        x = self.enc.conv1(image)
        B, C, H, W = x.shape

        x = x.reshape(B, C, -1).permute(0, 2, 1)  # [B, L, C]

        cls_token = self.enc.class_embedding.to(x.dtype) + torch.zeros(
            B, 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)  # [B, L+1, C]

        pos_embed = self.enc.positional_embedding.to(x.dtype)
        if pos_embed.shape[0] != x.shape[1]:
            pos_embed_cls = pos_embed[0:1]        # (1, C)
            pos_embed_spatial = pos_embed[1:]     # (N-1, C)
            orig_size = int(math.sqrt(pos_embed_spatial.shape[0]))

            pos_embed_spatial = pos_embed_spatial.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            pos_embed_spatial = F.interpolate(
                pos_embed_spatial, size=(H, W), mode="bicubic", align_corners=False
            )
            pos_embed_spatial = pos_embed_spatial.permute(0, 2, 3, 1).reshape(1, -1, C)

            pos_embed_cls = pos_embed_cls.unsqueeze(0)  # (1, 1, C)
            pos_embed = torch.cat([pos_embed_cls, pos_embed_spatial], dim=1)

        x = x + pos_embed
        x = self.enc.ln_pre(x)

        # 跑除最后一层外的 transformer blocks
        x = x.permute(1, 0, 2)  # LND
        for i in range(len(self.enc.transformer.resblocks) - 1):
            x = self.enc.transformer.resblocks[i](x)
        x = x.permute(1, 0, 2)  # NLD

        spatial_out = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)
        return {"res4": spatial_out}

    def forward_res5(self, x):
        # ViT 路径：ROI 上不要跑 ViT block（避免分布漂移导致 AP=0）
        return x

    def attention_global_pool(self, input):
        assert self.enc is not None, "ClipViT.enc is None. Did you forget to call set_backbone_model()?"

        x = input.mean(dim=(2, 3))  # [B, width]

        if hasattr(self.enc, "ln_post") and self.enc.ln_post is not None:
            x = self.enc.ln_post(x)

        if hasattr(self.enc, "proj") and self.enc.proj is not None:
            proj = self.enc.proj.to(dtype=x.dtype, device=x.device)
            x = x @ proj  # [B, 512]

        return x

    def output_shape(self):
        return {"res4": ShapeSpec(channels=self.out_channels, stride=self.patch_size)}


@BACKBONE_REGISTRY.register()
class ClipSwin(Backbone):
    def __init__(self, cfg, clip_visual):
        super().__init__()
        self.enc = None
        self.unfreeze = cfg.MODEL.BACKBONE.UNFREEZE
        # Swin Base Stage 2 输出通道 512 (用于 res4)
        # Swin Base Stage 3 输出通道 1024 (用于 res5)
        self.out_channels = 512

    def set_backbone_model(self, model):
        self.enc = model
        # 冻结逻辑
        for name, val in self.enc.named_parameters():
            val.requires_grad = False
            for freeze_key in self.unfreeze:
                if freeze_key in name:
                    val.requires_grad = True
                    break

    def forward(self, image):
        # Swin Forward 流程
        x = image
        # OpenCLIP Swin 实现通常包含 patch_embed
        x = self.enc.patch_embed(x)
        if hasattr(self.enc, 'absolute_pos_embed') and self.enc.absolute_pos_embed is not None:
            x = x + self.enc.absolute_pos_embed
        x = self.enc.pos_drop(x)

        # 逐层运行 Stage 0, 1, 2
        # layers 是 nn.ModuleList
        for i, layer in enumerate(self.enc.layers):
            x = layer(x)
            if i == 2:  # Stage 2 (Stride 16) -> res4
                # Swin 输出通常是 (B, H, W, C)，需要转回 (B, C, H, W)
                out = x.permute(0, 3, 1, 2).contiguous()
                return {"res4": out}
        return {}

    def forward_res5(self, x):
        # 输入 x 是 ROIAlign 后的特征 (B, 512, 7, 7)
        # 转为 (B, 7, 7, 512)
        x = x.permute(0, 2, 3, 1).contiguous()

        # 运行 Stage 3 (最后一层)
        # 注意：如果 ROI 大小 (7x7) 小于 Window Size (7x7)，Swin 也能工作
        x = self.enc.layers[3](x)
        x = self.enc.norm(x)  # 最后的 LayerNorm

        # 输出 (B, 7, 7, 1024) -> 转回 (B, 1024, 7, 7)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def attention_global_pool(self, input):
        # input: (B, C, H, W)
        # Global Average Pooling
        x = input.mean(dim=[2, 3])  # (B, C)

        # 如果有 head (OpenCLIP Swin 通常有 head 映射到 512/768 维度)
        if hasattr(self.enc, 'head'):
            x = self.enc.head(x)

        return x

    def output_shape(self):
        return {"res4": ShapeSpec(channels=self.out_channels, stride=16)}


@BACKBONE_REGISTRY.register()
class ClipConvNeXt(Backbone):
    def __init__(self, cfg, clip_visual):
        super().__init__()
        self.enc = None
        self.unfreeze = cfg.MODEL.BACKBONE.UNFREEZE
        self.out_channels = 512

    def set_backbone_model(self, model):
        self.enc = model
        # 冻结/解冻逻辑
        # named_parameters 会递归遍历 .trunk，所以这里的逻辑不用变
        for name, val in self.enc.named_parameters():
            val.requires_grad = False
            for freeze_key in self.unfreeze:
                if freeze_key in name:
                    val.requires_grad = True
                    break

    def forward(self, image):
        # [关键修改] OpenCLIP 的模型被包裹在 .trunk 中
        # 1. Stem
        x = self.enc.trunk.stem(image)

        # 2. Run Stage 0, 1, 2
        for i in range(3):
            x = self.enc.trunk.stages[i](x)

        # Stage 2 输出即为 res4 (Stride 16, Channel 512)
        return {"res4": x}

    def forward_res5(self, x):
        # 输入 x 是 res4 特征 (B, 512, H, W)
        # 运行 Stage 3
        # [关键修改] 使用 .trunk.stages
        x = self.enc.trunk.stages[3](x)

        # OpenCLIP 的 ConvNeXt 可能有 norm_pre (LayerNorm)
        # [关键修改] 使用 .trunk.norm_pre
        if hasattr(self.enc.trunk, 'norm_pre'):
            x = self.enc.trunk.norm_pre(x)

        # 输出即为 res5 (Stride 32, Channel 1024)
        return x

    def attention_global_pool(self, input):
        # input: res5 feature (B, 1024, H, W)
        # Global Average Pooling
        x = input.mean(dim=[2, 3])  # (B, 1024)

        # 投影到 Embedding 维度
        # TimmModel 包装器通常把投影层放在 .head 中，所以这里直接用 self.enc.head 是对的
        # if hasattr(self.enc, 'head'):
        #     x = self.enc.head(x)

        return x

    def output_shape(self):
        return {"res4": ShapeSpec(channels=self.out_channels, stride=16)}

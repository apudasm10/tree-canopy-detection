import timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock


class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, p, c, names):
        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        p.extend([p6, p7])
        names.extend(["p6", "p7"])
        return p, names


class CustomFPN(nn.Module):
    def __init__(self, in_dims, out_dim, feature_module_names, vit_implementation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_dims = in_dims
        if len(self.in_dims) == 5:
            self.in_dims = in_dims[1:]
        self.out_dim = out_dim
        self.feature_module_names = feature_module_names
        if len(self.feature_module_names) == 5:
            self.feature_module_names = feature_module_names[1:]
        self.vit_implementation = vit_implementation

        if self.vit_implementation:
            # ViT Output is Stride 16. We need to reconstruct Strides 4, 8, 16, 32.
            embed_dim = in_dims[0]

            self.in_dims = [embed_dim, embed_dim, embed_dim, embed_dim]
            
            self.c2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2), # 1/16 -> 1/8
            nn.GroupNorm(32, embed_dim), # Normalization helps stability
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)  # 1/8 -> 1/4
            )

            # C3 (Target: 1/8 scale) -> Upsample 2x from 1/16
            self.c3 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
            )

            # C4 (Target: 1/16 scale) -> Identity (It is already 1/16)
            self.c4 = nn.Identity()

            # C5 (Target: 1/32 scale) -> Downsample 2x from 1/16
            self.c5 = nn.MaxPool2d(kernel_size=2, stride=2)

        extra_block = LastLevelP6P7(in_channels=self.out_dim, out_channels=self.out_dim)
        self.fpn = FeaturePyramidNetwork(self.in_dims, self.out_dim, extra_blocks=extra_block)

    def forward(self, inputs):

        if len(inputs) == 5:
            inputs = inputs[1:]
        if self.vit_implementation:
            last_feat = inputs[-1]
            c2 = self.c2(last_feat)
            c3 = self.c3(last_feat)
            c4 = self.c4(last_feat)
            c5 = self.c5(last_feat)
        else:
            c2 = inputs[0]
            c3 = inputs[1]
            c4 = inputs[2]
            c5 = inputs[3]

        return self.fpn({"p2": c2, "p3": c3, "p4": c4, "p5": c5})
    

class CustomBackbone(nn.Module):
    def __init__(self, model_name, freeze=True, unfreeze_pct=.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.is_swin = "swin" in model_name
        self.vit_implementation = "vit" in model_name

        if self.is_swin:
            self.backbone = timm.create_model(model_name, features_only=True, pretrained=True, img_size=None, strict_img_size=False)
        else:
            self.backbone = timm.create_model(model_name, features_only=True, pretrained=True)
        self.out_indices = self.backbone.feature_info.out_indices

        for params in self.backbone.parameters():
            params.requires_grad = False
        if not freeze:
            self.backbone = unfreeze_backbone_by_pct(self.backbone, unfreeze_pct=unfreeze_pct)

        self.in_channels_list = self.backbone.feature_info.channels()
        self.feature_module_names = self.backbone.feature_info.module_name()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval() 
        return self
    
    def forward(self, x):
        outputs = self.backbone(x)

        if self.is_swin:
            outputs = [o.permute(0, 3, 1, 2).contiguous() for o in outputs]

        return outputs
    
class CustomModelFPN(nn.Module):
    def __init__(self, backbone, fpn, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.fpn = fpn
        self.out_channels = fpn.out_dim

    def forward(self, x):
        out = self.backbone(x)
        out = self.fpn(out)

        return out
    
def unfreeze_backbone_by_pct(backbone, unfreeze_pct=.25):        
    total_params = sum(p.numel() for p in backbone.parameters())
    paramater_to_unfreeze = total_params*unfreeze_pct
    params = list(backbone.named_parameters())
    
    unfrozen_count = 0
    last_unfrozen_layer = None
    
    for name, param in reversed(params):
        if unfrozen_count >= paramater_to_unfreeze:
            break
            
        param.requires_grad = True
        unfrozen_count += param.numel()
        last_unfrozen_layer = name
        
    print(f"--> Unfrozen Backbone Params: {unfrozen_count / 1e6:.2f} M ({100 * unfrozen_count / total_params:.1f}%)")
    print(f"--> Deepest unfrozen layer: {last_unfrozen_layer}")
    
    return backbone
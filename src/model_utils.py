import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
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


class DINOV3FPN(nn.Module):
    def __init__(self, in_dims, out_dim, feature_module_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.feature_module_names = feature_module_names

        self.c2 = nn.Conv2d(in_dims[0], in_dims[0], 3, 2, 1)
        self.c3 = nn.Conv2d(in_dims[1], in_dims[1], 3, 2, 1)
        self.c4 = nn.Conv2d(in_dims[2], in_dims[2], 3, 2, 1)
        self.c5 = nn.Conv2d(in_dims[3], in_dims[3], 3, 2, 1)

        extra_block = LastLevelP6P7(in_channels=out_dim, out_channels=out_dim)
        self.fpn = FeaturePyramidNetwork(in_dims, out_dim, extra_blocks=extra_block)

    def forward(self, inputs):
        c2 = self.c2(inputs[0])
        c3 = self.c3(inputs[1])
        c4 = self.c4(inputs[2])
        c5 = self.c5(inputs[3])

        return self.fpn({"p2": c2, "p3": c3, "p4": c4, "p5": c5})
    

class DINOBackbone(nn.Module):
    def __init__(self, model_name, freeze, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

        self.backbone = timm.create_model(model_name, features_only=True, pretrained=True)
        self.out_indices = self.backbone.feature_info.out_indices
        self.backbone = self.backbone.eval()
        if freeze:
            for params in self.backbone.parameters():
                params.requires_grad = False

        self.in_channels_list = self.backbone.feature_info.channels()
        self.feature_module_names = self.backbone.feature_info.module_name()
    
    def forward(self, x):
        outputs = self.backbone(x)

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
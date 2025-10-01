
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights



class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, n_classes):
        super().__init__(
            ASPP(in_channels, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )


class DeepLabV3(nn.Module):
    def __init__(self, n_classes=5, pretrained=True):
        super().__init__()
        model = deeplabv3_resnet50(pretrained=pretrained, progress=True)
        model.classifier = DeepLabHead(2048, n_classes)
        self.model = model

    def forward(self, x, return_encoder_output=False):
        features = self.model.backbone(x)
        enc = features['out']

        out = self.model.classifier(enc)

        if return_encoder_output:
            return enc, out
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.out = nn.Conv2d(out_channels * 5, out_channels, 1)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.image_conv(self.image_pool(x))
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=False)
        out = torch.cat([
            self.atrous_block1(x),
            self.atrous_block6(x),
            self.atrous_block12(x),
            self.atrous_block18(x),
            image_features
        ], dim=1)
        return self.out(out)

class DeepLabV3EffB0(nn.Module):
    def __init__(self, n_classes=5, pretrained=True):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
        self.aspp = ASPP(320, 256)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )

    def extract_features(self, x):
        x = self.backbone._conv_stem(x)
        x = self.backbone._bn0(x)
        x = self.backbone._swish(x)
        for block in self.backbone._blocks:
            x = block(x)
        return x

    def forward(self, x, return_encoder_output=False):
        features = self.extract_features(x)
        x = self.aspp(features)
        x = self.classifier(x)
        out = F.interpolate(x, size=(x.shape[2] * 32, x.shape[3] * 32), mode='bilinear', align_corners=False)

        if return_encoder_output:
                    return features, out

        return out


class UNetEffB0(nn.Module):
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()
        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.encoder = base_model.features

        # Encoder stages
        self.enc0 = nn.Sequential(*self.encoder[:2])    # 512 -> 256
        self.enc1 = self.encoder[2]                     # 256 -> 128
        self.enc2 = self.encoder[3]                     # 128 -> 64
        self.enc3 = nn.Sequential(*self.encoder[4:6])   # 64 -> 32
        self.enc4 = nn.Sequential(*self.encoder[6:])    # 32 -> 16

        # Decoder blocks
        self.up4 = self.upsample_block(1280, 112)  # x3 = 112
        self.up3 = self.upsample_block(112, 40)    # x2 = 40
        self.up2 = self.upsample_block(40, 24)     # x1 = 24
        self.up1 = self.upsample_block(24, 16)     # x0 = 16
        self.final_up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)  # 256 -> 512

        self.final_conv = nn.Conv2d(16, n_classes, kernel_size=1)

    def upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, return_encoder_output=False):
        x0 = self.enc0(x)  # 256
        x1 = self.enc1(x0) # 128
        x2 = self.enc2(x1) # 64
        x3 = self.enc3(x2) # 32
        features = self.enc4(x3) # 16

        d4 = self.up4(features) + x3  # 32
        d3 = self.up3(d4) + x2  # 64
        d2 = self.up2(d3) + x1  # 128
        d1 = self.up1(d2) + x0  # 256
        d0 = self.final_up(d1)  # 512
        out = self.final_conv(d0)

        if return_encoder_output:
            return features, out

        return out



class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convs with ReLU (classic U-Net block).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """
    Downscaling: MaxPool -> DoubleConv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Upscaling: ConvTranspose2d -> concat skip -> DoubleConv
    Handles off-by-one shape mismatches via padding.
    """
    def __init__(self, in_ch: int, out_ch: int):
        """
        in_ch is the number of channels coming into the up block (from decoder).
        After upsample, it becomes in_ch//2, then concatenated with skip (same channels),
        and fused by DoubleConv to out_ch.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # If input H/W are odd, shapes can be off by 1; pad the upsampled map to match skip.
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffX != 0 or diffY != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution to num_classes.
    """
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Vanilla U-Net (Ronneberger et al., 2015), transpose-conv upsampling,
    no BatchNorm, no dropout, same-padding 3x3 convs.

    Args:
        in_channels: input channels (e.g., 3 for RGB)
        n_classes: segmentation classes (logits; use CrossEntropyLoss)
        base_channels: width multiplier (default 64, classic)
    """
    def __init__(self, n_classes: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels

        # Encoder
        self.inc   = DoubleConv(3, c)
        self.down1 = Down(c,   c * 2)
        self.down2 = Down(c*2, c * 4)
        self.down3 = Down(c*4, c * 8)
        self.down4 = Down(c*8, c * 16)

        # Decoder
        self.up1   = Up(c*16, c * 8)
        self.up2   = Up(c*8,  c * 4)
        self.up3   = Up(c*4,  c * 2)
        self.up4   = Up(c*2,  c)

        # Head
        self.outc  = OutConv(c, n_classes)

        self._init_weights()

    def _init_weights(self):
        # Kaiming init for convs (good default for ReLU)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_encoder_output=False) -> torch.Tensor:
        # Encoder path
        x1 = self.inc(x)     # C
        x2 = self.down1(x1)  # 2C
        x3 = self.down2(x2)  # 4C
        x4 = self.down3(x3)  # 8C
        features = self.down4(x4)  # 16C

        # Decoder path with skips
        x  = self.up1(features, x4)  # 8C
        x  = self.up2(x,  x3)  # 4C
        x  = self.up3(x,  x2)  # 2C
        x  = self.up4(x,  x1)  # C

        out = self.outc(x)  # [B, n_classes, H, W]

        if return_encoder_output:
            return features, out
        return out
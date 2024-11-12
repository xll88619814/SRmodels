''' network architecture for EDVR '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels,  out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNET1(nn.Module):
    def __init__(self, nf):
        super(UNET1, self).__init__()

        self.inc = DoubleConv(nf, nf)
        self.down1 = Down(nf, nf*2)
        self.down2 = Down(nf*2, nf*4)
        self.up1 = Up(nf*4, nf*2)
        self.up2 = Up(nf*2, nf)

    def forward(self, x):
        x_lr = x

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        out = self.up2(x, x1)
        
        return out

class UNET2(nn.Module):
    def __init__(self, nf):
        super(UNET2, self).__init__()

        self.inc = DoubleConv(nf, nf)
        self.down1 = Down(nf, nf*2)
        self.down2 = Down(nf*2, nf*4)
        self.down3 = Down(nf*4, nf*8)
        self.up1 = Up(nf*8, nf*4)
        self.up2 = Up(nf*4, nf*2)
        self.up3 = Up(nf*2, nf)

    def forward(self, x):
        x_lr = x

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        out = self.up3(x, x1)
        
        return out

class USRNET(nn.Module):
    def __init__(self, nf=16):
        super(USRNET, self).__init__()
        self.nf = nf
        self.firstconv = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.unet1 = UNET1(nf)

        self.conv = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

        self.mid_conv = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.unet2 = UNET2(nf)
        self.lastconv = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
  
    def forward(self, x):
        firstx = self.firstconv(x)
        unet1_out = self.unet1(firstx)
        mid_out1 = self.conv(F.interpolate(unet1_out, scale_factor=2, mode='bilinear', align_corners=True))
        mid_out = mid_out1 + F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        midconv = self.mid_conv(mid_out)

        unet2_out = self.unet2(midconv)
        lastout = self.lastconv(unet2_out)

        out = lastout + mid_out
        return out

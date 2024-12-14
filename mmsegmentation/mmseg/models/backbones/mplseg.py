# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import BACKBONES, build_backbone


class SToD(BaseModule):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        x = rearrange(x, 'b c (h h2) (w w2) -> b (h2 w2 c) h w', h2=self.bs, w2=self.bs)
        return x
    

class DToS(BaseModule):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        x = rearrange(x, 'b (h2 w2 c) h w -> b c (h h2) (w w2)', h2=self.bs, w2=self.bs)
        return x


class Encoder(BaseModule):
    def __init__(self, backbone_cfg):
        super(Encoder, self).__init__()
        self.bb = build_backbone(backbone_cfg)

    def forward(self, x):
        feat4, feat8, feat16, feat32 = self.bb(x)
        return feat4, feat8, feat16, feat32


class AFM(BaseModule):
    def __init__(self, in_ch, mid_ch, out_ch, upsample=False, norm_cfg=None):
        super(AFM, self).__init__()
    
        self.mp = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch//4, 1, padding=0, groups=1, bias=False),
            nn.InstanceNorm2d(mid_ch//4),
            nn.Sigmoid(),
            nn.Conv2d(mid_ch//4, mid_ch//4, 3, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(mid_ch//4),
            nn.Sigmoid(),
            nn.Conv2d(mid_ch//4, mid_ch, 1, padding=0, groups=1, bias=False),
            nn.InstanceNorm2d(mid_ch),
            nn.Sigmoid(),
        )

        self.pa = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch//4, 1, padding=0, groups=1, bias=False),
            nn.InstanceNorm2d(mid_ch//4),
            nn.Tanh(),
            nn.Conv2d(mid_ch//4, mid_ch//4, 3, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(mid_ch//4),
            nn.Tanh(),
            nn.Conv2d(mid_ch//4, mid_ch, 1, padding=0, groups=1, bias=False),
            nn.InstanceNorm2d(mid_ch),
            nn.Tanh(),
        ) 

        self.pw = ConvModule(
                        in_channels=mid_ch,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                        conv_cfg=None,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'))

        self.eps = 1e-3
        self.init_weight()


    def forward(self, x):
        fr = torch.fft.rfft2(x, norm='ortho')
        # fr = torch.fft.fftshift(fr, dim=(2, 3))

        mag = torch.abs(fr)
        pha = torch.angle(fr+self.eps)

        mag = mag*(1-self.mp(torch.sigmoid(torch.log(mag+self.eps))))
        pha = pha+self.pa(pha)

        fr = mag * (torch.e**(1j*pha))

        # fr = torch.fft.ifftshift(fr, dim=(2, 3))
        output = torch.fft.irfft2(fr, norm='ortho')
        output = F.relu(output, True)
        output = self.pw(output)
        return output
    
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=0)


@BACKBONES.register_module()
class MPLSeg(BaseModule):

    def __init__(self,
                 backbone_cfg,
                 backbone_channels=(256, 512, 1024, 2048),
                 mid_channels=(256, 512, 1024, 1024),
                 fuse_channels=(128, 256, 512, 1024),
                 out_indices=(0, 1, 2, 3),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None):

        super(MPLSeg, self).__init__(init_cfg=init_cfg)

        self.out_indices = out_indices

        self.enc = Encoder(backbone_cfg)
        
        self.up32to16   = AFM(backbone_channels[3],                  mid_channels[3], fuse_channels[3], upsample=True, norm_cfg=norm_cfg)
        self.up16to8    = AFM(backbone_channels[2]+fuse_channels[3], mid_channels[2], fuse_channels[2], upsample=True, norm_cfg=norm_cfg)
        self.up8to4     = AFM(backbone_channels[1]+fuse_channels[2], mid_channels[1], fuse_channels[1], upsample=True, norm_cfg=norm_cfg)
        self.up4tofinal = AFM(backbone_channels[0]+fuse_channels[1], mid_channels[0], fuse_channels[0], upsample=True, norm_cfg=norm_cfg)

    def forward(self, x):
        feat4, feat8, feat16, feat32= self.cp(x)
        up32 = self.up32to16(feat32)
        up32 =  F.interpolate(up32, size=feat16.shape[2:], mode='bilinear', align_corners=False)
        up16 = self.up16to8(torch.cat([up32, feat16], 1))
        up16 =  F.interpolate(up16, size=feat8.shape[2:], mode='bilinear', align_corners=False)
        up8 = self.up8to4(torch.cat([up16, feat8], 1))
        up8 =  F.interpolate(up8, size=feat4.shape[2:], mode='bilinear', align_corners=False)
        up4 = self.up4tofinal(torch.cat([up8, feat4], 1))

        outs = [up4, up8, up16, up32]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)


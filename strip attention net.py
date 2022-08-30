import torch

from ..builder import HEADS
from .fcn_head import FCNHead
import torch.nn as nn

from .aspp_head import ASPPHead,ASPPModule
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from .sa_module import SALayer
from .caf import CAFLayer

@HEADS.register_module()
class CCSPHead(ASPPHead):
    def __init__(self, c1_in_channels, c1_channels, dilations=(1, 6, 12, 18), **kwargs):
        super(CCSPHead, self).__init__(**kwargs)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.c1_bottleneck = ConvModule(
                    c1_in_channels,
                    c1_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
        self.sa = SALayer()
        self.caf = CAFLayer()

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        c1_output = self.sa(inputs[1])
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        aspp_outs = self.bottleneck(aspp_outs)
        output = self.caf(aspp_outs , c1_output)
        output = self.cls_seg(output)
        return output

# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss

from einops import rearrange, repeat


@LOSSES.register_module()
class PSLLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_psl',
                 avg_non_ignore=False,
                num_classes=19,
                
                 **kwargs):
        super(PSLLoss, self).__init__()
        self._loss_name = loss_name
        self.eps = 1e-5
        self.n_c = num_classes

    def forward(self,
                seg_logit,
                seg_label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):

        valid_mask = (seg_label != ignore_index).long()
        seg_logit = seg_logit.softmax(1)
        seg_label = F.one_hot(
            torch.clamp(seg_label, 0, self.n_c-1), num_classes=self.n_c).permute(0, 3, 1, 2)
        
        seg_label[:,self.n_c-1,:,:] *= valid_mask

        seg_label = torch.fft.rfft2(seg_label, norm='ortho')
        seg_logit = torch.fft.rfft2(seg_logit, norm='ortho')

        im_label = torch.angle(seg_label+self.eps)
        im_predict=torch.angle(seg_logit+self.eps)

        re_label = torch.abs(seg_label)

        mask = re_label>1

        im_label = im_label*mask
        im_predict = im_predict*mask
        delta = torch.abs(im_label-im_predict)
        delta = delta.sum()/(mask.sum()+self.eps)

        return delta

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

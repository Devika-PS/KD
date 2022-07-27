# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def distillation(pred, soft_label, Temperature, detach_target=True):
    p = F.log_softmax(pred/Temperature, dim=1)
    q = F.softmax(soft_label/Temperature, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (Temperature**2) / pred.shape[0]
    return l_kl


@LOSSES.register_module()
class DistillationLoss(nn.Module):
    def __init__(self, reduction='mean', Temperature=4.0):
        super(DistillationLoss, self).__init__()
        assert Temperature >= 1
        self.reduction = reduction
        self.Temperature = Temperature

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = distillation(
            pred,
            soft_label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            Temperature=self.Temperature)

        return loss_kd

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def attentionloss(x, y):
    return (at(x) - at(y)).pow(2).mean()

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

@LOSSES.register_module()
class atloss(nn.Module):
    def __init__(self):
        super(atloss, self).__init__()

    def forward(self, x, y):
        loss_at = attentionloss(x, y)
        return loss_at
# owner: Anshul Paigwar
# Forked from https://github.com/ClementPinard/SfmLearner-Pytorch/blob/e1b5b0de40fe212f7ba8807e3037fedd0fe4f12f/loss_functions.py#L71

import torch
import torch.nn as nn

'''
2021.1.18 增加loss from MIT paper
2021.2.15 focal loss
'''


class ClsLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=-100):
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def CrossEntropyLoss(self, input, target):
        n, _, _, _ = input.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='sum')
        criterion = criterion.cuda()
        loss = criterion(input, target.long())

        if self.reduction == 'mean':
            loss /= n

        return loss

    def FocalLoss(self, input, target, gamma=2, alpha=0.5):
        n, _, _, _ = input.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='sum')

        criterion = criterion.cuda()

        logpt = -criterion(input, target.long())
        pt = torch.exp(logpt)

        if alpha is not None:
            logpt *= alpha

        loss = -((1 - pt) ** gamma) * logpt

        if self.reduction == 'mean':
            loss /= n

        return loss


class SpatialSmoothLoss(torch.nn.Module):
    def __init__(self):
        super(SpatialSmoothLoss, self).__init__()

    def forward(self, pred_map):
        def gradient(pred):
            D_dy = pred[:, 1:] - pred[:, :-1]
            D_dx = pred[:, :, 1:] - pred[:, :, :-1]
            return D_dx, D_dy

        # pdb.set_trace()

        dx, dy = gradient(pred_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss = dx2.abs().mean(axis=(1, 2)) + dxdy.abs().mean(axis=(1, 2)) + dydx.abs().mean(
            axis=(1, 2)) + dy2.abs().mean(axis=(1, 2))
        return loss.mean()


class MaskedHuberLoss(torch.nn.Module):
    def __init__(self):
        super(MaskedHuberLoss, self).__init__()

    def forward(self, output, labels, mask):
        lossHuber = nn.SmoothL1Loss(reduction="none").cuda()
        l = lossHuber(output * mask, labels * mask)  # (B,100,100)
        l = l.sum(dim=(1, 2))
        mask = mask.sum(dim=(1, 2))
        l = l / mask
        return l.mean()

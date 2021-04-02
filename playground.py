import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

class DWConv(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=in_plane,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_plane)
        self.point_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=out_plane,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


def deubg_dw():
    DW_model = DWConv(3, 32)
    x = torch.rand((32, 3, 320, 320))
    out = DW_model(x)
    print(out.shape)

if __name__ == '__main__':
    deubg_dw()


arr1= torch.tensor([[0,1,2],[1,0,1],[0,1,1]])
arr2=torch.rand((3,3))

print(arr2,arr1)

# print(arr2.view(-1,6))
# mask=arr1.clone()
# mask[mask!=0]=1
# print(mask)
# arr2=arr2.mul(mask)
# print(arr2)
# arr1=arr1.view(-1,arr1.size(-1))

import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1, with_logits=False, reduce='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.with_logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce == 'mean':
            return torch.mean(focal_loss)
        elif self.reduce == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=1, eps=1e-7, with_logits=True, ignore_index=-100, reduction='mean', smooth_eps=None):
        super().__init__()

        assert reduction in ['none', 'mean', 'sum'], 'FocalLoss: reduction must be one of [\'none\', \'mean\', \'sum\']'

        self.gamma = gamma
        self.eps = eps
        self.with_logits = with_logits
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        return focal_loss(input, target, self.gamma, self.eps, self.with_logits, self.ignore_index, self.reduction,
                          smooth_eps=self.smooth_eps)

def focal_loss(input, target, gamma=1, eps=1e-7, with_logits=True, ignore_index=-100, reduction='mean',
               smooth_eps=None):

    smooth_eps = smooth_eps or 0

    # make target
    y = F.one_hot(target, input.size(-1))

    # apply label smoothing according to target = [eps/K, eps/K, ..., (1-eps) + eps/K, eps/K, eps/K, ...]
    if smooth_eps > 0:
        y = y * (1 - smooth_eps) + smooth_eps/y.size(-1)

    if with_logits:
        pt = F.softmax(input, dim=-1)
    else:
        pt = input

    pt = pt.clamp(eps, 1. - eps)  # a hack-y way to prevent taking the log of a zero, because we might be dealing with
                                  # probabilities directly.

    loss = -y * torch.log(pt)  # cross entropy
    loss *= (1 - pt) ** gamma  # focal loss factor
    loss = torch.sum(loss, dim=-1)

    # mask the logits so that values at indices which are equal to ignore_index are ignored
    loss = loss[target != ignore_index]

    # batch reduction
    if reduction == 'mean':
        return torch.mean(loss, dim=-1)
    elif reduction == 'sum':
        return torch.sum(loss, dim=-1)
    else:  # 'none'
        return loss


if __name__ == '__main__':
    torch.manual_seed(0)

    # confirm that with_logits works as intended
    fl = FocalLoss(gamma=1, with_logits=True, reduction='none')
    fl2 = FocalLoss(gamma=1, with_logits=False, reduction='none')

    input = torch.randn((5, 10))
    pt = F.softmax(input, dim=-1)
    target = torch.randint(0, 9, (5,))

    print(fl(input, target))
    print(fl2(pt, target))

    # confirm that FocalLoss(gamma=0) == nn.CrossEntropyLoss
    fl_nogamma = FocalLoss(gamma=0, with_logits=True, reduction='mean')
    ce = nn.CrossEntropyLoss(reduction='mean')

    print(fl_nogamma(input, target))
    print(ce(input, target))

    # test FocalLoss with ignore_index
    target2 = torch.tensor([1, 8, 5, 1, 9])
    print(focal_loss(input, target2, ignore_index=1, reduction='none'))
    print(focal_loss(input, target2, ignore_index=1, reduction='mean'))

    # test FocalLoss with label smoothing
    fl_smooth = FocalLoss(gamma=1, with_logits=True, smooth_eps=0.1, reduction='none')
    print(fl_smooth(input, target))


loss= Masked_focal_loss()
loss(arr2,arr1)
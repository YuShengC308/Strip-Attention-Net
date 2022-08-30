import torch
from torch import nn
from mmseg.ops import resize

class CAFLayer(nn.Module):
    '''Channel Attention Fusion Module'''
    def __init__(self, reduction=16):
        super(CAFLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 512 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512 // reduction, 512, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        b, c, _, _ = x.size()
        m = self.avg_pool(x).view(b, c)
        n = self.max_pool(x).view(b, c)
        m = self.fc(m).view(b, c, 1, 1)
        n = self.fc(n).view(b, c, 1, 1)
        p = m + n
        p = y * p.expand_as(y)
        return p
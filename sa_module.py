import torch
from torch import nn

class SALayer(nn.Module):
    '''Strip Attention Module'''
    def __init__(self):
        super(SALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        b, c, h, w = x.size()
        y = x.permute(0, 3, 2, 1)
        output_1 = self.avg_pool(y).view(b, w)
        output_1 = nn.Sequential(
            nn.Linear(w, w // 2, bias=False).cuda(),
            nn.ReLU(inplace=True),
            nn.Linear(w // 2, w, bias=False).cuda(),
            nn.Sigmoid()
        )(output_1).view(b, w, 1, 1)
        output_1 = output_1.expand_as(y)
        output_1 = output_1.permute(0, 3, 2, 1)
        x_1 = x * output_1

        z = x.permute(0, 2, 3, 1)
        output_2 = self.avg_pool(z).view(b, h)
        output_2 = nn.Sequential(
            nn.Linear(h, h // 2, bias=False).cuda(),
            nn.ReLU(inplace=True),
            nn.Linear(h // 2, h, bias=False).cuda(),
            nn.Sigmoid()
        )(output_2).view(b, h, 1, 1)
        output_2 = output_2.expand_as(z)
        output_2 = output_2.permute(0, 3, 1, 2)
        x_2 = x * output_2
        output = x_1 + x_2
        return output
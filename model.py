

import torch
import torch.nn as nn
from torchsummary import summary

import math


class Con2D(nn.Module):
    def __init__(self, in_c, out_c, k_size, is_bn=True):
        super(Con2D, self).__init__()

        if is_bn:
            self.sequential = nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, k_size, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        else:
            self.sequential = nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, k_size, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.sequential(x)




class U_net_pp(nn.Module):
    def __init__(self, in_channles, out_channels):
        super(U_net_pp, self).__init__()
        nb_filter = [32, 64, 128, 256, 512]



    def forward(self, x):


        return x


if __name__ == '__main__':
    from torchsummary import summary
    u_net_pp = U_net_pp(12, 1)
    s = summary(u_net_pp, (12, 40, 40), device='cpu')
    print(s)
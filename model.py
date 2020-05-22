

import torch
import torch.nn as nn
from torchsummary import summary

import math


class Con2D(nn.Module):
    def __init__(self, in_c, out_c, k_size, padding=1, is_bn=True):
        super(Con2D, self).__init__()

        if is_bn:
            self.sequential = nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size, padding=padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, k_size, padding=padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        else:
            self.sequential = nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, k_size, padding=padding),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.sequential(x)


class U_net_pp(nn.Module):
    def __init__(self, in_channles, out_channels):
        super(U_net_pp, self).__init__()

        filters = [32, 64, 128, 256, 512]
        filter_size = 3
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = Con2D(in_channles, filters[0], filter_size)
        self.conv1_0 = Con2D(filters[0], filters[1], filter_size)
        self.conv2_0 = Con2D(filters[1], filters[2], filter_size)
        self.conv3_0 = Con2D(filters[2], filters[3], filter_size)
        # self.conv4_0 = Con2D(filters[3], filters[4], filter_size)

        self.conv0_1 = Con2D(filters[0]+filters[1], filters[0], filter_size)
        self.conv1_1 = Con2D(filters[1]+filters[2], filters[1], filter_size)
        self.conv2_1 = Con2D(filters[2]+filters[3], filters[2], filter_size)
        # self.conv3_1 = Con2D(filters[3]+filters[4], filters[3], filter_size)

        self.conv0_2 = Con2D(filters[0]*2+filters[1], filters[0], filter_size)
        self.conv1_2 = Con2D(filters[1]*2+filters[2], filters[1], filter_size)
        # self.conv2_2 = Con2D(filters[2]*2+filters[3], filters[2], filter_size)

        self.conv0_3 = Con2D(filters[0]*3+filters[1], filters[0], filter_size)
        # self.conv1_3 = Con2D(filters[1]*3+filters[2], filters[1], filter_size)

        # self.conv0_4 = Con2D(filters[0]*4+filters[1], filters[0], filter_size)

        self.final = Con2D(filters[0], out_channels, 1, padding=0)
        self.init_weights()

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        # x4_0 = self.conv4_0(self.pool(x3_0))
        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        return self.final(x0_3)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from torchsummary import summary
    u_net_pp = U_net_pp(12, 4)
    s = summary(u_net_pp, (12, 40, 40), device='cpu')
    print(s)


import torch
import torch.nn as nn
import math

class Con2D(nn.Module):
    def __init__(self, in_c, out_c, k_size=3, padding=1, is_bn=True):
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


class Downsample(nn.Module):
    def __init__(self, in_c, out_c, ratio):
        """
            ratio : if 1/2 downsample, ratio=2 -> 1 time conv operation, ratio must be >= 2
        """
        super(Downsample, self).__init__()
        m = []
        r = int(math.log(ratio//2, 2))

        m.append(nn.Conv2d(in_c, out_c, 3, stride=2, padding=1))
        for i in range(r):
            m.append(nn.Conv2d(out_c, out_c, 3, stride=2, padding=1))

        self.sequential = nn.Sequential(*m)

    def forward(self, x):
        return self.sequential(x)


class Upsample_HR(nn.Module):
    def __init__(self, in_c, out_c, ratio):
        super(Upsample_HR, self).__init__()
        self.sequential = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=ratio),
            nn.Conv2d(in_c, out_c, 1)
            )

    def forward(self, x):
        return self.sequential(x)


class BlockConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockConv, self).__init__()
        modules = [Con2D(in_c, out_c),
                   Con2D(out_c, out_c),
                   Con2D(out_c, out_c),
                   Con2D(out_c, out_c)]
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class Branch(nn.Module):
    def __init__(self, in_c, out_ratios):
        super(Branch, self).__init__()
        self.m_list = nn.ModuleList()
        self.make_branch(in_c, out_ratios)

    def make_branch(self, in_c, out_ratios):
        """
            out_ratios : dictionary for up and down , len(out_ratios) is number of out branches
                i.e)  {'up': 2, 'up':1, 'down':2}
        """
        m = self.m_list
        for ratio in out_ratios:
            op, r = ratio.popitem()
            if op == 'up':
                if r == 1:
                    m.append(Con2D(in_c, in_c))
                else:
                    m.append(Upsample_HR(in_c, int(in_c * (1 / r)), r))
            elif op == 'down':
                m.append(Downsample(in_c, int(in_c * r), r))
            else:
                raise ValueError('Sampling operation is not correct.')

    def forward(self, x):
        values = [m(x) for m in self.m_list]
        return values


class HRNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HRNet, self).__init__()
        c_ = [64, 128, 256, 512]
        # Merge operation is in forward step

        self.block1_1 = BlockConv(in_channels, c_[0])
        self.branch1_1 = Branch(c_[0], [{'up': 1}, {'down': 2}])

        self.block2_1 = BlockConv(c_[0], c_[0])
        self.block2_2 = BlockConv(c_[1], c_[1])
        self.branch2_1 = Branch(c_[0], [{'up': 1}, {'down': 2}, {'down': 4}])
        self.branch2_2 = Branch(c_[1], [{'up': 2}, {'up': 1}, {'down': 2}])

        self.block3_1 = BlockConv(c_[0], c_[0])
        self.block3_2 = BlockConv(c_[1], c_[1])
        self.block3_3 = BlockConv(c_[2], c_[2])
        self.branch3_1 = Branch(c_[0], [{'up': 1}, {'down': 2},
                                        {'down': 4}, {'down': 8}])
        self.branch3_2 = Branch(c_[1], [{'up': 2}, {'up': 1},
                                        {'down': 2}, {'down': 4}])
        self.branch3_3 = Branch(c_[2], [{'up': 4}, {'up': 2},
                                        {'up': 1}, {'down': 2}])

        self.block4_1 = BlockConv(c_[0], c_[0])
        self.block4_2 = BlockConv(c_[1], c_[1])
        self.block4_3 = BlockConv(c_[2], c_[2])
        self.block4_4 = BlockConv(c_[3], c_[3])
        self.branch4_1 = Branch(c_[0], [{'up': 1}, {'down': 2},
                                        {'down': 4}, {'down': 8}])
        self.branch4_2 = Branch(c_[1], [{'up': 2}, {'up': 1},
                                        {'down': 2}, {'down': 4}])
        self.branch4_3 = Branch(c_[2], [{'up': 4}, {'up': 2},
                                        {'up': 1}, {'down': 2}])
        self.branch4_4 = Branch(c_[3], [{'up': 8}, {'up': 4},
                                        {'up': 2}, {'up': 1}])

        self.f_up1 = Upsample_HR(c_[1], c_[0], 2)
        self.f_up2 = Upsample_HR(c_[2], c_[0], 4)
        self.f_up3 = Upsample_HR(c_[3], c_[0], 8)

        self.init_weights()

    def forward(self, x):
        x = self.block1_1(x)
        b1 = self.branch1_1(x)

        x11 = self.block2_1(b1[0])
        x12 = self.block2_2(b1[1])

        b21 = self.branch2_1(x11)
        b22 = self.branch2_2(x12)

        x11 = self.block3_1(b21[0]+b22[0])
        x12 = self.block3_2(b21[1]+b22[1])
        x13 = self.block3_3(b21[2]+b22[2])

        b31 = self.branch3_1(x11)
        b32 = self.branch3_2(x12)
        b33 = self.branch3_3(x13)

        x11 = self.block4_1(b31[0] + b32[0] + b33[0])
        x12 = self.block4_2(b31[1] + b32[1] + b33[1])
        x13 = self.block4_3(b31[2] + b32[2] + b33[2])
        x14 = self.block4_4(b31[3] + b32[3] + b33[3])

        b41 = self.branch4_1(x11)
        b42 = self.branch4_2(x12)
        b43 = self.branch4_3(x13)
        b44 = self.branch4_4(x14)

        f1 = b41[0] + b42[0] + b43[0] + b44[0]
        f2 = self.f_up1(b41[1] + b42[1] + b43[1] + b44[1])
        f3 = self.f_up2(b41[2] + b42[2] + b43[2] + b44[2])
        f4 = self.f_up3(b41[3] + b42[3] + b43[3] + b44[3])

        return torch.cat([f1, f2, f3, f4], dim=1)

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
    hrnet = HRNet(12,4)
    print(hrnet)

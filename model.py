

import torch
import torch.nn as nn


class Deeplab_v3_resnet101_pretraind(nn.Module):
    def __init__(self, in_channels, out_channels, is_freeze=True, pretrained=True):
        super(Deeplab_v3_resnet101_pretraind, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=pretrained)
        if pretrained:
            self.model.requires_grad_(not is_freeze)
        else:
            self.model.requires_grad_(is_freeze)
        self.model.backbone.conv1 = nn.Conv2d(in_channels, 64, 7, padding=3, bias=False)
        self.model.backbone.conv1.requires_grad_(is_freeze)
        self.model.classifier[4] = nn.Conv2d(256, out_channels, 3, padding=1, bias=False)
        self.model.classifier[4].requires_grad_(is_freeze)
        self.model.aux_classifier = None

    def forward(self, x):
        return self.model(x)['out']


if __name__ == '__main__':
    m = Deeplab_v3_resnet101_pretraind(9, 12)

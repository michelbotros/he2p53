import torch.nn as nn


def conv_block(in_channels, out_channels):
    conv_block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return conv_block


class CBR5(nn.Module):
    """
    CBR-5: from Transfusion: Understanding Transfer Learning for Medical Imaging
    """
    def __init__(self):
        super().__init__()

        # feature extraction
        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128)
        self.conv4 = conv_block(128, 256)
        self.conv5 = conv_block(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d((2, 2), stride=2)

        # classification head
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        if x.size(-1) > 4:
             x = self.maxpool(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out
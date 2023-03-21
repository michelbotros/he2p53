import torch.nn as nn
import torchvision


def ResNet50(in_channels=3, out_features=1, weights='ResNet50_Weights.DEFAULT'):
    resnet = torchvision.models.resnet50(pretrained=weights)
    resnet.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(in_features=2048, out_features=out_features)
    return resnet
"""CIFAR-sized classifiers exposing the vector used by the final linear layer."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs)

    def forward(self, x):
        shortcut = x
        if self.stride == 2:
            half = x.shape[1] // 2
            shortcut = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, half, half))
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + shortcut)


class Classifier(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers = []
        if arch == "resnet32":
            layers.extend([nn.Conv2d(3, 16, 3, padding=1, bias=False),
                           nn.BatchNorm2d(16), nn.ReLU()])
            width = 16
            for channels in (16, 32, 64):
                for _ in range(5):
                    layers.append(ResidualBlock(width, channels, 1 if width == channels else 2))
                    width = channels
        elif arch == "vgg19":
            width = 3
            for channels, count in ((64, 2), (128, 2), (256, 4), (512, 4), (512, 4)):
                for _ in range(count):
                    layers.extend([nn.Conv2d(width, channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(channels), nn.ReLU()])
                    width = channels
                layers.append(nn.MaxPool2d(2))
        else:
            raise ValueError(f"Unknown architecture {arch}")
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(width, 10)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x, return_features=False):
        h = self.features(x).mean(dim=(2, 3))
        logits = self.fc(h)
        return (logits, h) if return_features else logits


def eligible(model):
    return {name + ".weight": layer for name, layer in model.named_modules()
            if isinstance(layer, (nn.Conv2d, nn.Linear))}

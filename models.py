import torch
import torch.nn as nn
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_bn=True, dropout_rate=None):
        super(ConvBlock, self).__init__()

        pad = kernel_size // 2

        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))

        layers.append(nn.ReLU(inplace=True))

        if dropout_rate:
            layers.append(nn.Dropout(dropout_rate))

        self.layers = nn.Sequential(*layers)

    def forward(self, in_tensor):
        return self.layers(in_tensor)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True, dropout_rate=None):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_ch, out_ch, 3, use_bn, dropout_rate),
            ConvBlock(out_ch, out_ch, 3, use_bn, dropout_rate),
            nn.Conv2d(out_ch, in_ch, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_ch)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, in_tensor):
        residual = in_tensor

        features = self.layers(in_tensor)
        out = residual + features

        return self.relu(out)


class Classifier(nn.Module):
    def __init__(self, use_bn=True, dropout_rate=None):
        super(Classifier, self).__init__()

        layers = []
        layers.append(ConvBlock(in_ch=1, out_ch=16, use_bn=use_bn, dropout_rate=dropout_rate))
        layers.append(nn.MaxPool2d(2))

        layers.append(ConvBlock(in_ch=16, out_ch=64, use_bn=use_bn, dropout_rate=dropout_rate))
        layers.append(ResidualBlock(in_ch=64, out_ch=64, use_bn=use_bn, dropout_rate=dropout_rate))
        layers.append(nn.MaxPool2d(2))

        layers.append(ConvBlock(in_ch=64, out_ch=96, use_bn=use_bn, dropout_rate=dropout_rate))
        layers.append(ResidualBlock(in_ch=96, out_ch=96, use_bn=use_bn, dropout_rate=dropout_rate))
        layers.append(nn.MaxPool2d(2))

        layers.append(ConvBlock(in_ch=96, out_ch=128, use_bn=use_bn, dropout_rate=dropout_rate))
        layers.append(ResidualBlock(in_ch=128, out_ch=128, use_bn=use_bn, dropout_rate=dropout_rate))
        layers.append(ResidualBlock(in_ch=128, out_ch=128, use_bn=use_bn, dropout_rate=dropout_rate))
        layers.append(nn.MaxPool2d(2))

        layers.append(ConvBlock(in_ch=128, out_ch=64, use_bn=use_bn, dropout_rate=dropout_rate))
        layers.append(ResidualBlock(in_ch=64, out_ch=64, use_bn=use_bn, dropout_rate=dropout_rate))
        layers.append(ConvBlock(in_ch=64, out_ch=128, use_bn=use_bn, dropout_rate=dropout_rate))
        layers.append(nn.MaxPool2d(2))

        self.features = nn.Sequential(*layers)

        out_ch = 128
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(out_ch, out_ch // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch // 4, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, in_img):
        features = self.features(in_img.float())
        probs = self.fc(features)

        return self.sigmoid(probs)


def get_model(backbone, use_bn=True, dropout_rate=None, train=False):
    if backbone == 'classifier':
        model = Classifier(use_bn=use_bn, dropout_rate=dropout_rate)

    else:
        model = None

    return model

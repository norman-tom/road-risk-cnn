import torchvision.models as models

import torch.nn as nn


class RESNET18:
    def __new__(cls, n_in_channels, n_classes):
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(
            n_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Linear(512, n_classes)
        return model

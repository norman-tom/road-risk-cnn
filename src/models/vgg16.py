import torchvision.models as models

import torch.nn as nn


class VGG16:
    def __new__(cls, n_in_channels, n_classes):
        model = models.vgg16(pretrained=False)
        model.features[0] = nn.Conv2d(n_in_channels, 64, kernel_size=3, padding=1)
        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        model.classifier[0] = nn.Linear(512, 256)
        model.classifier[3] = nn.Linear(256, 64)
        model.classifier[6] = nn.Linear(64, n_classes)
        return model

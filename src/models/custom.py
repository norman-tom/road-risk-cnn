import torch
import torch.nn as nn


class CustomNet(nn.Module):
    def __init__(self, n_in_channels, n_classes):
        super().__init__()
        

        def block_outer(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        outer_blocks = [n_in_channels, 32, 32, 64, 64, 128]
        self.outer = nn.Sequential(
            *[
                block_outer(in_channels, out_channels)
                for in_channels, out_channels in zip(outer_blocks, outer_blocks[1:])
            ]
        )

        def block_inner(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        inner_blocks = [n_in_channels, 32, 64, 128]
        self.inner = nn.Sequential(
            *[
                block_inner(in_channels, out_channels)
                for in_channels, out_channels in zip(inner_blocks, inner_blocks[1:])
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(3200, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x_outer = self.outer(x)
        x_inner = self.inner(x[:, :, 32:64, 32:64])
        x_outer = x_outer.view(x_outer.size(0), -1)
        x_inner = x_inner.view(x_inner.size(0), -1)
        x = torch.cat((x_outer, x_inner), dim=1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

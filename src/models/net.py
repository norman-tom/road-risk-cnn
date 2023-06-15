import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_in_channels, architecture, n_classes):
        super().__init__()

        block_channels, l1_l2 = architecture
        assert len(block_channels) <= 5

        block_channels = [n_in_channels] + block_channels

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        conv_blocks = [
            block(in_channels, out_channels)
            for in_channels, out_channels in zip(block_channels, block_channels[1:])
        ]
        self.features = nn.Sequential(*conv_blocks)

        l1, l2 = l1_l2
        n_spatial = int((96 / (2 ** (len(block_channels) - 1))))
        self.classifier = nn.Sequential(
            nn.Linear(block_channels[-1] * n_spatial**2, l1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(l1, l2),
            nn.ReLU(inplace=True),
            nn.Linear(l2, n_classes),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

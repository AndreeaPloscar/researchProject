import torch
import torch.nn as nn


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(Unit, self).__init__()
        self.conv = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.relu(output)
        return output


class SimpleNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleNet, self).__init__()

        self.unit2 = Unit(in_channels=120, out_channels=128, kernel_size=3)
        self.unit3 = Unit(in_channels=128, out_channels=256, kernel_size=3)
        self.unit5 = Unit(in_channels=256, out_channels=512, kernel_size=3)
        self.unit6 = Unit(in_channels=512, out_channels=256, kernel_size=3)
        self.unit7 = Unit(in_channels=256, out_channels=128, kernel_size=3)
        self.unit8 = Unit(in_channels=128, out_channels=64, kernel_size=3)
        self.unit9 = Unit(in_channels=64, out_channels=num_classes, kernel_size=3)

        self.net = nn.Sequential(self.unit2, self.unit3,
                                 self.unit5, self.unit6, self.unit7,
                                 self.unit8, self.unit9)

    def forward(self, input):
        output = self.net(input)
        return output

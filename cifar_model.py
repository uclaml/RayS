import torch.nn as nn


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(3200, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 3
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(64, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.Conv2d(128, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

from torch import nn

from ..base import *


class MLPConfig(TextEncoderConfig):
    def __init__(self,
                 mode='supervised',
                 class_weight=None,
                 num_classes=2,
                 d_hidden=1024):
        super().__init__(mode=mode,
                         class_weight=class_weight,
                         num_classes=num_classes,
                         d_hidden=d_hidden)


class MLP(nn.Module):
    def __init__(self, input_dim=840, d_hidden=128, output_dim=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, d_hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(d_hidden, output_dim)  # positive, negative, neutral
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

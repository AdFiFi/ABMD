import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        ds = [1, 2, 4, 8, 16]
        ps = [16, 32, 64, 128, 256]
        self.conv_list = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, 10, (1, 33), stride=(1, 2), dilation=(1, d), padding=(0, p)),
                          nn.BatchNorm2d(10),
                          nn.ReLU()
                          )
            for d, p in zip(ds, ps)])

    def forward(self, hidden_state):
        hidden_states = []
        for conv in self.conv_list:
            hidden_states.append(conv(hidden_state))
        hidden_state = torch.concat(hidden_states, dim=1)
        return hidden_state


class SpatialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        ks = [128, 64, 32, 16]
        ps = [63, 31, 15, 7]
        inp_channels = 50
        self.conv_list = nn.ModuleList([
            nn.Sequential(nn.Conv2d(inp_channels, inp_channels, (k, 1), stride=(2, 1), dilation=(1, 1), padding=(p, 0)),
                          nn.BatchNorm2d(inp_channels),
                          nn.ReLU()
                          )
            for k, p in zip(ks, ps)])

    def forward(self, hidden_state):
        hidden_states = []
        for conv in self.conv_list:
            hidden_states.append(conv(hidden_state))
        hidden_state = torch.concat(hidden_states, dim=1)
        return hidden_state


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        inp_channels = 200
        self.conv_list = nn.ModuleList([
            nn.Sequential(nn.Conv2d(inp_channels, inp_channels, (3, 3), stride=(2, 2), dilation=(1, 1), padding=(1, 1)),
                          nn.BatchNorm2d(inp_channels),
                          nn.ReLU()
                          )
            for _ in range(4)])

    def forward(self, hidden_state):
        for conv in self.conv_list:
            hidden_state = conv(hidden_state)
        return hidden_state

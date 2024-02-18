import torch
import torch.nn.functional as F
from torch import nn
from ..base import *


"""
A Reproduction of BrainNetCNN in:
BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment
J. Kawahara, C. J. Brown, S. P. Miller, B. G. Booth, V. Chau, R. E. Grunau, et al.
NeuroImage 2017 Vol. 146 Pages 1038-1049
"""


class E2EBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNNConfig(BrainEncoderConfig):
    def __init__(self,
                 node_size,
                 num_classes):
        super(BrainNetCNNConfig, self).__init__(node_size=node_size,
                                                d_model=256,
                                                num_classes=num_classes)


class BrainNetCNN(nn.Module):
    def __init__(self, config: BrainNetCNNConfig):
        super().__init__()
        self.in_planes = 1
        self.d = config.node_size

        self.e2econv1 = E2EBlock(1, 32, config.node_size, bias=True)
        self.e2econv2 = E2EBlock(32, 64, config.node_size, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, config.d_model, (self.d, 1))
        self.dense1 = torch.nn.Linear(config.d_model, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, config.num_classes)

    def forward(self, node_feature: torch.tensor, **kwargs):
        node_feature = node_feature.unsqueeze(dim=1)
        hidden_state = F.leaky_relu(self.e2econv1(node_feature), negative_slope=0.33)
        hidden_state = F.leaky_relu(self.e2econv2(hidden_state), negative_slope=0.33)
        hidden_state = F.leaky_relu(self.E2N(hidden_state), negative_slope=0.33)
        hidden_state = F.dropout(F.leaky_relu(
            self.N2G(hidden_state), negative_slope=0.33), p=0.5)
        hidden_state = hidden_state.view(hidden_state.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(hidden_state), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(
            self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return BrainEncoderOutputs(logits=out,
                                   brain_state=hidden_state)

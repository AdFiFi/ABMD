import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from ..base import *

"""
Graphormer is a reproduction in:
Do Transformers Really Perform Bad for Graph Representation?
C. Ying, T. Cai, S. Luo, S. Zheng, G. Ke, D. He, et al.
Neural Information Processing Systems 2021??????????????????????????????????????????????????????????????????????
"""


class TransformerConfig(BrainEncoderConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 readout='concat',
                 num_layers=2,
                 num_classes=2,
                 d_hidden=1024):
        super(TransformerConfig, self).__init__(node_size=node_size,
                                                node_feature_size=node_feature_size,
                                                num_classes=num_classes,
                                                d_hidden=d_hidden)
        self.readout = readout
        self.num_layers = num_layers
        self.d_model = 8 * node_size


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.attention_list = nn.ModuleList()
        self.readout = config.readout
        self.node_size = config.node_size

        for _ in range(config.num_layers):
            self.attention_list.append(
                TransformerEncoderLayer(d_model=config.node_feature_size, nhead=4,
                                        dim_feedforward=config.d_hidden,
                                        batch_first=True)
            )

        final_dim = config.node_feature_size

        if self.readout == "concat":
            self.dim_reduction = nn.Sequential(
                nn.Linear(config.node_feature_size, 8),
                nn.LeakyReLU()
            )
            final_dim = 8 * self.node_size

        elif self.readout == "sum":
            self.norm = nn.BatchNorm1d(config.node_feature_size)

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, config.num_classes)
        )

    def forward(self, node_feature, **kwargs):
        bz, _, _, = node_feature.shape

        for atten in self.attention_list:
            node_feature = atten(node_feature)

        if self.readout == "concat":
            node_feature = self.dim_reduction(node_feature)
            feature = node_feature.reshape((bz, -1))
        elif self.readout == "mean":
            feature = torch.mean(node_feature, dim=1)
        elif self.readout == "max":
            feature, _ = torch.max(node_feature, dim=1)
        elif self.readout == "sum":
            feature = torch.sum(node_feature, dim=1)
            feature = self.norm(feature)
        else:
            raise

        logits = self.fc(feature)
        return BrainEncoderOutputs(logits=logits,
                                   brain_state=node_feature)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

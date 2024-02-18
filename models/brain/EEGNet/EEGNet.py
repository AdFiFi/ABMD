import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import *
from .Layers import BaseSpatialModule, BaseTemporalModule, BaseFrequencyModule


class EEGNetConfig(BrainEncoderConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 dropout=0.5,
                 frequency=128,
                 D=2,
                 num_kernels=8,
                 p1=4,
                 p2=8,
                 use_temporal=True,
                 fusion="flatten"):
        super(EEGNetConfig, self).__init__(node_size=node_size,
                                           node_feature_size=node_feature_size,
                                           time_series_size=time_series_size,
                                           num_classes=num_classes,
                                           dropout=dropout,
                                           use_temporal=use_temporal,
                                           fusion=fusion)
        self.frequency = frequency
        self.num_kernels = num_kernels
        self.D = D
        self.p1 = p1
        self.p2 = p2
        self.output_time_feature_size = num_kernels * D
        self.output_time_series_size = int(time_series_size // p1 // p2)
        if self.use_temporal:
            self.d_model = self.output_time_feature_size
        else:
            if self.fusion == "flatten":
                self.d_model = self.output_time_feature_size * self.output_time_series_size
            else:
                self.d_model = self.output_time_feature_size


class EEGNet(nn.Module):
    def __init__(self, config: EEGNetConfig):
        super(EEGNet, self).__init__()
        self.config = config
        self.input_node_size = config.node_size
        self.output_node_size = config.D
        self.output_time_feature_size = config.num_kernels * self.output_node_size
        self.output_time_series_size = int(config.time_series_size//config.p1//config.p2)

        self.frequency_layer = BaseFrequencyModule(config)

        self.spatial_layer = BaseSpatialModule(config,
                                               input_node_size=self.input_node_size,
                                               output_node_size=self.output_node_size)

        self.temporal_layer = BaseTemporalModule(config, D=self.output_node_size)
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.config.d_model, config.num_classes)
        )

    def forward(self, time_series, **kwargs):
        time_series = time_series.unsqueeze(1)
        hidden_state = self.frequency_layer(time_series)
        hidden_state = self.spatial_layer(hidden_state)
        hidden_state = self.temporal_layer(hidden_state)
        if self.config.fusion == "flatten":
            features = hidden_state.reshape(hidden_state.size(0), -1)
        elif self.config.fusion == "mean":
            # features = hidden_state.mean(dim=-1)
            B, S, T = hidden_state.shape
            brain_signals_mask = kwargs["brain_signals_mask"]
            # mask_length = torch.ceil(brain_signals_mask.sum(dim=-1) / self.config.p1 / self.config.p2)
            # mask_length = mask_length.int()
            # brain_signals_mask = torch.zeros((B, T), device=hidden_state.device)
            # index = torch.arange(T, device=hidden_state.device).unsqueeze(0)  # 创建索引序列
            # mask = torch.lt(index, mask_length.unsqueeze(1))  # 检查索引是否小于n_i
            # values = torch.div(mask, mask_length.unsqueeze(1).float())  # 广播乘以1/n_i
            # brain_signals_mask += values
            brain_signals_mask = torch.div(brain_signals_mask, brain_signals_mask.sum(dim=1).unsqueeze(1))
            features = torch.einsum("bst, bt -> bs", hidden_state, brain_signals_mask)
        else:
            raise
        logits = F.softmax(self.fc(features), dim=1)
        return BrainEncoderOutputs(logits=logits,
                                   brain_state=features,
                                   hidden_state=hidden_state)

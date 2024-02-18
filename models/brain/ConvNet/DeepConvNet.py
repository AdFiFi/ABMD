import torch
import torch.nn as nn
from ..base import *


class DeepConvNetConfig(BrainEncoderConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 num_kernels=25,
                 use_temporal=False,
                 fusion="flatten"):
        super(DeepConvNetConfig, self).__init__(node_size=node_size,
                                                node_feature_size=node_feature_size,
                                                time_series_size=time_series_size,
                                                num_classes=num_classes,
                                                fusion=fusion,
                                                use_temporal=use_temporal)
        self.num_kernels = num_kernels
        output_time_series_size = (((time_series_size - 5 + 1) - 2) // 2 + 1)
        output_time_series_size = (((output_time_series_size - 5 + 1) - 2) // 2 + 1)
        output_time_series_size = (((output_time_series_size - 5 + 1) - 2) // 2 + 1)
        self.output_time_feature_size = num_kernels * 4
        self.output_time_series_size = output_time_series_size
        if self.use_temporal:
            self.d_model = self.output_time_feature_size
        else:
            if self.fusion == "flatten":
                self.d_model = self.output_time_feature_size * self.output_time_series_size
            else:
                self.d_model = self.output_time_feature_size


class DeepConvNet(nn.Module):
    def __init__(self, config: DeepConvNetConfig):
        super(DeepConvNet, self).__init__()
        self.config = config
        self.block1 = nn.Sequential(
            nn.Conv2d(1, config.num_kernels, (1, 5)),
            nn.Conv2d(config.num_kernels, config.num_kernels, (config.node_size, 1), bias=False),
            nn.BatchNorm2d(config.num_kernels),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )
        hidden_size = (((config.time_series_size - 5 + 1) - 2) // 2 + 1)

        self.block2 = nn.Sequential(
            nn.Conv2d(config.num_kernels, config.num_kernels*2, (1, 5)),
            nn.BatchNorm2d(config.num_kernels*2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )
        hidden_size = (((hidden_size - 5 + 1) - 2) // 2 + 1)

        self.block3 = nn.Sequential(
            nn.Conv2d(config.num_kernels*2, config.num_kernels*4, (1, 5)),
            nn.BatchNorm2d(config.num_kernels*4),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )
        hidden_size = (((hidden_size - 5 + 1) - 2) // 2 + 1) * 4
        self.classifier = nn.Linear(hidden_size, config.num_classes)

    def forward(self, time_series, **kwargs):
        time_series = time_series.unsqueeze(1)
        hidden_state = self.block1(time_series)
        hidden_state = self.block2(hidden_state)
        hidden_state = self.block3(hidden_state)
        if self.config.fusion == "flatten":
            features = torch.flatten(hidden_state, 1)
        elif self.config.fusion == "mean":
            # features = hidden_state.mean(dim=-1)
            B, S, T = hidden_state.shape
            brain_signals_mask = kwargs["brain_signals_mask"]
            brain_signals_mask = torch.div(brain_signals_mask, brain_signals_mask.sum(dim=1).unsqueeze(1))
            features = torch.einsum("bst, bt -> bs", hidden_state, brain_signals_mask)
        else:
            raise
        logits = self.classifier(features)
        return BrainEncoderOutputs(logits=logits,
                                   brain_state=hidden_state)

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import wandb

from .MVA import MultiViewAttention
from .DCA import DynamicConnectogramAttention
from .LTSA import LocalTemporalSlidingAttention
from ..base import *


class DFaSTConfig(BrainEncoderConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 frequency=200,
                 window_size=50,
                 window_stride=5,
                 dynamic_stride=1,
                 dynamic_length=600,
                 k=5,
                 sparsity=0.7,
                 num_kernels=3,
                 D=5,
                 p1=4,
                 p2=8,
                 num_heads=4,
                 activation='gelu',
                 dropout=0.1,
                 fusion="flatten",
                 use_temporal=False
                 ):
        super(DFaSTConfig, self).__init__(node_size=node_size,
                                          node_feature_size=node_feature_size,
                                          time_series_size=time_series_size,
                                          activation=activation,
                                          dropout=dropout,
                                          num_classes=num_classes,
                                          use_temporal=use_temporal,
                                          fusion=fusion)
        self.k = k
        self.sparsity = sparsity
        self.frequency = frequency
        self.num_kernels = num_kernels
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_stride = window_stride
        self.dynamic_stride = dynamic_stride
        self.dynamic_length = dynamic_length
        self.aggregate = fusion
        self.d_hidden = None
        self.reg_lambda = 1e-4
        self.orthogonal = True
        self.freeze_center = True
        self.project_assignment = True
        self.p1 = p1
        self.p2 = p2
        self.D = D
        self.fs_fusion_type = 'add'
        # self.fs_fusion_type = 'concat'
        self.qkv_projector = 'linear'
        # self.qkv_projector = 'conv'
        
        self.output_time_feature_size = num_kernels * D
        self.output_time_series_size = int(time_series_size // p1 // p2)
        if self.use_temporal:
            self.d_model = self.output_time_feature_size
        else:
            if self.fusion == "flatten":
                self.d_model = self.output_time_feature_size * self.output_time_series_size
            else:
                self.d_model = self.output_time_feature_size


class DFaST(nn.Module):
    def __init__(self, config: DFaSTConfig):
        super().__init__()
        self.config = config
        self.input_node_size = config.node_size
        self.output_node_size = config.D
        self.output_time_feature_size = config.num_kernels * self.output_node_size
        # self.output_time_series_size = int(config.time_series_size//config.p1//config.p2) + 1
        self.output_time_series_size = int(config.time_series_size // config.p1 // config.p2)
        self.fs_fusion_type = config.fs_fusion_type
        self.dropout = nn.Dropout(config.dropout)
        if self.fs_fusion_type == 'concat':
            self.config.num_kernels = self.config.num_kernels // 2
        self.mva = MultiViewAttention(config)
        self.dca = DynamicConnectogramAttention(config)
        if self.fs_fusion_type == 'concat':
            self.config.num_kernels = self.config.num_kernels * 2
        self.ltsa = LocalTemporalSlidingAttention(self.config, d_model=self.output_time_feature_size)

        self.batch_norm1 = nn.BatchNorm2d(self.config.num_kernels * self.output_node_size)

    def forward(self, hidden_state):
        hidden_state = hidden_state.unsqueeze(1)
        hidden_state1 = self.mva(hidden_state)
        hidden_state2 = self.dca(hidden_state)
        hidden_state = self.dropout(self.frequency_spatial_fusion(hidden_state1, hidden_state2))
        hidden_state = self.batch_norm1(hidden_state)
        hidden_state = self.ltsa(hidden_state)
        return hidden_state

    def frequency_spatial_fusion(self, frequency_feature, spatial_feature):
        if self.fs_fusion_type == 'add':
            return frequency_feature + spatial_feature
        elif self.fs_fusion_type == 'concat':
            frequency_feature = rearrange(frequency_feature, 'b (f n) m t -> b f (n m) t',
                                          f=self.config.num_kernels // 2)
            spatial_feature = rearrange(spatial_feature, 'b (f n) m t -> b f (n m) t', f=self.config.num_kernels // 2)
            fusion = torch.concat([frequency_feature, spatial_feature], dim=1)
            fusion = rearrange(fusion, 'b f (n m) t -> b (f n) m t', m=1)
            return fusion


class DFaSTForClassification(nn.Module):
    def __init__(self, config: DFaSTConfig):
        super().__init__()
        self.config = config
        self.d_fast = DFaST(config)
        self.dropout = nn.Dropout(config.dropout)
        if config.aggregate == 'attention':
            self.attention = nn.Linear(self.fast_p.output_time_series_size, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes)
        )

    def forward(self, time_series, **kwargs):
        hidden_state = self.d_fast(time_series)
        features = self.aggregate(hidden_state, **kwargs)
        logits = F.softmax(self.fc(features), dim=1)
        return BrainEncoderOutputs(logits=logits,
                                   hidden_state=hidden_state,
                                   brain_state=features)

    def aggregate(self, features, **kwargs):
        if self.config.aggregate == 'flatten':
            N, _, _ = features.shape
            features = features.reshape(N, -1)
        elif self.config.aggregate == 'mean':
            if "brain_signals_mask" in kwargs:
                brain_signals_mask = kwargs["brain_signals_mask"]
                # mask_length = torch.ceil(brain_signals_mask.sum(dim=-1) / self.config.p1 / self.config.p2)
                # mask_length = mask_length.int()
                # brain_signals_mask = torch.zeros((B, T), device=features.device)
                # index = torch.arange(T, device=features.device).unsqueeze(0)  # 创建索引序列
                # mask = torch.lt(index, mask_length.unsqueeze(1))  # 检查索引是否小于n_i
                # values = torch.div(mask, mask_length.unsqueeze(1).float())  # 广播乘以1/n_i
                # brain_signals_mask += values
                brain_signals_mask = torch.div(brain_signals_mask, brain_signals_mask.sum(dim=1).unsqueeze(1))
                features = torch.einsum("bst, bt -> bs", features, brain_signals_mask)
            else:
                features = features.mean(dim=-1)
        elif self.config.aggregate == 'attention':
            features = self.attention(features).squeeze(-1)
        return features

    def get_gradients(self):
        mean = 0
        var = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                mean += abs(param.grad.mean().item())
                var += abs(param.grad.var().item())
        wandb.log({f'Gradients/mean': mean,
                   f'Gradients/variance': var})
        return mean, var

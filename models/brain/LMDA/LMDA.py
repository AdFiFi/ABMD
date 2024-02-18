import torch
import torch.nn as nn
from torch.nn import functional as F
from ..base import *


class LMDAConfig(BrainEncoderConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 depth=9,
                 channel_depth1=24,
                 channel_depth2=9,
                 ave_depth=1,
                 avepool=5,
                 fusion="flatten",
                 use_temporal=False):
        super(LMDAConfig, self).__init__(node_size=node_size,
                                         node_feature_size=node_feature_size,
                                         time_series_size=time_series_size,
                                         num_classes=num_classes,
                                         use_temporal=use_temporal,
                                         fusion=fusion)
        self.depth = depth
        self.channel_depth1 = channel_depth1
        self.channel_depth2 = channel_depth2
        self.ave_depth = ave_depth
        self.avepool = avepool

        self.output_time_feature_size = channel_depth2
        self.output_time_series_size = (time_series_size - 75 + 1) // avepool
        if self.use_temporal:
            self.d_model = self.output_time_feature_size
        else:
            if self.fusion == "flatten":
                self.d_model = self.output_time_feature_size * self.output_time_series_size
            else:
                self.d_model = self.output_time_feature_size


class EEGDepthAttention(nn.Module):
    """
    Build EEG Depth Attention module.
    :arg
    C: num of channels
    W: num of time samples
    k: learnable kernel size
    """
    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)  # original kernel k
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        """
        :arg
        """
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)

        return y * self.C * x


class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    """
    def __init__(self, config: LMDAConfig):
        super(LMDA, self).__init__()
        self.config = config
        self.ave_depth = config.ave_depth
        self.channel_weight = nn.Parameter(torch.randn(config.depth, 1, config.node_size), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(config.depth, config.channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(config.channel_depth1),
            nn.Conv2d(config.channel_depth1, config.channel_depth1, kernel_size=(1, 75),
                      groups=config.channel_depth1, bias=False),
            nn.BatchNorm2d(config.channel_depth1),
            nn.GELU(),
        )
        # self.avgPool1 = nn.AvgPool2d((1, 24))
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(config.channel_depth1, config.channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(config.channel_depth2),
            nn.Conv2d(config.channel_depth2, config.channel_depth2, kernel_size=(config.node_size, 1),
                      groups=config.channel_depth2, bias=False),
            nn.BatchNorm2d(config.channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, config.avepool)),
            # nn.AdaptiveAvgPool3d((9, 1, 35)),
            nn.Dropout(p=0.65),
        )
        hidden_size = config.time_series_size - 75 + 1
        self.depthAttention = EEGDepthAttention(hidden_size, config.node_size, k=7)
        hidden_size = config.channel_depth2 * (hidden_size // config.avepool)
        self.classifier = nn.Linear(hidden_size, config.num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, time_series, **kwargs):
        time_series = time_series.unsqueeze(1)
        time_series = torch.einsum('bdcw, hdc->bhcw', time_series, self.channel_weight)  # 导联权重筛选

        x_time = self.time_conv(time_series)  # batch, depth1, channel, samples_
        x_time = self.depthAttention(x_time)  # DA1

        time_series = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        time_series = self.norm(time_series)

        features = torch.flatten(time_series, 1)
        logits = F.softmax(self.classifier(features), dim=1)
        return BrainEncoderOutputs(logits=logits,
                                   brain_state=time_series)

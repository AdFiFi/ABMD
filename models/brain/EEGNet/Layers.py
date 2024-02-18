import torch.nn as nn
from torch.nn import functional as f


class BaseSpatialModule(nn.Module):
    def __init__(self, config, input_node_size=30, output_node_size=5, pooling=True):
        super(BaseSpatialModule, self).__init__()
        self.config = config
        self.conv = nn.Conv2d(config.num_kernels, config.num_kernels * output_node_size, (input_node_size, 1),
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(config.num_kernels * output_node_size, False)
        self.pooling = nn.AvgPool2d((1, config.p1)) if pooling else None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_state):
        spatial_feature = f.elu(self.batch_norm(self.conv(hidden_state)))
        if self.pooling is not None:
            spatial_feature = self.pooling(spatial_feature)
        spatial_feature = self.dropout(spatial_feature)
        return spatial_feature


class BaseFrequencyModule(nn.Module):
    def __init__(self, config):
        super(BaseFrequencyModule, self).__init__()
        self.config = config
        kern_length = config.frequency // 2
        self.padding_flag = False if kern_length % 2 else True

        # Layer 1
        self.padding = nn.ZeroPad2d((0, 1, 0, 0))
        self.conv = nn.Conv2d(1, config.num_kernels, (1, kern_length), padding=(0, (kern_length-1)//2), bias=False)
        self.batch_norm = nn.BatchNorm2d(config.num_kernels, False)

    def forward(self, hidden_state):
        hidden_state = self.padding(hidden_state) if self.padding_flag else hidden_state
        hidden_state = self.batch_norm(self.conv(hidden_state))
        return hidden_state


class BaseTemporalModule(nn.Module):
    def __init__(self, config, D=5, pooling=True):
        super(BaseTemporalModule, self).__init__()
        self.config = config
        kern_length = config.frequency // 2
        self.padding = nn.ZeroPad2d((0, 1, 0, 0))
        self.conv = nn.Conv2d(config.num_kernels * D, config.num_kernels * D, (1, int(kern_length*0.5)),
                              padding=(0, int(kern_length*0.5)//2-1), bias=False)
        self.batch_norm = nn.BatchNorm2d(config.num_kernels * D, False)
        self.pooling = nn.AvgPool2d((1, config.p2)) if pooling else None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_state):
        hidden_state = self.padding(hidden_state)
        hidden_state = f.elu(self.batch_norm(self.conv(hidden_state)))
        if self.pooling is not None:
            hidden_state = self.pooling(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = hidden_state.squeeze(2)
        return hidden_state

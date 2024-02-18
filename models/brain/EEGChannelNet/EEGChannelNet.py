from ..base import *
from .EEGChannlNetLayers import *


class EEGChannelNetConfig(BrainEncoderConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 use_temporal=False,
                 fusion="flatten"):
        super(EEGChannelNetConfig, self).__init__(node_size=node_size,
                                                  node_feature_size=node_feature_size,
                                                  time_series_size=time_series_size,
                                                  num_classes=num_classes,
                                                  use_temporal=use_temporal,
                                                  fusion=fusion)
        self.num_kernels = 50
        self.output_time_feature_size = (((((node_size // 2) // 2) // 2) // 2 + 1) // 2 + 1 - 2) * self.num_kernels
        self.output_time_series_size = (((((time_series_size // 2) // 2) // 2 + 1) // 2 + 1) // 2 + 1 - 2)
        if self.use_temporal:
            self.d_model = self.output_time_feature_size
        else:
            if self.fusion == "flatten":
                self.d_model = self.output_time_feature_size * self.output_time_series_size
            else:
                self.d_model = self.output_time_feature_size


class EEGChannelNet(nn.Module):
    """
    A Reproduction of EEGChannelNet:
    Decoding brain representations by multimodal learning of neural activity and visual features

    S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt and M. Shah

    IEEE Transactions on Pattern Analysis and Machine Intelligence 2020 Vol. 43 Issue 11 Pages 3833-3849

    """

    def __init__(self, config: EEGChannelNetConfig):
        super(EEGChannelNet, self).__init__()
        self.config = config
        self.temporal_block = TemporalBlock()
        self.spatial_block = SpatialBlock()
        self.residual_block = ResidualBlock()
        inp_channels = 200
        num_kernels = 50
        hidden_size = (((((config.time_series_size // 2) // 2) // 2 + 1) // 2 + 1) // 2 + 1 - 2) * \
                      (((((config.node_size // 2) // 2) // 2) // 2 + 1) // 2 + 1 - 2) * num_kernels
        self.output_layer = nn.Conv2d(inp_channels, num_kernels, (3, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 0))
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(hidden_size, hidden_size * 2),
                                        nn.Linear(hidden_size * 2, config.num_classes))

    def forward(self, time_series, **kwargs):
        time_series = time_series.unsqueeze(1)
        hidden_state = self.temporal_block(time_series)
        hidden_state = self.spatial_block(hidden_state)
        hidden_state = self.residual_block(hidden_state)
        features = self.output_layer(hidden_state)
        logits = self.classifier(features)
        return BrainEncoderOutputs(logits=logits,
                                   brain_state=features)

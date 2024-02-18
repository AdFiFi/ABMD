import json


class BrainEncoderConfig(object):
    def __init__(self,
                 num_classes=2,
                 node_size=200,
                 node_feature_size=200,
                 time_series_size=100,
                 activation='gelu',
                 dropout=0.1,
                 d_model=1024,
                 d_hidden=2048,
                 use_temporal=True,
                 fusion="flatten"
                 ):
        self.num_classes = num_classes
        self.node_size = node_size
        self.time_series_size = time_series_size
        self.node_feature_size = node_feature_size
        self.activation = activation
        self.dropout = dropout
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.use_temporal = use_temporal
        self.fusion = fusion

    def load(self, path):
        config = json.load(open(path))['brain_config']
        for k, v in config.items():
            self.__dict__[k] = v


class BrainEncoderOutputs:
    def __init__(self,
                 logits,
                 brain_state,
                 hidden_state=None):
        self.logits = logits
        self.brain_state = brain_state
        self.hidden_state = hidden_state

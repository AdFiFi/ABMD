import json


class TextEncoderConfig(object):
    def __init__(self,
                 mode: object = '',
                 class_weight: object = None,
                 num_classes: object = 2,
                 d_model: object = 1024,
                 d_hidden: object = 2048,
                 dropout: object = 0.5,
                 fusion: object = "flatten"
                 ):
        self.mode = mode
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.d_model = d_model

        self.d_hidden = d_hidden
        self.dropout = dropout
        self.fusion = fusion

    def load(self, path):
        config = json.load(open(path))['text_config']
        for k, v in config.items():
            self.__dict__[k] = v


class TextEncoderOutputs:
    def __init__(self,
                 logits: object,
                 text_state: object,
                 hidden_state: object = None) -> object:
        self.logits = logits
        self.text_state = text_state
        self.hidden_state = hidden_state

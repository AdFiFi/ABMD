import json


class VisionEncoderConfig(object):
    def __init__(self,
                 mode='supervised',
                 class_weight=None,
                 num_classes=2,
                 d_hidden=1024):
        self.mode = mode
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.d_hidden = d_hidden

    def load(self, path):
        config = json.load(open(path))['vision_config']
        for k, v in config.items():
            self.__dict__[k] = v


class VisionEncoderOutputs:
    def __init__(self,
                 logits=None,
                 vision_state=None,
                 projected_text_state=None):
        self.logits = logits
        self.vision_state = vision_state
        self.projected_text_state = projected_text_state

import torch
import math
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np

from .base import ModelConfig, ModelOutputs, BaseModel

"""
HMAV
"""


class HMAVConfig(ModelConfig):
    def __init__(self,
                 vision_config=None,
                 text_config=None,
                 brain_config=None,
                 d_model=512,
                 d_hidden=2048,
                 num_classes=2,
                 modality='vision-brain',
                 class_weight=None,
                 label_smoothing=0,
                 use_temporal=False,
                 use_sequence=False):
        super().__init__(text_config=text_config,
                         vision_config=vision_config,
                         brain_config=brain_config,
                         d_model=d_model,
                         d_hidden=d_hidden,
                         modality=modality,
                         class_weight=class_weight,
                         label_smoothing=label_smoothing,
                         num_classes=num_classes,
                         use_temporal=use_temporal,
                         use_sequence=use_sequence)
        self.d_drop = 0.1
        self.alpha = 0.1


class HMAVModelOutputs(ModelOutputs):
    def __init__(self,
                 logits=None,
                 text_logits=None,
                 brain_logits=None,
                 vision_logits=None,
                 audio_logits=None,
                 feature=None,
                 brain_state=None,
                 text_state=None,
                 vision_state=None,
                 loss=None,
                 loss_regression=None
                 ):
        super().__init__(logits=logits,
                         loss=loss,
                         feature=feature,
                         brain_state=brain_state,
                         text_state=text_state,
                         vision_state=vision_state,
                         )
        self.text_logits = text_logits
        self.brain_logits = brain_logits
        self.vision_logits = vision_logits
        self.audio_logits = audio_logits
        self.loss_regression = loss_regression


class RidgeRegression(nn.Module):
    def __init__(self, config:HMAVConfig):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(config.d_model, config.d_model)
        self.alpha = config.alpha
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        x = self.linear(x)
        loss = self.mse(x, y)
        return loss

    def l2_penalty(self):
        w = self.linear.weight
        return torch.sum(w.view(-1) ** 2) * 0.5 * self.alpha


class HMAV(BaseModel):
    def __init__(self, config: HMAVConfig,
                 vision_encoder=None, text_encoder=None, audio_encoder=None, brain_encoder=None):
        super().__init__(config=config,
                         vision_encoder=vision_encoder,
                         text_encoder=text_encoder,
                         audio_encoder=audio_encoder,
                         brain_encoder=brain_encoder,
                         classifier="common"
                         )
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                                 weight=torch.tensor(config.class_weight))

        self.regressor = RidgeRegression(config)

    def text_brain_task(self, text, brain_signals, labels, **kwargs):
        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
        text_state = self.encode_text(text, **kwargs).text_state

        brain_logits = self.common_classifier(brain_state)
        text_logits = self.common_classifier(text_state)

        loss_regression = self.regressor(text_state, brain_state)

        if "step_one" in kwargs and kwargs["step_one"]:
            loss = self.cross_entropy(brain_logits, labels)
        else:
            loss = self.cross_entropy(brain_logits, labels) + self.cross_entropy(text_logits, labels) + loss_regression

        return HMAVModelOutputs(loss=loss,
                                loss_regression=loss_regression,
                                text_state=text_state,
                                brain_state=brain_state,
                                text_logits=text_logits,
                                brain_logits=brain_logits
                               )

    def text_task(self, text, **kwargs):
        text_state = self.encode_text(text, **kwargs).text_state
        feature_text = self.shared_feature_extractor(text_state)
        mu_specific_text, log_var_specific_text, specific_text_state = self.text_exclusive_specific_encoder(text_state)
        text_state = torch.concat([specific_text_state, feature_text], dim=-1)
        text_logits = self.text_classifier(text_state)
        loss = self.cross_entropy(text_logits, target=kwargs["labels"])
        return HMAVModelOutputs(loss=loss,
                                text_logits=text_logits,
                                text_state=text_state
                                )

    def brain_task(self, brain_signals, **kwargs):
        brain_state = self.encode_text(brain_signals, **kwargs).text_state
        feature_brain = self.shared_feature_extractor(brain_state)
        mu_specific_brain, log_var_specific_text, specific_text_state = self.text_exclusive_specific_encoder(brain_state)
        brain_state = torch.concat([specific_text_state, feature_brain], dim=-1)
        brain_logits = self.text_classifier(brain_state)
        loss = self.cross_entropy(brain_logits, target=kwargs["labels"])
        return HMAVModelOutputs(loss=loss,
                                brain_logits=brain_logits,
                                brain_state=brain_state
                                )

import torch
from copy import deepcopy
from torch import nn
from torch.nn import functional as F


class HBDConfig(object):
    def __init__(self, vision_config, brain_config, mode='distill', class_weight=None, label_smoothing=0, lam=0.5, t=3):
        self.mode = mode
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing
        self.t = t
        self.lam = lam
        self.vision_config = vision_config
        self.brain_config = brain_config
        self.sample_size = 5


class HBDModelOutputs:
    def __init__(self, logits=None, loss=None, loss_kl=None):
        self.logits = logits
        self.loss = loss
        self.loss_kl = loss_kl


class DistillLoss(nn.Module):
    def __init__(self, t=3):
        super().__init__()
        self.t = t

    def forward(self, vision_state, brain_state):
        vision_state = F.log_softmax(vision_state / self.t, dim=-1)
        brain_state = F.softmax(brain_state / self.t, dim=-1)
        loss = -torch.sum(vision_state * brain_state, dim=-1).mean()
        return loss



class HBD(nn.Module):
    def __init__(self, config: HBDConfig, vision_encoder, brain_encoder):
        super().__init__()
        self.config = config
        self.vision_encoder = vision_encoder
        self.brain_encoder = brain_encoder
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                                 weight=torch.tensor(config.class_weight))
        self.kl = DistillLoss(t=config.t)
        if self.config.mode == 'distill':
            self.freeze(self.brain_encoder)
            self.fit_layer = nn.Linear(config.brain_config.d_hidden, config.vision_config.d_hidden, bias=False)
            self.dropout = nn.Dropout(0.5)
            # nn.init.orthogonal_(self.fit_layer.weight)
            # nn.init.zeros_(self.fit_layer.weight)
            # self.freeze(self.fit_layer)

    def forward(self, labels, images=None, brain_signals=None):
        if self.config.mode == 'vision':
            output = self.supervise_vision(images, labels)
        elif self.config.mode == 'brain':
            output = self.supervise_brain(brain_signals, labels)
        elif self.config.mode == 'distill':
            output = self.distill(images, brain_signals, labels)
        else:
            output = None
        return output

    def supervise_vision(self, images, labels):
        logits, _ = self.vision_encoder(images)
        loss = self.cross_entropy(logits, labels)
        return HBDModelOutputs(logits=logits,
                               loss=loss)

    def supervise_brain(self, brain_signals, labels):
        logits, _ = self.brain_encoder(brain_signals)
        loss = self.cross_entropy(logits, labels)
        return HBDModelOutputs(logits=logits,
                               loss=loss)

    def distill(self, labels, images, brain_signals=None):
        vision_logits, vision_state = self.vision_encoder(images)
        loss = self.cross_entropy(vision_logits, labels)
        loss_kl = None
        if brain_signals is not None:
            brain_logits, brain_state = self.brain_encoder(brain_signals)
            brain_state = self.dropout(self.fit_layer(brain_state))
            loss_kl = self.kl(vision_state, brain_state)
            # loss2 = self.kl(vision_logits, brain_logits)
            # loss3 = self.cross_entropy(brain_logits, labels)
            loss = loss + loss_kl * self.config.lam
        return HBDModelOutputs(loss=loss,
                               logits=vision_logits,
                               loss_kl=loss_kl)

    @staticmethod
    def freeze(layer):
        for param in layer.parameters():
            param.require_grad = False

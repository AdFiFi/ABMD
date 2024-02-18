import torch
import math
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np

from .base import ModelConfig, ModelOutputs, BaseModel

"""
MV2D
"""


class MV2DConfig(ModelConfig):
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
        self.sample_size = 5


class MV2DModelOutputs(ModelOutputs):
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
                 loss_vcd=None,
                 loss_vmd=None
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
        self.loss_vcd = loss_vcd
        self.loss_vmd = loss_vmd


class SharedEncoder(nn.Module):
    def __init__(self, config: MV2DConfig):
        super().__init__()
        self.mu_dims = config.d_model
        self.layer = nn.Sequential(nn.Dropout(p=config.d_drop),
                                   nn.Linear(config.d_model, config.d_hidden),
                                   # nn.Tanh(),
                                   nn.Dropout(p=config.d_drop),
                                   nn.Linear(config.d_hidden, config.d_model))
        self.fc_mu_var = nn.Sequential(nn.Linear(config.d_model, config.d_model * 2))

    def forward(self, hidden_state):
        hidden_state = self.layer(hidden_state)
        hidden_state = self.fc_mu_var(hidden_state)
        mu, log_var = torch.split(hidden_state, [self.mu_dims, self.mu_dims], dim=-1)
        hidden_state = self.reparameterize(mu, log_var)
        return mu, log_var, hidden_state

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + (eps * std)


class IB(nn.Module):
    def __init__(self, config: MV2DConfig):
        super().__init__()
        self.initial_value = 5.0
        self.alpha = nn.Parameter(torch.full((config.d_model,), fill_value=self.initial_value))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _sample_t(mu, noise_var):
        # log_noise_var = torch.clamp(log_noise_var, -10, 10)
        noise_std = noise_var.sqrt()
        eps = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps

    def forward(self, x, **kwargs):
        lamb = self.sigmoid(self.alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1])
        masked_mu = x * lamb
        masked_var = (1 - lamb) ** 2
        t = self._sample_t(masked_mu, masked_var)
        return masked_mu, torch.log(masked_var), t


class MV2D(BaseModel):
    def __init__(self, config: MV2DConfig,
                 vision_encoder=None, text_encoder=None, audio_encoder=None, brain_encoder=None):
        super().__init__(config=config,
                         vision_encoder=vision_encoder,
                         text_encoder=text_encoder,
                         audio_encoder=audio_encoder,
                         brain_encoder=brain_encoder,
                         classifier="private"
                         )
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                                 weight=torch.tensor(config.class_weight),
                                                 reduction="sum")

        self.shared_encoder = SharedEncoder(config)

        if text_encoder is not None and "text" in self.config.modality:
            self.text_ib = IB(config)
            self.text_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if vision_encoder is not None and "vision" in self.config.modality:
            self.vision_ib = IB(config)
            self.vision_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if audio_encoder is not None and "audio" in self.config.modality:
            self.audio_ib = IB(config)
            self.audio_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if brain_encoder is not None and "brain" in self.config.modality:
            self.brain_ib = IB(config)
            self.brain_classifier = nn.Linear(config.d_model, self.config.num_classes)

    def text_brain_task(self, text, brain_signals, labels, **kwargs):
        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
        text_state = self.encode_text(text, **kwargs).text_state

        y_brain_state_mu, y_brain_state_log_var, y_brain_state = self.shared_encoder(brain_state)
        y_text_state_mu, y_text_state_log_var, y_text_state = self.shared_encoder(text_state)

        z_brain_state_mu, z_brain_state_log_var, z_brain_state = self.brain_ib(brain_state)
        z_text_state_mu, z_text_state_log_var, z_text_state = self.text_ib(text_state)

        brain_logits = self.brain_classifier(z_brain_state)
        text_logits = self.text_classifier(z_text_state)

        loss_vcd = self.kl_div(y_brain_state_mu, y_brain_state_log_var, z_text_state_mu, z_text_state_log_var) + \
                   self.kl_div(y_text_state_mu, y_text_state_log_var, z_brain_state_mu, z_brain_state_log_var)
        loss_vmd = self.js_div(z_brain_state_mu, z_brain_state_log_var, z_text_state_mu, z_text_state_log_var)
        loss_ce = self.cross_entropy(brain_logits, labels) + self.cross_entropy(text_logits, labels)

        if "step_one" in kwargs and kwargs["step_one"]:
            loss = loss_ce
        else:
            loss = 100 * loss_ce + 0.005 * (loss_vmd + loss_vcd)
        return MV2DModelOutputs(loss=loss,
                                loss_vmd=loss_vmd,
                                loss_vcd=loss_vcd,
                                brain_state=z_brain_state,
                                text_state=z_text_state,
                                text_logits=text_logits,
                                brain_logits=brain_logits
                                )

    def text_task(self, text, **kwargs):
        text_state = self.encode_text(text, **kwargs).text_state
        y_text_state_mu, y_text_state_log_var, y_text_state = self.shared_encoder(text_state)
        z_text_state_mu, z_text_state_log_var, z_text_state = self.text_ib(text_state)
        text_logits = self.text_classifier(z_text_state)
        loss = self.cross_entropy(text_logits, target=kwargs["labels"]) + \
               self.kl_div(y_text_state_mu, y_text_state_log_var, z_text_state_mu, z_text_state_log_var)
        return MV2DModelOutputs(loss=loss,
                                text_logits=text_logits,
                                text_state=text_state
                                )

    def brain_task(self, brain_signals, **kwargs):
        brain_state = self.encode_text(brain_signals, **kwargs).text_state
        y_brain_state_mu, y_brain_state_log_var, y_brain_state = self.shared_encoder(brain_state)
        z_brain_state_mu, z_brain_state_log_var, z_brain_state = self.brain_ib(brain_state)
        brain_logits = self.brain_classifier(z_brain_state)
        loss = self.cross_entropy(brain_logits, target=kwargs["labels"]) + \
               self.kl_div(y_brain_state_mu, y_brain_state_log_var, z_brain_state_mu, z_brain_state_log_var)
        return MV2DModelOutputs(loss=loss,
                                brain_logits=brain_logits,
                                brain_state=brain_state
                                )

    @staticmethod
    # def kl_div(mu_q, std_q, mu_p, std_p):
    #     """Computes the KL divergence between the two given variational distribution.\
    #        This computes KL(q||p), which is not symmetric. It quantifies how far is\
    #        The estimated distribution q from the true distribution of p."""
    #     k = mu_q.size(1)
    #     mu_diff = mu_p - mu_q
    #     mu_diff_sq = torch.mul(mu_diff, mu_diff)
    #     logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
    #     logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
    #     fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
    #     kl_divergence = (fs - k + logdet_std_p - logdet_std_q) * 0.5
    #     return kl_divergence.mean()

    @staticmethod
    def kl_div(mu0, log_var0, mu1, log_var1):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + (log_var0 - log_var1) - ((mu0 - mu1) ** 2 + log_var0.exp() / log_var1.exp()), dim=1),
            dim=0)
        return kld_loss

    def js_div(self, mu_q, std_q, mu_p, std_p):
        return 0.5 * (self.kl_div(mu_q, std_q, mu_p, std_p) + self.kl_div(mu_p, std_p, mu_q, std_q))

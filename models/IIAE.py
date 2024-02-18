import torch
import math
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np

from .base import ModelConfig, ModelOutputs, BaseModel

"""
IIAE
"""


class IIAEConfig(ModelConfig):
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


class IIAEModelOutputs(ModelOutputs):
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
                 reconstruction_loss=None,
                 kl_loss=None,
                 inter_kl_loss=None
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
        self.reconstruction_loss = reconstruction_loss
        self.kl_loss = kl_loss
        self.inter_kl_loss = inter_kl_loss


class SpecificEncoder(nn.Module):
    def __init__(self, config: IIAEConfig):
        super().__init__()
        self.mu_dims = config.d_model
        self.layer = nn.Sequential(nn.Dropout(p=config.d_drop),
                                   nn.Linear(config.d_model, config.d_hidden),
                                   nn.LeakyReLU(0.02),
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


class SharedEncoder(nn.Module):
    def __init__(self, config: IIAEConfig):
        super().__init__()
        self.mu_dims = config.d_model
        self.layer = nn.Sequential(nn.Dropout(p=config.d_drop),
                                   nn.Linear(2 * config.d_model, config.d_hidden),
                                   nn.LeakyReLU(0.02),
                                   nn.Dropout(p=config.d_drop),
                                   nn.Linear(config.d_hidden, config.d_model))
        self.fc_mu_var = nn.Sequential(nn.Linear(config.d_model, config.d_model * 2))

    def forward(self, hidden_state1, hidden_state2):
        hidden_state = torch.concat([hidden_state1, hidden_state2], dim=-1)
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


class FeatureExtractor(nn.Module):
    def __init__(self, config: IIAEConfig):
        super().__init__()
        self.layer = nn.Sequential(nn.Dropout(p=config.d_drop),
                                   nn.Linear(config.d_model, config.d_hidden),
                                   nn.LeakyReLU(0.02),
                                   nn.Dropout(p=config.d_drop),
                                   nn.Linear(config.d_hidden, config.d_model))

    def forward(self, hidden_state):
        return self.layer(hidden_state)


class Decoder(nn.Module):
    def __init__(self, config: IIAEConfig):
        super().__init__()
        self.layer = nn.Sequential(nn.Dropout(p=config.d_drop),
                                   nn.Linear(2 * config.d_model, config.d_hidden),
                                   nn.LeakyReLU(0.02),
                                   nn.Dropout(p=config.d_drop),
                                   nn.Linear(config.d_hidden, config.d_model))

    def forward(self, specific_hidden_state, shared_hidden_state):
        hidden_state = torch.concat([specific_hidden_state, shared_hidden_state], dim=-1)
        hidden_state = self.layer(hidden_state)
        return hidden_state


class ModifiedIIAE(BaseModel):
    def __init__(self, config: IIAEConfig,
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
                                                 reduction='sum')
        self.cross_entropy_brain = nn.CrossEntropyLoss(label_smoothing=0,
                                                       weight=torch.tensor(config.class_weight),
                                                       reduction='sum')
        self.normal_mu = torch.tensor(np.float32(0))
        self.normal_log_var = torch.tensor(np.float32(0))

        if text_encoder is not None and "text" in self.config.modality:
            self.text_exclusive_specific_encoder = SpecificEncoder(config)
            self.text_decoder = Decoder(config)
            self.text_classifier = nn.Linear(config.d_model * 2, self.config.num_classes)

        if vision_encoder is not None and "vision" in self.config.modality:
            self.vision_exclusive_specific_encoder = SpecificEncoder(config)
            self.vision_decoder = Decoder(config)
            self.vision_classifier = nn.Linear(config.d_model * 2, self.config.num_classes)

        if audio_encoder is not None and "audio" in self.config.modality:
            self.audio_exclusive_specific_encoder = SpecificEncoder(config)
            self.audio_decoder = Decoder(config)
            self.audio_classifier = nn.Linear(config.d_model * 2, self.config.num_classes)

        if brain_encoder is not None and "brain" in self.config.modality:
            self.brain_exclusive_specific_encoder = SpecificEncoder(config)
            self.brain_decoder = Decoder(config)
            self.brain_classifier = nn.Linear(config.d_model * 2, self.config.num_classes)

        self.shared_feature_extractor = FeatureExtractor(config)
        self.shared_encoder = SharedEncoder(config)

    def text_brain_task(self, text, brain_signals, labels, **kwargs):
        if "step_one" in kwargs and kwargs["step_one"]:
            brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
            feature_brain = self.shared_feature_extractor(brain_state)
            mu_specific_brain, log_var_specific_brain, specific_brain_state = self.brain_exclusive_specific_encoder(
                brain_state)
            brain_state = torch.concat([specific_brain_state, feature_brain], dim=-1)
            brain_logits = self.brain_classifier(brain_state)
            loss = self.cross_entropy(brain_logits, labels)
            return IIAEModelOutputs(loss=loss,
                                    brain_logits=brain_logits,
                                    brain_state=brain_state)
        else:
            brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
            text_state = self.encode_text(text, **kwargs).text_state

            feature_brain = self.shared_feature_extractor(brain_state)
            mu_specific_brain, log_var_specific_brain, specific_brain_state = self.brain_exclusive_specific_encoder(
                brain_state)

            feature_text = self.shared_feature_extractor(text_state)
            mu_specific_text, log_var_specific_text, specific_text_state = self.text_exclusive_specific_encoder(
                text_state)

            mu_shared, log_var_shared, shared_state = self.shared_encoder(feature_brain, feature_text)

            recons_brain_state = self.brain_decoder(specific_brain_state, shared_state)
            recons_text_state = self.brain_decoder(specific_text_state, shared_state)

            recon_brain_loss = self.reconstruction_loss(brain_state, recons_brain_state, "L1")
            recon_text_loss = self.reconstruction_loss(text_state, recons_text_state, "L1")

            kl_brain_loss = self.KLD_loss(mu_specific_brain, log_var_specific_brain, self.normal_mu,
                                          self.normal_log_var)
            kl_text_loss = self.KLD_loss(mu_specific_text, log_var_specific_text, self.normal_mu,
                                         self.normal_log_var)
            kl_shared_loss = self.KLD_loss(mu_shared, log_var_shared, self.normal_mu, self.normal_log_var)
            kl_inter_brain_loss = self.KLD_loss(mu_shared, log_var_shared, mu_specific_brain,
                                                log_var_specific_brain)
            kl_inter_text_loss = self.KLD_loss(mu_shared, log_var_shared, mu_specific_text, log_var_specific_text)

            brain_state = torch.concat([specific_brain_state, feature_brain], dim=-1)
            text_state = torch.concat([specific_text_state, feature_text], dim=-1)

            logits_brain = self.brain_classifier(brain_state)
            logits_text = self.text_classifier(text_state)

            ce_loss = self.cross_entropy(logits_brain, labels) + self.cross_entropy(logits_text, labels)

            loss = 0.05 * (recon_brain_loss + recon_text_loss) \
                   + 0.05 * (kl_brain_loss + kl_text_loss) \
                   + 0.05 * kl_shared_loss \
                   + 0.05 * (kl_inter_brain_loss + kl_inter_text_loss) \
                   + ce_loss

            return IIAEModelOutputs(loss=loss,
                                    text_state=text_state,
                                    brain_state=brain_state,
                                    text_logits=logits_text,
                                    brain_logits=logits_brain,
                                    reconstruction_loss=recon_brain_loss + recon_text_loss,
                                    kl_loss=recon_text_loss + kl_brain_loss + kl_text_loss + kl_shared_loss + kl_inter_brain_loss + kl_inter_text_loss
                                    )

    def text_task(self, text, **kwargs):
        text_state = self.encode_text(text, **kwargs).text_state
        feature_text = self.shared_feature_extractor(text_state)
        mu_specific_text, log_var_specific_text, specific_text_state = self.text_exclusive_specific_encoder(text_state)
        text_state = torch.concat([specific_text_state, feature_text], dim=-1)
        text_logits = self.text_classifier(text_state)
        loss = self.cross_entropy(text_logits, target=kwargs["labels"])
        return IIAEModelOutputs(loss=loss,
                                text_logits=text_logits,
                                text_state=text_state
                                )

    def brain_task(self, brain_signals, **kwargs):
        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
        feature_brain = self.shared_feature_extractor(brain_state)
        mu_specific_brain, log_var_specific_brain, specific_brain_state = self.brain_exclusive_specific_encoder(
            brain_state)
        brain_state = torch.concat([specific_brain_state, feature_brain], dim=-1)
        brain_logits = self.text_classifier(brain_state)
        loss = self.cross_entropy(brain_logits, target=kwargs["labels"])
        return IIAEModelOutputs(loss=loss,
                                brain_logits=brain_logits,
                                brain_state=brain_state
                                )

    @staticmethod
    def reconstruction_loss(recon, input, name):
        if name == "L1":
            rec_loss = nn.L1Loss()
        elif name == "MSE":
            rec_loss = nn.MSELoss()
        else:
            rec_loss = nn.L1Loss()

        return rec_loss(recon, input)

    @staticmethod
    def KLD_loss(mu0, log_var0, mu1, log_var1):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + (log_var0 - log_var1) - ((mu0 - mu1) ** 2 + log_var0.exp() / log_var1.exp()), dim=1),
            dim=0)
        return kld_loss

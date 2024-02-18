import torch
from torch import nn
import numpy as np

from .base import ModelConfig, ModelOutputs, BaseModel

"""
CLIP (Contrastive Language-Image Pre-training) is a reproduction of paper:
Learning transferable visual models from natural language supervision
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, et al.
International conference on machine learning 2021
Publisher: PMLR Pages: 8748-8763

Code is mostly based on the code at https://github.com/OpenAI/CLIP.
"""


class CLIPConfig(ModelConfig):
    def __init__(self,
                 vision_config=None,
                 text_config=None,
                 brain_config=None,
                 d_model=2048,
                 d_hidden=512,
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


class CLIPModelOutputs(ModelOutputs):
    def __init__(self,
                 logits=None,
                 logits_tva=None,
                 logits_brain=None,
                 feature=None,
                 brain_state=None,
                 text_state=None,
                 vision_state=None,
                 loss=None):
        super().__init__(logits=logits,
                         loss=loss,
                         feature=feature,
                         brain_state=brain_state,
                         text_state=text_state,
                         vision_state=vision_state,
                         )
        self.logits_tva = logits_tva
        self.logits_brain = logits_brain


class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, brain_state, tva_state):
        # normalized features
        tva_state = tva_state / tva_state.norm(dim=1, keepdim=True)
        brain_state = brain_state / brain_state.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_tva = logit_scale * tva_state @ brain_state.t()
        logits_brain = logits_tva.t()

        labels = torch.eye(logits_brain.shape[0], device=brain_state.device)
        loss = (self.cross_entropy(logits_tva, labels) + self.cross_entropy(logits_brain, labels)) / 2

        return loss, logits_tva, logits_brain


class PretrainedCLIP(BaseModel):
    def __init__(self, config: CLIPConfig,
                 vision_encoder=None, text_encoder=None, audio_encoder=None, brain_encoder=None):
        super().__init__(config=config,
                         vision_encoder=vision_encoder,
                         text_encoder=text_encoder,
                         audio_encoder=audio_encoder,
                         brain_encoder=brain_encoder,
                         classifier="private"
                         )
        self.config = config
        self.symmetric_loss = SymmetricCrossEntropyLoss()

    def vision_brain_task(self, brain_signals, images, **kwargs):
        vision_state = self.encode_vision(images, **kwargs).vision_state
        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state

        loss, logits_vision, logits_brain = self.symmetric_loss(vision_state, brain_state)
        return CLIPModelOutputs(loss=loss,
                                vision_state=vision_state,
                                brain_state=brain_state,
                                logits_tva=logits_vision,
                                logits_brain=logits_brain
                                )

    def text_brain_task(self, text, brain_signals, **kwargs):
        text_state = self.encode_text(text, **kwargs).text_state
        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state

        loss, logits_text, logits_brain = self.symmetric_loss(text_state, brain_state)
        return CLIPModelOutputs(loss=loss,
                                text_state=text_state,
                                brain_state=brain_state,
                                logits_tva=logits_text,
                                logits_brain=logits_brain
                                )


class ModifiedCLIPFewShot(BaseModel):
    def __init__(self, config: CLIPConfig,
                 vision_encoder=None, text_encoder=None, audio_encoder=None, brain_encoder=None,
                 text_projector=None, vision_projector=None, audio_projector=None, brain_projector=None):
        super().__init__(config=config,
                         vision_encoder=vision_encoder,
                         text_encoder=text_encoder,
                         audio_encoder=audio_encoder,
                         brain_encoder=brain_encoder,
                         text_projector=text_projector,
                         vision_projector=vision_projector,
                         audio_projector=audio_projector,
                         brain_projector=brain_projector,
                         classifier="private"
                         )
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                                 weight=torch.tensor(config.class_weight))

        if text_encoder is not None and "text" in self.config.modality:
            self.text_encoder.require_grad = False
            self.text_projector.require_grad = False

        if vision_encoder is not None and "vision" in self.config.modality:
            self.vision_encoder.require_grad = False
            self.vision_projector.require_grad = False

        if audio_encoder is not None and "audio" in self.config.modality:
            self.audio_encoder.require_grad = False
            self.audio_projector.require_grad = False

        if brain_encoder is not None and "brain" in self.config.modality:
            self.brain_encoder.require_grad = False
            self.brain_projector.require_grad = False

    def vision_task(self, labels, **kwargs):
        vision_state = self.encode_vision(**kwargs).vision_state
        logits = self.vision_classifier(vision_state)
        loss = self.cross_entropy(logits, labels)
        return CLIPModelOutputs(loss=loss,
                                logits=logits,
                                vision_state=vision_state
                                )

    def text_task(self, labels, **kwargs):
        text_state = self.encode_text(**kwargs).text_state
        logits = self.text_classifier(text_state)
        loss = self.cross_entropy(logits, labels)
        return CLIPModelOutputs(loss=loss,
                                logits=logits,
                                text_state=text_state
                                )

    def brain_task(self, labels, **kwargs):
        brain_state = self.encode_brain(**kwargs).brain_state
        logits = self.brain_classifier(brain_state)
        loss = self.cross_entropy(logits, labels)
        return CLIPModelOutputs(loss=loss,
                                logits=logits,
                                brain_state=brain_state
                                )

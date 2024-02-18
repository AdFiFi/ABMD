import torch
from torch import nn

from .base import ModelConfig, ModelOutputs, BaseModel

"""
NAVF (Multimodal Learning of Neural Activity and Visual Features) is a reproduction of paper:
Decoding brain representations by multimodal learning of neural activity and visual features
S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt and M. Shah
IEEE Transactions on Pattern Analysis and Machine Intelligence 2020 Vol. 43 Issue 11 Pages 3833-3849

"""


class NAVFConfig(ModelConfig):
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


class NAVFModelOutputs(ModelOutputs):
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
                 loss=None):
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


class NAVFTripletLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, anchor_brain_state, positive_tva_state, negative_tva_state):
        matched_compatibility = self.compatibility_function(anchor_brain_state, positive_tva_state)
        mismatched_compatibility = self.compatibility_function(anchor_brain_state, negative_tva_state)
        loss = self.relu(mismatched_compatibility-matched_compatibility).mean()
        return loss

    @staticmethod
    def compatibility_function(brain_state, tva_state):
        return torch.einsum("n e, n e -> n", brain_state, tva_state)


class PretrainedNAVF(BaseModel):
    def __init__(self, config: NAVFConfig,
                 vision_encoder=None, text_encoder=None, audio_encoder=None, brain_encoder=None):
        super().__init__(config=config,
                         vision_encoder=vision_encoder,
                         text_encoder=text_encoder,
                         audio_encoder=audio_encoder,
                         brain_encoder=brain_encoder,
                         classifier="private"
                         )
        self.config = config
        self.triplet_loss = NAVFTripletLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def vision_brain_task(self, anchor_brain_signals, positive_images, negative_images):
        positive_vision_state = self.encode_vision(positive_images).vision_state
        negative_vision_state = self.encode_vision(negative_images).vision_state

        anchor_brain_state = self.encode_brain(anchor_brain_signals).brain_state

        loss = self.triplet_loss(anchor_brain_state, positive_vision_state, negative_vision_state)
        return NAVFModelOutputs(loss=loss,
                                vision_state=[positive_vision_state, negative_vision_state],
                                brain_state=anchor_brain_state
                                )

    def text_brain_task(self, anchor_brain_signals, positive_text, negative_text, labels=None, **kwargs):
        if "step_one" in kwargs and kwargs["step_one"]:
            brain_state = self.encode_brain(anchor_brain_signals, **kwargs).brain_state
            brain_logits = self.brain_classifier(brain_state)
            loss = self.cross_entropy(brain_logits, labels)
            return NAVFModelOutputs(loss=loss,
                                    brain_logits=brain_logits,
                                    brain_state=brain_state
                                    )
        else:
            positive_text_state = self.encode_text(positive_text,
                                                   attention_mask=kwargs["positive_attention_mask"]).text_state
            negative_text_state = self.encode_text(negative_text,
                                                   attention_mask=kwargs["negative_attention_mask"]).text_state

            anchor_brain_state = self.encode_brain(anchor_brain_signals, **kwargs).brain_state

            loss = self.triplet_loss(anchor_brain_state, positive_text_state, negative_text_state)
            return NAVFModelOutputs(loss=loss,
                                    text_state=[positive_text_state, negative_text_state],
                                    brain_state=anchor_brain_state
                                    )


class ModifiedNAVF(BaseModel):
    def __init__(self, config: NAVFConfig,
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

    def text_task(self, text, **kwargs):
        text_state = self.encode_text(text, **kwargs).text_state
        logits = self.text_classifier(text_state)
        loss = self.cross_entropy(logits, target=kwargs["labels"])
        return NAVFModelOutputs(loss=loss,
                                logits=logits,
                                text_state=text_state
                                )
    
    def vision_task(self, images, **kwargs):
        vision_state = self.encode_vision(images, **kwargs).vision_state
        logits = self.vision_classifier(vision_state)
        loss = self.cross_entropy(logits, target=kwargs["labels"])
        return NAVFModelOutputs(loss=loss,
                                logits=logits,
                                vision_state=vision_state
                                )

    def brain_task(self, brain_signals, **kwargs):
        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
        logits = self.brain_classifier(brain_state)
        loss = self.cross_entropy(logits, target=kwargs["labels"])
        return NAVFModelOutputs(loss=loss,
                                logits=logits,
                                brain_state=brain_state
                                )

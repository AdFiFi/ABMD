from copy import deepcopy
from einops import rearrange
import torch
from torch import nn


class ModelConfig(object):
    def __init__(self,
                 vision_config=None,
                 text_config=None,
                 audio_config=None,
                 brain_config=None,
                 d_model=1024,
                 d_hidden=2048,
                 modality='text-brain',
                 class_weight=None,
                 label_smoothing=0,
                 num_classes=2,
                 use_temporal=False,
                 use_sequence=False):
        self.modality = modality
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing
        self.vision_config = vision_config
        self.text_config = text_config
        self.audio_config = audio_config
        self.brain_config = brain_config
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_classes = num_classes
        self.use_temporal = use_temporal
        self.use_sequence = use_sequence

    def dict(self):
        config = deepcopy(self.__dict__)
        config['vision_config'] = self.vision_config.__dict__ if self.vision_config is not None else None
        config['text_config'] = self.text_config.__dict__ if self.text_config is not None else None
        config['audio_config'] = self.audio_config.__dict__ if self.audio_config is not None else None
        config['brain_config'] = self.brain_config.__dict__ if self.brain_config is not None else None
        return config


class ModelOutputs:
    def __init__(self,
                 logits=None,
                 loss=None,
                 feature=None,
                 brain_state=None,
                 text_state=None,
                 vision_state=None,
                 audio_state=None,
                 ):
        self.logits = logits
        self.loss = loss
        self.feature = feature
        self.brain_state = brain_state
        self.text_state = text_state
        self.vision_state = vision_state
        self.audio_state = audio_state


class BaseModel(nn.Module):
    def __init__(self, config: ModelConfig,
                 vision_encoder=None, text_encoder=None, audio_encoder=None, brain_encoder=None,
                 text_projector=None, vision_projector=None, audio_projector=None, brain_projector=None,
                 classifier="private"):
        super().__init__()
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0,
                                                 weight=torch.tensor(config.class_weight))

        if text_encoder is not None and "text" in self.config.modality:
            self.text_encoder = text_encoder
            if text_projector is None:
                self.text_projector = nn.Linear(config.text_config.d_model, config.d_model, bias=False)
            else:
                self.text_projector = text_projector
            if classifier == "private":
                self.text_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if vision_encoder is not None and "vision" in self.config.modality:
            self.vision_encoder = vision_encoder
            if vision_projector is None:
                self.vision_projector = nn.Linear(config.vision_config.d_model, config.d_model, bias=False)
            else:
                self.vision_projector = vision_projector
            if classifier == "private":
                self.vision_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if audio_encoder is not None and "audio" in self.config.modality:
            self.audio_encoder = audio_encoder
            if audio_projector is None:
                self.audio_projector = nn.Linear(config.audio_config.d_model, config.d_model, bias=False)
            else:
                self.audio_projector = audio_projector
            if classifier == "private":
                self.audio_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if brain_encoder is not None and "brain" in self.config.modality:
            self.brain_encoder = brain_encoder
            if brain_projector is None:
                self.brain_projector = nn.Linear(config.brain_config.d_model, config.d_model, bias=False)
            else:
                self.brain_projector = brain_projector
            if classifier == "private":
                self.brain_classifier = nn.Linear(config.d_model, self.config.num_classes)
        if classifier == "common":
            self.common_classifier = nn.Linear(config.d_model, self.config.num_classes)

    def forward(self, **kwargs):
        if self.config.modality == "text":
            return self.text_task(**kwargs)

        elif self.config.modality == "vision":
            return self.vision_task(**kwargs)

        elif self.config.modality == "audio":
            return self.audio_task(**kwargs)

        elif self.config.modality == "brain":
            return self.brain_task(**kwargs)

        elif self.config.modality == "text-brain":
            return self.text_brain_task(**kwargs)

        elif self.config.modality == "vision-brain":
            return self.vision_brain_task(**kwargs)

        elif self.config.modality == "audio-brain":
            return self.audio_brain_task(**kwargs)

        elif self.config.modality == "text-vision-audio-brain":
            return self.text_vision_audio_brain_task(**kwargs)

    def encode_text(self, text, **kwargs):
        text_outputs = self.text_encoder(text, **kwargs)
        if not self.config.use_sequence:
            if self.config.text_config.fusion == "flatten":
                text_state = torch.flatten(text_outputs.text_state, start_dim=1)
            elif self.config.text_config.fusion == "mean":
                text_state = text_outputs.text_state
            elif self.config.text_config.fusion == "first":
                text_state = text_outputs.text_state
            elif self.config.text_config.fusion == "last":
                text_state = text_outputs.text_state
            elif self.config.text_config.fusion == "attention":
                raise
            else:
                raise
        else:
            raise
        text_state = self.text_projector(text_state)
        text_outputs.text_state = text_state
        return text_outputs

    def encode_vision(self, images, **kwargs):
        vision_outputs = self.vision_encoder(images, **kwargs)
        vision_state = vision_outputs.vision_state
        vision_state = self.vision_projector(vision_state)
        vision_outputs.vision_state = vision_state
        return vision_outputs

    def encode_audio(self, audio, **kwargs):
        audio_outputs = self.audio_encoder(audio, **kwargs)
        audio_state = audio_outputs.audio_state
        # if self.config.use_temporal:
        #     audio_state = torch.flatten(audio_state, start_dim=1)
        audio_state = self.audio_projector(audio_state)
        audio_outputs.audio_state = audio_state
        return audio_outputs

    def encode_brain(self, brain_signals, **kwargs):
        brain_outputs = self.brain_encoder(brain_signals, **kwargs)
        if not self.config.use_temporal:
            if self.config.brain_config.fusion == "flatten":
                brain_state = torch.flatten(brain_outputs.brain_state, start_dim=1)
            elif self.config.brain_config.fusion == "mean":
                brain_state = brain_outputs.brain_state
            elif self.config.brain_config.fusion == "attention":
                raise
            else:
                raise
        else:
            raise
        brain_state = self.brain_projector(brain_state)
        brain_outputs.brain_state = brain_state
        return brain_outputs

    def text_task(self, labels, **kwargs):
        text_outputs = self.encode_text(**kwargs)
        logits = text_outputs.logits
        loss = self.cross_entropy(logits, labels)
        return ModelOutputs(logits=logits,
                            loss=loss,
                            text_state=text_outputs.text_state)

    def vision_task(self, labels, **kwargs):
        vision_outputs = self.encode_vision(**kwargs)
        logits = vision_outputs.logits
        loss = self.cross_entropy(logits, labels)
        return ModelOutputs(loss=loss,
                            logits=logits,
                            vision_state=vision_outputs.vision_state
                            )

    def audio_task(self, labels, **kwargs):
        audio_outputs = self.encode_audio(**kwargs)
        logits = audio_outputs.logits
        loss = self.cross_entropy(logits, labels)
        return ModelOutputs(loss=loss,
                            logits=logits,
                            audio_state=audio_outputs.audio_state
                            )

    def brain_task(self, labels, **kwargs):
        brain_outputs = self.encode_brain(**kwargs)
        logits = brain_outputs.logits
        loss = self.cross_entropy(logits, labels)
        return ModelOutputs(loss=loss,
                            logits=logits,
                            brain_state=brain_outputs.brain_state
                            )

    def text_brain_task(self, **kwargs):
        return

    def vision_brain_task(self, **kwargs):
        return

    def audio_brain_task(self, **kwargs):
        return

    def text_vision_audio_brain_task(self, **kwargs):
        return

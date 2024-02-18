import torch
from torch import nn

from .base import ModelConfig, ModelOutputs, BaseModel

"""
BMCL (Brain-Machine Coupled Learning) is a reproduction of paper:
Brain-Machine Coupled Learning Method for Facial Emotion Recognition
D. Liu, W. Dai, H. Zhang, X. Jin, J. Cao and W. Kong
IEEE Transactions on Pattern Analysis and Machine Intelligence 2023 

"""


class BMCLConfig(ModelConfig):
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
                 lam=0.5,
                 t=3,
                 k=5,
                 alpha=0.5,
                 beta=0.5,
                 gamma=1,
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
        self.t = t
        self.k = k
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class BMCLModelOutputs(ModelOutputs):
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
                 loss_sim=None,
                 loss_diff=None):
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
        self.loss_sim = loss_sim
        self.loss_diff = loss_diff


class CMDLoss(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def forward(self, tva_state, brain_state):
        mean_vision_state = tva_state.mean(0)
        mean_brain_state = brain_state.mean(0)
        central_moments_vision = tva_state - mean_vision_state
        central_moments_brain = brain_state - mean_brain_state
        dm = torch.linalg.norm(central_moments_vision - central_moments_brain)
        cmd = dm
        for i in range(self.k - 1):
            k_central_moments_vision = torch.pow(central_moments_vision, i).mean(0)
            k_central_moments_brain = torch.pow(central_moments_brain, i).mean(0)
            cmd = cmd + torch.linalg.norm(k_central_moments_vision - k_central_moments_brain)
        return cmd


class SoftSubspaceOrthogonalityConstraint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, common_tva_state, common_brain_state, private_tva_state, private_brain_state):
        return torch.linalg.norm(torch.mm(common_tva_state.t(), private_tva_state), ord='fro') + \
            torch.linalg.norm(torch.mm(common_brain_state.t(), private_brain_state), ord='fro') + \
            torch.linalg.norm(torch.mm(private_tva_state.t(), private_brain_state), ord='fro')


class ModifiedBMCL(BaseModel):
    def __init__(self, config: BMCLConfig,
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
                                                 # )
                                                 reduction='sum')
        self.loss_sim = CMDLoss(k=config.k)
        self.loss_diff = SoftSubspaceOrthogonalityConstraint()

        if text_encoder is not None and "text" in self.config.modality:
            self.text_private_channel = nn.Sequential(nn.Linear(config.d_model, config.d_hidden),
                                                      nn.Linear(config.d_hidden, config.d_model))
            self.text_classifier = nn.Linear(config.d_model*2, self.config.num_classes)

        if vision_encoder is not None and "vision" in self.config.modality:
            self.vision_private_channel = nn.Sequential(nn.Linear(config.d_model, config.d_hidden),
                                                        nn.Linear(config.d_hidden, config.d_model))
            self.vision_classifier = nn.Linear(config.d_model*2, self.config.num_classes)

        if audio_encoder is not None and "audio" in self.config.modality:
            self.audio_private_channel = nn.Sequential(nn.Linear(config.d_model, config.d_hidden),
                                                       nn.Linear(config.d_hidden, config.d_model))
            self.audio_classifier = nn.Linear(config.d_model*2, self.config.num_classes)

        if brain_encoder is not None and "brain" in self.config.modality:
            self.brain_private_channel = nn.Sequential(nn.Linear(config.d_model, config.d_hidden),
                                                       nn.Linear(config.d_hidden, config.d_model))
            self.brain_classifier = nn.Linear(config.d_model*2, self.config.num_classes)

        self.common_channel = nn.Sequential(nn.Linear(config.d_model, config.d_hidden),
                                            nn.Linear(config.d_hidden, config.d_model))

    def vision_brain_task(self, labels, images, brain_signals, **kwargs):
        vision_state = self.encode_vision(images, **kwargs).vision_state
        common_vision_state = self.common_channel(vision_state)
        private_vision_state = self.vision_private_channel(vision_state)
        vision_state = torch.concat([common_vision_state, private_vision_state], dim=-1)
        vision_logits = self.vision_classifier(vision_state)

        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
        common_brain_state = self.common_channel(brain_state)
        private_brain_state = self.brain_private_channel(brain_state)
        brain_state = torch.concat([common_brain_state, private_brain_state], dim=-1)
        brain_logits = self.brain_classifier(brain_state)

        loss = self.cross_entropy(vision_logits, labels)
        loss = loss + self.cross_entropy(brain_logits, labels)
        loss_sim = self.loss_sim(common_vision_state, common_brain_state)
        loss_diff = self.loss_diff(common_vision_state, common_brain_state, private_vision_state, private_brain_state)
        # loss = loss + self.config.alpha * loss_sim + self.config.beta * loss_diff
        return BMCLModelOutputs(loss=loss,
                                vision_logits=vision_logits,
                                brain_logits=brain_logits,
                                loss_sim=loss_sim,
                                loss_diff=loss_diff,
                                vision_state=vision_state,
                                brain_state=brain_state
                                )

    def text_brain_task(self, labels, text, brain_signals, **kwargs):
        text_state = self.encode_text(text, **kwargs).text_state
        common_text_state = self.common_channel(text_state)
        private_text_state = self.text_private_channel(text_state)
        text_state = torch.concat([common_text_state, private_text_state], dim=-1)
        text_logits = self.text_classifier(text_state)

        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
        common_brain_state = self.common_channel(brain_state)
        private_brain_state = self.brain_private_channel(brain_state)
        brain_state = torch.concat([common_brain_state, private_brain_state], dim=-1)
        brain_logits = self.brain_classifier(brain_state)

        loss = self.cross_entropy(text_logits, labels)
        loss = self.config.gamma * loss + self.cross_entropy(brain_logits, labels)
        loss_sim = self.loss_sim(common_text_state, common_brain_state)
        loss_diff = self.loss_diff(common_text_state, common_brain_state, private_text_state, private_brain_state)
        loss = loss + self.config.alpha * loss_sim + self.config.beta * loss_diff
        return BMCLModelOutputs(loss=loss,
                                text_logits=text_logits,
                                brain_logits=brain_logits,
                                loss_sim=loss_sim,
                                loss_diff=loss_diff,
                                text_state=text_state,
                                brain_state=brain_state
                                )


class ModifiedBMCLTwoSteps(BaseModel):
    def __init__(self, config: BMCLConfig,
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
                                                 # )
                                                 reduction='sum')
        self.loss_sim = CMDLoss(k=config.k)
        self.loss_diff = SoftSubspaceOrthogonalityConstraint()

        if text_encoder is not None and "text" in self.config.modality:
            self.text_private_channel = nn.Sequential(nn.Linear(config.d_model, config.d_hidden),
                                                      nn.Linear(config.d_hidden, config.d_model))
            self.text_classifier = nn.Linear(config.d_model*2, self.config.num_classes)

        if vision_encoder is not None and "vision" in self.config.modality:
            self.vision_private_channel = nn.Sequential(nn.Linear(config.d_model, config.d_hidden),
                                                        nn.Linear(config.d_hidden, config.d_model))
            self.vision_classifier = nn.Linear(config.d_model*2, self.config.num_classes)

        if audio_encoder is not None and "audio" in self.config.modality:
            self.audio_private_channel = nn.Sequential(nn.Linear(config.d_model, config.d_hidden),
                                                       nn.Linear(config.d_hidden, config.d_model))
            self.audio_classifier = nn.Linear(config.d_model*2, self.config.num_classes)

        if brain_encoder is not None and "brain" in self.config.modality:
            self.brain_private_channel = nn.Sequential(nn.Linear(config.d_model, config.d_hidden),
                                                       nn.Linear(config.d_hidden, config.d_model))
            self.brain_classifier = nn.Linear(config.d_model*2, self.config.num_classes)

        self.common_channel = nn.Sequential(nn.Linear(config.d_model, config.d_hidden),
                                            nn.Linear(config.d_hidden, config.d_model))

    def vision_brain_task(self, labels, images, brain_signals, **kwargs):
        vision_state = self.encode_vision(images, **kwargs).vision_state
        common_vision_state = self.common_channel(vision_state)
        private_vision_state = self.vision_private_channel(vision_state)
        vision_state = torch.concat([common_vision_state, private_vision_state], dim=-1)
        vision_logits = self.vision_classifier(vision_state)

        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
        common_brain_state = self.common_channel(brain_state)
        private_brain_state = self.brain_private_channel(brain_state)
        brain_state = torch.concat([common_brain_state, private_brain_state], dim=-1)
        brain_logits = self.brain_classifier(brain_state)

        loss = self.cross_entropy(vision_logits, labels)
        loss = loss + self.cross_entropy(brain_logits, labels)
        loss_sim = self.loss_sim(common_vision_state, common_brain_state)
        loss_diff = self.loss_diff(common_vision_state, common_brain_state, private_vision_state, private_brain_state)
        loss = loss + self.config.alpha * loss_sim + self.config.beta * loss_diff
        return BMCLModelOutputs(loss=loss,
                                vision_logits=vision_logits,
                                brain_logits=brain_logits,
                                loss_sim=loss_sim,
                                loss_diff=loss_diff,
                                vision_state=vision_state,
                                brain_state=brain_state
                                )

    def text_brain_task(self, labels, text, brain_signals, **kwargs):
        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
        common_brain_state = self.common_channel(brain_state)
        private_brain_state = self.brain_private_channel(brain_state)
        brain_state = torch.concat([common_brain_state, private_brain_state], dim=-1)
        brain_logits = self.brain_classifier(brain_state)

        if "step_one" in kwargs and kwargs["step_one"]:
            loss = self.cross_entropy(brain_logits, labels)
            return BMCLModelOutputs(loss=loss,
                                    brain_logits=brain_logits,
                                    brain_state=brain_state
                                    )
        else:
            text_state = self.encode_text(text, **kwargs).text_state
            common_text_state = self.common_channel(text_state)
            private_text_state = self.text_private_channel(text_state)
            text_state = torch.concat([common_text_state, private_text_state], dim=-1)
            text_logits = self.text_classifier(text_state)
            loss = self.cross_entropy(text_logits, labels)
            loss = self.config.gamma * loss + self.cross_entropy(brain_logits, labels)
            loss_sim = self.loss_sim(common_text_state, common_brain_state)
            loss_diff = self.loss_diff(common_text_state, common_brain_state, private_text_state, private_brain_state)
            loss = loss + self.config.alpha * loss_sim + self.config.beta * loss_diff
            return BMCLModelOutputs(loss=loss,
                                    text_logits=text_logits,
                                    brain_logits=brain_logits,
                                    loss_sim=loss_sim,
                                    loss_diff=loss_diff,
                                    text_state=text_state,
                                    brain_state=brain_state
                                    )

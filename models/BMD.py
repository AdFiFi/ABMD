import torch
import math
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np

from .base import ModelConfig, ModelOutputs, BaseModel

"""
BCD: Brain Machine Contrastive Decomposable Learning
"""


class BMDConfig(ModelConfig):
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
                 gamma=500,
                 m=0.999,
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
        self.d_drop = 0.2
        self.m = m
        self.sample_size = 5


class BMDModelOutputs(ModelOutputs):
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
                 loss_parallel=None,
                 loss_orthogonal=None
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
        self.loss_parallel = loss_parallel
        self.loss_orthogonal = loss_orthogonal


class VICReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.sim_coeff = nn.Parameter(torch.tensor([0.001], requires_grad=True))
        self.std_coeff = nn.Parameter(torch.tensor([0.001], requires_grad=True))
        self.cov_coeff = nn.Parameter(torch.tensor([0.001], requires_grad=True))

    def forward(self, x, y):
        repr_loss = self.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + self.off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
                self.sim_coeff * repr_loss
                + self.std_coeff * std_loss
                + self.cov_coeff * cov_loss
        )
        return loss

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class DistillLoss(nn.Module):
    def __init__(self, t=4):
        super().__init__()
        self.t = t

    def forward(self, vision_state, brain_state):
        vision_state = F.log_softmax(vision_state / self.t, dim=-1)
        brain_state = F.softmax(brain_state / self.t, dim=-1)
        loss = -torch.sum(vision_state * brain_state, dim=-1).mean()
        return loss


class ParallelLoss(nn.Module):
    def __init__(self, config: BMDConfig):
        super().__init__()
        # self.loss_fn = nn.CosineSimilarity()
        self.tva_norm = nn.InstanceNorm1d(config.d_model)
        self.brain_norm = nn.InstanceNorm1d(config.d_model)
        self.loss_fn = DistillLoss()
        # self.loss_fn = nn.MSELoss()

    # def forward(self, tva_state, brain_state):
    #     tva_state = self.tva_norm(tva_state)
    #     brain_state = self.brain_norm(brain_state)
    #     parallel_loss = self.loss_fn(tva_state, brain_state) + \
    #         self.loss_fn(brain_state, tva_state)
    #     return parallel_loss

    def forward(self, tva_state, brain_state):
        # parallel_loss = self.loss_fn(tva_state, brain_state)

        # tva_state = tva_state.expand(brain_state.shape)
        # parallel_loss = torch.linalg.norm(tva_state - brain_state, dim=(1, 2), ord='fro').mean()
        parallel_loss = torch.linalg.norm(tva_state - brain_state, ord='fro')

        # parallel_loss = (1 - torch.cosine_similarity(tva_state, brain_state)).mean()

        # tva_state = self.tva_norm(tva_state)
        # brain_state = self.brain_norm(brain_state)
        # # parallel_loss = parallel_loss + self.loss_fn(tva_state, brain_state)
        # parallel_loss = parallel_loss * torch.linalg.norm(tva_state - brain_state, ord='fro')
        return parallel_loss


class OrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    # def forward(self, orthogonal_tva_state, parallel_tva_state, orthogonal_brain_state, parallel_brain_state):
    #     return torch.linalg.norm(torch.mm(parallel_tva_state.t(), orthogonal_tva_state), ord='fro') + \
    #         torch.linalg.norm(torch.mm(parallel_brain_state.t(), orthogonal_brain_state), ord='fro') + \
    #         torch.linalg.norm(torch.mm(orthogonal_tva_state.t(), orthogonal_brain_state), ord='fro')

    def forward(self, orthogonal_tva_state, parallel_tva_state):
        return torch.linalg.norm(torch.mm(parallel_tva_state.t(), orthogonal_tva_state), ord='fro')


class VIB(nn.Module):
    def __init__(self, config: BMDConfig):
        super().__init__()
        self.sample_size = config.sample_size
        self.emb2mean = nn.Sequential(nn.Dropout(p=config.d_drop),
                                      nn.Linear(config.d_model, config.d_hidden),
                                      nn.Tanh(),
                                      nn.Dropout(p=config.d_drop),
                                      nn.Linear(config.d_hidden, config.d_model))
        self.emb2std = nn.Sequential(nn.Dropout(p=config.d_drop),
                                     nn.Linear(config.d_model, config.d_hidden),
                                     nn.Tanh(),
                                     nn.Dropout(p=config.d_drop),
                                     nn.Linear(config.d_hidden, config.d_model))
        self.mean_p = nn.Parameter(torch.randn(config.d_model))
        self.std_p = nn.Parameter(torch.randn(config.d_model))

    def forward(self, hidden_state):
        B, E = hidden_state.shape
        mean = self.emb2mean(hidden_state)
        std = torch.nn.functional.softplus(self.emb2std(hidden_state))

        mean_p = self.mean_p.view(1, -1).expand(B, -1)
        std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(B, -1))

        kl_loss = self.kl_div(mean, std, mean_p, std_p)
        hidden_state = self.re_parameterize(mean, std)
        return hidden_state, kl_loss

    @staticmethod
    def kl_div(mu_q, std_q, mu_p, std_p):
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logit_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logit_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logit_std_p - logit_std_q) * 0.5
        return kl_divergence.mean()

    def re_parameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1], device=mu.device)
        return mu + std * z


class IB(nn.Module):
    def __init__(self, config: BMDConfig):
        super().__init__()
        self.initial_value = 5.0
        self.std = nn.Sequential(nn.Dropout(p=config.d_drop),
                                 nn.Linear(config.d_model, config.d_hidden),
                                 nn.ReLU(),
                                 nn.Dropout(p=config.d_drop),
                                 nn.Linear(config.d_hidden, config.d_model))
        self.readout = nn.Linear(config.d_model, config.d_model//2)
        self.alpha = nn.Parameter(torch.full((config.d_model,), fill_value=self.initial_value))
        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None

        self.reset_alpha()

    @staticmethod
    def _sample_t(mu, noise_var):
        # log_noise_var = torch.clamp(log_noise_var, -10, 10)
        noise_std = noise_var.sqrt()
        eps = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps

    @staticmethod
    def _calc_capacity(mu, var):
        # KL[P(t|x)||Q(t)] where Q(t) is N(0,1)
        kl = -0.5 * (1 + torch.log(var) - mu ** 2 - var)
        return kl

    def reset_alpha(self):
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)
        return self.alpha

    def forward(self, x, **kwargs):
        # lamb = self.sigmoid(self.std(x))
        lamb = self.sigmoid(self.alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1])
        masked_mu = x * lamb
        masked_var = (1 - lamb) ** 2
        self.buffer_capacity = self._calc_capacity(masked_mu, masked_var)
        t = self._sample_t(masked_mu, masked_var)
        # t = self.readout(t)
        return t

    # def forward(self, x, y):
    #     alpha = self.readout(y)
    #     lamb = self.sigmoid(alpha)
    #     lamb = lamb.expand(x.shape[0], x.shape[1])
    #     masked_mu = x * lamb
    #     masked_var = (1 - lamb) ** 2
    #     self.buffer_capacity = self._calc_capacity(masked_mu, masked_var)
    #     t = self._sample_t(masked_mu, masked_var)
    #     return t


class BMD(BaseModel):
    def __init__(self, config: BMDConfig,
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
        self.vic_reg = VICReg()
        self.loss_parallel = ParallelLoss(config)
        self.loss_orthogonal = OrthogonalLoss()

        if text_encoder is not None and "text" in self.config.modality:
            # 初始化？ todo
            self.text_orthogonalization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                   nn.Linear(config.d_model, config.d_hidden),
                                                                   # nn.Tanh(),
                                                                   nn.Dropout(p=self.config.d_drop),
                                                                   nn.Linear(config.d_hidden, config.d_model))
            self.text_parallelization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                 nn.Linear(config.d_model, config.d_hidden),
                                                                 # nn.Tanh(),
                                                                 nn.Dropout(p=self.config.d_drop),
                                                                 nn.Linear(config.d_hidden, config.d_model))
            self.text_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if vision_encoder is not None and "vision" in self.config.modality:
            self.vision_orthogonalization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                     nn.Linear(config.d_model, config.d_hidden),
                                                                     nn.Tanh(),
                                                                     nn.Dropout(p=self.config.d_drop),
                                                                     nn.Linear(config.d_hidden, config.d_model))
            self.vision_parallelization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                   nn.Linear(config.d_model, config.d_hidden),
                                                                   nn.Tanh(),
                                                                   nn.Dropout(p=self.config.d_drop),
                                                                   nn.Linear(config.d_hidden, config.d_model))
            self.vision_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if audio_encoder is not None and "audio" in self.config.modality:
            self.audio_orthogonalization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                    nn.Linear(config.d_model, config.d_hidden),
                                                                    nn.Tanh(),
                                                                    nn.Dropout(p=self.config.d_drop),
                                                                    nn.Linear(config.d_hidden, config.d_model))
            self.audio_parallelization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                  nn.Linear(config.d_model, config.d_hidden),
                                                                  nn.Tanh(),
                                                                  nn.Dropout(p=self.config.d_drop),
                                                                  nn.Linear(config.d_hidden, config.d_model))
            self.audio_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if brain_encoder is not None and "brain" in self.config.modality:
            self.brain_orthogonalization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                    nn.Linear(config.d_model, config.d_hidden),
                                                                    # nn.Tanh(),
                                                                    nn.Dropout(p=self.config.d_drop),
                                                                    nn.Linear(config.d_hidden, config.d_model))
            self.brain_parallelization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                  nn.Linear(config.d_model, config.d_hidden),
                                                                  # nn.Tanh(),
                                                                  nn.Dropout(p=self.config.d_drop),
                                                                  nn.Linear(config.d_hidden, config.d_model))
            self.brain_classifier = nn.Linear(config.d_model, self.config.num_classes)

    def text_brain_task(self, text, brain_signals, labels, **kwargs):
        text_state = self.encode_text(text, **kwargs).text_state
        orthogonal_text_state = self.text_orthogonalization_decomposer(text_state)
        parallel_text_state = self.text_parallelization_decomposer(text_state)
        text_state = orthogonal_text_state + parallel_text_state
        text_logits = self.text_classifier(text_state)

        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
        orthogonal_brain_state = self.brain_orthogonalization_decomposer(brain_state)
        parallel_brain_state = self.brain_parallelization_decomposer(brain_state)
        brain_state = orthogonal_brain_state + parallel_brain_state
        brain_logits = self.brain_classifier(brain_state)

        loss = self.cross_entropy(text_logits, labels)
        loss = loss + self.cross_entropy(brain_logits, labels)
        loss_parallel = self.loss_parallel(parallel_text_state, parallel_brain_state)
        loss_orthogonal = self.loss_orthogonal(orthogonal_text_state, parallel_text_state,
                                               orthogonal_brain_state, parallel_brain_state)
        loss = loss + self.config.alpha * loss_parallel + self.config.beta * loss_orthogonal
        return BMDModelOutputs(loss=loss,
                               text_logits=text_logits,
                               brain_logits=brain_logits,
                               text_state=text_state,
                               brain_state=brain_state,
                               loss_parallel=loss_parallel,
                               loss_orthogonal=loss_orthogonal
                               )

    def text_task(self, text, **kwargs):
        text_state = self.encode_text(text, **kwargs).text_state
        orthogonal_text_state = self.text_orthogonalization_decomposer(text_state)
        parallel_text_state = self.text_parallelization_decomposer(text_state)
        text_state = torch.concat([orthogonal_text_state, parallel_text_state], dim=-1)
        text_logits = self.text_classifier(text_state)
        loss = self.cross_entropy(text_logits, target=kwargs["labels"])
        return BMDModelOutputs(loss=loss,
                               text_logits=text_logits,
                               text_state=text_state
                               )

    def brain_task(self, brain_signals, **kwargs):
        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
        orthogonal_brain_state = self.brain_orthogonalization_decomposer(brain_state)
        parallel_brain_state = self.brain_parallelization_decomposer(brain_state)
        brain_state = torch.concat([orthogonal_brain_state, parallel_brain_state], dim=-1)
        brain_logits = self.brain_classifier(brain_state)

        loss = self.cross_entropy(brain_logits, target=kwargs["labels"])

        return BMDModelOutputs(loss=loss,
                               brain_logits=brain_logits,
                               brain_state=brain_state
                               )


class BMDTwoSteps(BaseModel):
    def __init__(self, config: BMDConfig,
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
        self.vic_reg = VICReg()
        self.loss_parallel = ParallelLoss(config)
        self.loss_orthogonal = OrthogonalLoss()
        self.m = config.m

        if text_encoder is not None and "text" in self.config.modality:
            # 初始化？ todo
            self.text_orthogonalization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                   nn.Linear(config.d_model, config.d_hidden),
                                                                   # nn.Tanh(),
                                                                   nn.Dropout(p=self.config.d_drop),
                                                                   nn.Linear(config.d_hidden, config.d_model))
            self.text_parallelization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                 nn.Linear(config.d_model, config.d_hidden),
                                                                 # nn.Tanh(),
                                                                 nn.Dropout(p=self.config.d_drop),
                                                                 nn.Linear(config.d_hidden, config.d_model))
            self.text_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if vision_encoder is not None and "vision" in self.config.modality:
            self.vision_orthogonalization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                     nn.Linear(config.d_model, config.d_hidden),
                                                                     nn.Tanh(),
                                                                     nn.Dropout(p=self.config.d_drop),
                                                                     nn.Linear(config.d_hidden, config.d_model))
            self.vision_parallelization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                   nn.Linear(config.d_model, config.d_hidden),
                                                                   nn.Tanh(),
                                                                   nn.Dropout(p=self.config.d_drop),
                                                                   nn.Linear(config.d_hidden, config.d_model))
            self.vision_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if audio_encoder is not None and "audio" in self.config.modality:
            self.audio_orthogonalization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                    nn.Linear(config.d_model, config.d_hidden),
                                                                    nn.Tanh(),
                                                                    nn.Dropout(p=self.config.d_drop),
                                                                    nn.Linear(config.d_hidden, config.d_model))
            self.audio_parallelization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                  nn.Linear(config.d_model, config.d_hidden),
                                                                  nn.Tanh(),
                                                                  nn.Dropout(p=self.config.d_drop),
                                                                  nn.Linear(config.d_hidden, config.d_model))
            self.audio_classifier = nn.Linear(config.d_model, self.config.num_classes)

        if brain_encoder is not None and "brain" in self.config.modality:
            self.brain_orthogonalization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                    nn.Linear(config.d_model, config.d_hidden),
                                                                    # nn.Tanh(),
                                                                    nn.Dropout(p=self.config.d_drop),
                                                                    nn.Linear(config.d_hidden, config.d_model))
            self.brain_parallelization_decomposer = nn.Sequential(nn.Dropout(p=self.config.d_drop),
                                                                  nn.Linear(config.d_model, config.d_hidden),
                                                                  # nn.Tanh(),
                                                                  nn.Dropout(p=self.config.d_drop),
                                                                  nn.Linear(config.d_hidden, config.d_model))
            self.brain_classifier = nn.Linear(config.d_model, self.config.num_classes)
        self.delta = nn.Parameter(torch.rand(config.d_model), requires_grad=True)
        self.vib = VIB(config)
        self.ib = IB(config)
        self.ib_tva_o = IB(config)
        self.ib_tva_p = IB(config)
        # self.init_orthogonalization_decomposer()

    def text_brain_task(self, text, brain_signals, labels, **kwargs):
        if "step_one" in kwargs and kwargs["step_one"]:
            brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
            orthogonal_brain_state = self.brain_orthogonalization_decomposer(brain_state)
            parallel_brain_state = self.brain_parallelization_decomposer(brain_state)
            brain_state = orthogonal_brain_state + parallel_brain_state
            # brain_state, kl_loss = self.vib(brain_state)
            brain_logits = self.brain_classifier(brain_state)
            # brain_logits = brain_logits.mean(dim=0)
            # loss = self.cross_entropy(brain_logits, labels) + kl_loss
            loss = self.cross_entropy(brain_logits, labels)
            return BMDModelOutputs(loss=loss,
                                   brain_logits=brain_logits,
                                   brain_state=brain_state
                                   )
        else:
            brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
            text_state = self.encode_text(text, **kwargs).text_state

            # orthogonal_brain_state = self.brain_orthogonalization_decomposer(brain_state)
            # parallel_brain_state = self.brain_parallelization_decomposer(brain_state)
            # brain_state = orthogonal_brain_state + parallel_brain_state
            # brain_state = parallel_brain_state
            # star_brain_state = parallel_brain_state + torch.einsum("d, bd -> bd",
            #                                                        torch.softmax(self.delta, dim=-1),
            #                                                        orthogonal_brain_state)
            # brain_state, kl_loss = self.vib(brain_state)
            brain_state = self.ib(brain_state)
            brain_logits = self.brain_classifier(brain_state)
            # brain_logits = brain_logits.mean(dim=0)

            orthogonal_text_state = self.text_orthogonalization_decomposer(text_state)
            parallel_text_state = self.text_parallelization_decomposer(text_state)
            # orthogonal_text_state = self.ib_tva_o(text_state)
            # parallel_text_state = self.ib_tva_p(text_state)

            loss_orthogonal = self.loss_orthogonal(orthogonal_text_state, parallel_text_state)

            text_state = orthogonal_text_state + parallel_text_state
            # text_state = torch.concat([orthogonal_text_state, parallel_text_state], dim=-1)
            # text_state = parallel_text_state
            text_logits = self.text_classifier(text_state)

            loss = self.cross_entropy(text_logits, labels)
            # loss = loss + self.cross_entropy(brain_logits, labels)
            loss = loss + self.cross_entropy_brain(brain_logits, labels)
            loss_parallel = self.loss_parallel(parallel_text_state,
                                               brain_state)
            # loss_orthogonal = self.loss_orthogonal(orthogonal_text_state, parallel_text_state,
            #                                        orthogonal_brain_state, parallel_brain_state)


            # loss_orthogonal = self.loss_orthogonal(orthogonal_brain_state, parallel_brain_state)
            # loss = loss + self.config.alpha * loss_parallel + self.config.beta * loss_orthogonal + kl_loss
            loss = loss + self.config.alpha * loss_parallel + self.config.beta * loss_orthogonal
            return BMDModelOutputs(loss=loss,
                                   text_logits=text_logits,
                                   brain_logits=brain_logits,
                                   # text_state=text_state,
                                   text_state=torch.stack([orthogonal_text_state, parallel_text_state], dim=1),
                                   brain_state=brain_state,
                                   loss_parallel=loss_parallel,
                                   loss_orthogonal=loss_orthogonal
                                   )

    def text_task(self, text, **kwargs):
        text_state = self.encode_text(text, **kwargs).text_state
        orthogonal_text_state = self.text_orthogonalization_decomposer(text_state)
        parallel_text_state = self.text_parallelization_decomposer(text_state)
        text_state = torch.concat([orthogonal_text_state, parallel_text_state], dim=-1)
        text_logits = self.text_classifier(text_state)
        loss = self.cross_entropy(text_logits, target=kwargs["labels"])
        return BMDModelOutputs(loss=loss,
                               text_logits=text_logits,
                               text_state=text_state
                               )

    def brain_task(self, brain_signals, **kwargs):
        brain_state = self.encode_brain(brain_signals, **kwargs).brain_state
        orthogonal_brain_state = self.brain_orthogonalization_decomposer(brain_state)
        parallel_brain_state = self.brain_parallelization_decomposer(brain_state)
        brain_state = torch.concat([orthogonal_brain_state, parallel_brain_state], dim=-1)
        brain_logits = self.brain_classifier(brain_state)

        loss = self.cross_entropy(brain_logits, target=kwargs["labels"])

        return BMDModelOutputs(loss=loss,
                               brain_logits=brain_logits,
                               brain_state=brain_state
                               )

    @torch.no_grad()
    def _momentum_update_brain_parallelization_decomposer(self):
        """
        Momentum update of the key encoder
        """
        for param_s, param_t in zip(
                self.text_parallelization_decomposer.parameters(), self.brain_parallelization_decomposer.parameters()
        ):
            param_t.data = param_t.data * self.m + param_s.data * (1.0 - self.m)

    @torch.no_grad()
    def _update_text_parallelization_decomposer(self):
        """
        Momentum update of the key encoder
        """
        for param_s, param_t in zip(
                self.text_parallelization_decomposer.parameters(), self.brain_parallelization_decomposer.parameters()
        ):
            param_s.data = param_t.data

    def reset_brain_decomposer(self):
        def weight_init(m):
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(m.bias, -bound, bound)

        self.brain_parallelization_decomposer.apply(weight_init)
        self.brain_orthogonalization_decomposer.apply(weight_init)

    def reinit_text_decomposer(self):
        def weight_init(m):
            if isinstance(m, nn.Linear):
                init.zeros_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

        self.text_parallelization_decomposer = deepcopy(self.brain_parallelization_decomposer)
        self.text_orthogonalization_decomposer.apply(weight_init)

    def init_orthogonalization_decomposer(self):
        def weight_init(m):
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
        self.text_orthogonalization_decomposer[1].apply(weight_init)
        self.text_parallelization_decomposer[1].apply(weight_init)
        self.text_orthogonalization_decomposer[3].apply(weight_init)
        self.text_parallelization_decomposer[3].apply(weight_init)

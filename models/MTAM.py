import torch
from torch import nn
from scipy.stats import wasserstein_distance

from .base import ModelConfig, ModelOutputs, BaseModel


"""
This is a modified version of MTAM from paper:
An empirical exploration of cross-domain alignment between language and electroencephalogram
W. Han, J. Qiu, J. Zhu, M. Xu, D. Weber, B. Li and D. Zhao
arXiv preprint arXiv:2208.06348 2022 

Code is mostly based on the code at https://github.com/Jason-Qiu/EEG_Language_Alignment. Functions remain the same.

"""


class MTAMConfig(ModelConfig):
    def __init__(self,
                 text_config=None,
                 brain_config=None,
                 d_model=16,
                 d_hidden=32,
                 modality='text-brain',
                 class_weight=None,
                 label_smoothing=0,
                 dropout=0.3,
                 num_classes=2,
                 cca_weight=1,
                 wd_weight=1,
                 use_all_singular_values=False,
                 use_temporal=False,
                 use_sequence=False
                 ):
        super().__init__(text_config=text_config,
                         brain_config=brain_config,
                         d_model=d_model,
                         d_hidden=d_hidden,
                         modality=modality,
                         class_weight=class_weight,
                         label_smoothing=label_smoothing,
                         num_classes=num_classes,
                         use_temporal=use_temporal,
                         use_sequence=use_sequence)
        self.dropout = dropout
        self.num_classes = num_classes
        self.cca_weight = cca_weight
        self.wd_weight = wd_weight
        self.use_all_singular_values = use_all_singular_values


class MTAMModelOutput(ModelOutputs):
    def __init__(self,
                 logits=None,
                 loss=None,
                 feature=None,
                 brain_state=None,
                 text_state=None,
                 cca_loss=None,
                 wd_loss=None,
                 ):
        super().__init__(logits=logits,
                         loss=loss,
                         feature=feature,
                         brain_state=brain_state,
                         text_state=text_state)
        self.cca_loss = cca_loss
        self.wd_loss = wd_loss


class CCALoss(nn.Module):
    def __init__(self, config: MTAMConfig):
        super().__init__()
        self.outdim_size = config.num_classes
        self.use_all_singular_values = config.use_all_singular_values

    def forward(self, H1, H2):
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()

        o1 = H1.size(0)
        o2 = H2.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=H1.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=H1.device)

        # Workaround !!! USE BATCH > 16
        # [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        # [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        # Newest way but not debugged
        [D1, V1] = torch.linalg.eig(SigmaHat11)
        [D2, V2] = torch.linalg.eig(SigmaHat22)

        # Original
        # [D1, V1] = torch.eig(SigmaHat11, eigenvectors=True)
        # [D2, V2] = torch.eig(SigmaHat22, eigenvectors=True)

        D1 = D1.unsqueeze(1)
        posInd1 = torch.gt(abs(D1[:, 0]), eps).nonzero()[:, 0]
        D1 = D1[posInd1, 0]
        V1 = V1[:, posInd1]
        D1 = torch.squeeze(D1)  # Remove extra dimensions from D1
        # Reshape D1 to 1-dimensional tensor
        D2 = D2.unsqueeze(1)
        # posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        posInd2 = torch.gt(abs(D2[:, 0]), eps).nonzero()[:, 0]
        D2 = D2[posInd2, 0]
        V2 = V2[:, posInd2]
        D2 = torch.squeeze(D2)

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12.to(torch.complex64)), SigmaHat22RootInv)

        if self.use_all_singular_values:

            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))

        else:

            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0]) * r1).to(H1.device))
            # U, V = torch.symeig(trace_TT, eigenvectors=True)
            U, V = torch.linalg.eigh(trace_TT)
            U = torch.where(U > eps, U, (torch.ones(U.shape).float() * eps).to(H1.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr


class WDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, text_embed, eeg_embed):
        return torch.tensor(wasserstein_distance(text_embed.cpu().detach().numpy().flatten(),
                                                 eeg_embed.cpu().detach().numpy().flatten()),
                            requires_grad=True, device=text_embed.device)


class ModifiedMTAM(BaseModel):
    def __init__(self,
                 config: MTAMConfig,
                 brain_encoder=None,
                 text_encoder=None,
                 audio_encoder=None,
                 vision_encoder=None):

        super().__init__(config=config, 
                         vision_encoder=vision_encoder, 
                         text_encoder=text_encoder,
                         audio_encoder=audio_encoder,
                         brain_encoder=brain_encoder,
                         classifier="common")
        self.config = config
        if self.config.modality == "text-brain":
            self.linear1_cov_fusion = nn.Conv1d(config.text_config.d_feature_text + config.brain_config.d_feature_brain,
                                                1, kernel_size=1)
            # self.linear_fusion = nn.Linear(config.text_config.d_model + config.brain_config.d_model, config.d_model)

        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                                 weight=torch.tensor(config.class_weight),
                                                 reduction='sum')
        self.cca_loss = CCALoss(config)
        self.wd_loss = WDLoss()

    def text_task(self, text, labels):
        b_text, l_text = text.size()
        src_pos_text = torch.LongTensor(
            [list(range(1, l_text + 1)) for _ in range(b_text)]).to(text.device)

        text_outputs = self.encode_text(text=text, src_pos_text=src_pos_text)
        loss = self.cross_entropy(text_outputs.logits, labels)
        return MTAMModelOutput(logits=text_outputs.logits,
                               loss=loss,
                               text_state=text_outputs.text_state)

    def brain_task(self, brain_signals, labels):
        b_eeg, l_eeg = brain_signals.size()
        src_pos_eeg = torch.LongTensor(
            [list(range(1, l_eeg + 1)) for _ in range(b_eeg)]).to(brain_signals.device)

        brain_outputs = self.encode_brain(brain_signals=brain_signals, src_pos_eeg=src_pos_eeg)
        loss = self.cross_entropy(brain_outputs.logits, labels)
        return MTAMModelOutput(logits=brain_outputs.logits,
                               loss=loss,
                               brain_state=brain_outputs.brain_state)

    def text_brain_task(self, text, brain_signals, labels):
        text_outputs = self.text_encoder(text)
        brain_outputs = self.brain_encoder(brain_signals)

        concat_enc = torch.cat((text_outputs.hidden_state, brain_outputs.hidden_state), dim=1)

        res = self.linear1_cov_fusion(concat_enc)
        # res = self.linear_fusion(concat_enc)
        res = res.contiguous().view(res.size()[0], -1)
        logits = self.common_classifier(res)

        loss = self.cross_entropy(logits, labels)
        cca_loss = self.cca_loss(text_outputs.text_state, brain_outputs.brain_state)
        wd_loss = self.wd_loss(text_outputs.text_state, brain_outputs.brain_state)
        loss = loss + self.config.cca_weight * cca_loss + self.config.wd_weight * wd_loss
        return MTAMModelOutput(logits=logits,
                               loss=loss,
                               cca_loss=cca_loss,
                               wd_loss=wd_loss,
                               text_state=text_outputs.text_state,
                               brain_state=brain_outputs.brain_state)

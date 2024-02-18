from .Layers import *
from utils import get_sinusoid_encoding_table, get_non_pad_mask, get_attn_key_pad_mask
from ..base import *
KS = 3


"""
MTAMEEGEncoder is a eeg encoder designed by:
An empirical exploration of cross-domain alignment between language and electroencephalogram
W. Han, J. Qiu, J. Zhu, M. Xu, D. Weber, B. Li and D. Zhao
arXiv preprint arXiv:2208.06348 2022 

Code is mostly based on the code at https://github.com/Jason-Qiu/EEG_Language_Alignment. Functions remain the same.

"""


class MTAMEEGEncoderConfig(BrainEncoderConfig):
    def __init__(self,
                 d_model=16,
                 d_hidden=32,
                 num_classes=2,
                 d_feature_brain=832,
                 n_layers=1,
                 n_head=5,
                 d_k=64,
                 d_v=64,
                 dropout=0.3,
                 fusion="none"
                 ):
        super().__init__(d_model=d_model,
                         d_hidden=d_hidden,
                         num_classes=num_classes,
                         dropout=dropout,
                         use_temporal=False,
                         fusion=fusion)
        self.d_feature_brain = d_feature_brain
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v


class MTAMEEGEncoder(nn.Module):
    def __init__(self, config: MTAMEEGEncoderConfig):
        super().__init__()
        self.config = config
        n_position = config.d_feature_brain + 1
        self.src_word_emb = nn.Conv1d(1, config.d_model, kernel_size=KS, padding=int((KS - 1) / 2))

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, config.d_model, padding_idx=0),
            freeze=True)

        self.eeg_layer_stack = nn.ModuleList([
            EncoderLayer(config.d_model, config.d_hidden, config.n_head, config.d_k, config.d_v, config.dropout,
                         config.d_feature_brain)
            for _ in range(config.n_layers)])
        self.brain_projector = nn.Conv1d(config.d_feature_brain, 1, kernel_size=1)
        self.brain_classifier = nn.Linear(config.d_model, self.config.num_classes)

    def forward(self, src_seq, **kwargs):
        b_eeg, l_eeg = src_seq.size()
        src_pos = torch.LongTensor(
            [list(range(1, l_eeg + 1)) for _ in range(b_eeg)]).to(src_seq.device)
        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        enc_output = src_seq.unsqueeze(1)
        enc_output = self.src_word_emb(enc_output)
        enc_output = enc_output.transpose(1, 2)
        enc_output.add_(self.position_enc(src_pos))

        for enc_layer in self.eeg_layer_stack:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        projected_enc_output = self.brain_projector(enc_output)
        projected_enc_output = projected_enc_output.contiguous().view(projected_enc_output.size()[0], -1)
        logits = self.brain_classifier(projected_enc_output)
        return BrainEncoderOutputs(logits=logits,
                                   brain_state=projected_enc_output,
                                   hidden_state=enc_output)

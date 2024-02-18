from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..base import *


class LSTMConfig(TextEncoderConfig):
    def __init__(self,
                 mode='supervised',
                 class_weight=None,
                 num_classes=2,
                 d_hidden=1024):
        super().__init__(mode=mode,
                         class_weight=class_weight,
                         num_classes=num_classes,
                         d_hidden=d_hidden)


class LSTM(nn.Module):
    def __init__(self, input_dim=840, d_hidden=256, output_dim=3, num_layers=1):
        super(LSTM, self).__init__()
        self.d_hidden = d_hidden
        self.lstm = nn.LSTM(input_dim, d_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.hidden2sentiment = nn.Linear(d_hidden * 2, output_dim)

    def forward(self, x_packed):
        lstm_out, _ = self.lstm(x_packed)
        last_hidden_state = pad_packed_sequence(lstm_out, batch_first=True)[0][:, -1, :]
        out = self.hidden2sentiment(last_hidden_state)
        return out

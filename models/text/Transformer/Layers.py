import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertEncoder, BertLayer
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaLayer
from transformers.models.bart.modeling_bart import BartDecoder, BartDecoderLayer


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_token_tensor = hidden_states.mean(dim=1)
        # pooled_output = self.dropout(mean_token_tensor)
        pooled_output = self.dense(mean_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertLayers(BertEncoder):
    def __init__(self, config):
        super().__init__(config=config)
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.extra_layer_num)])


class RoBertaLayers(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config=config)
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.extra_layer_num)])


class BartLayers(BartDecoder):
    def __init__(self, config):
        super().__init__(config=config)
        self.layer = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.extra_layer_num)])

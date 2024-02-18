import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('[DEBUG] input size:', x.size())
        # print('[DEBUG] positional embedding size:', self.pe.size())
        x = x + self.pe[:x.size(0), :]
        # print('[DEBUG] output x with pe size:', x.size())
        return self.dropout(x)


class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, decoder_embedding_size=1024, additional_encoder_nhead=8,
                 additional_encoder_dim_feedforward=2048):
        super(BrainTranslator, self).__init__()
        self.pretrained = pretrained_layers
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,
                                                                   dim_feedforward=additional_encoder_dim_feedforward,
                                                                   batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)

        # self.positional_embedding = PositionalEncoding(in_feature)

        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):

        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch)

        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)

        # encoded_embedding = self.additional_encoder(input_embeddings_batch)
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        out = self.pretrained(inputs_embeds=encoded_embedding, attention_mask=input_masks_batch, return_dict=True,
                              labels=target_ids_batch_converted)

        return out

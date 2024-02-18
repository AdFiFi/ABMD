from copy import deepcopy

from torch import nn
from torch.nn import functional as F


class EEG2TextConfig(object):
    def __init__(self, vision_config, brain_config, d_model=2048, d_hidden=512, mode='coupled', class_weight=None,
                 label_smoothing=0, lam=0.5, t=3, k=5, alpha=0.05, beta=0.0005):
        self.mode = mode
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing
        self.t = t
        self.k = k
        self.lam = lam
        self.vision_config = vision_config
        self.brain_config = brain_config
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.alpha = alpha
        self.beta = beta

    def dict(self):
        config = deepcopy(self.__dict__)
        config['vision_config'] = self.vision_config.__dict__ if self.vision_config is not None else None
        config['brain_config'] = self.brain_config.__dict__ if self.brain_config is not None else None
        return config


class EEG2TextModelOutputs:
    def __init__(self, logits=None, loss=None, loss_sim=None, loss_diff=None):
        self.logits = logits
        self.loss = loss
        self.loss_sim = loss_sim
        self.loss_diff = loss_diff


class EEG2Text(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, d_model=1024, additional_encoder_nhead=8,
                 additional_encoder_dim_feedforward=2048, num_labels=3):
        super(EEG2Text, self).__init__()

        self.pretrained_generator = pretrained_layers
        # additional transformer encoder, following BART paper about
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,
                                                                   dim_feedforward=additional_encoder_dim_feedforward,
                                                                   batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        self.fc1 = nn.Linear(in_feature, d_model)
        self.num_labels = num_labels

        self.pooler = Pooler(d_model)
        self.classifier = BartClassificationHead(input_dim=d_model, inner_dim=d_model, num_classes=num_labels,
                                                 pooler_dropout=pretrained_layers.config.classifier_dropout)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,
                sentiment_labels):
        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        LMoutput = self.pretrained_generator(inputs_embeds=encoded_embedding, attention_mask=input_masks_batch,
                                             return_dict=True, labels=target_ids_batch_converted,
                                             output_hidden_states=True)
        hidden_states = LMoutput.decoder_hidden_states
        last_hidden_states = hidden_states[-1]
        sentence_representation = self.pooler(last_hidden_states)

        classification_logits = self.classifier(sentence_representation)
        loss_fct = nn.CrossEntropyLoss()
        classification_loss = loss_fct(classification_logits.view(-1, self.num_labels), sentiment_labels.view(-1))
        classification_output = {'loss': classification_loss, 'logits': classification_logits}
        return LMoutput, classification_output

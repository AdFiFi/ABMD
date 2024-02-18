from transformers import (
    BartModel, BartConfig as BartBaseConfig,
    BertModel, BertConfig as BertBaseConfig,
    RobertaModel, RobertaConfig as RobertaBaseConfig)
from transformers.models.bart.modeling_bart import BartClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from .Layers import *
from ..base import *


class TransformerOutPuts(TextEncoderOutputs):
    def __init__(self, logits,
                 text_state=None,
                 hidden_state=None,
                 last_hidden_state=None):
        super().__init__(logits=logits,
                         text_state=text_state,
                         hidden_state=hidden_state)
        self.last_hidden_state = last_hidden_state


class BertConfig(BertBaseConfig, TextEncoderConfig):
    def __init__(self,
                 mode='supervised',
                 class_weight=None,
                 num_classes=2,
                 fusion="first",
                 extra_layer_num=0):
        super().__init__(num_labels=num_classes)
        TextEncoderConfig.__init__(self, class_weight=class_weight,
                                   num_classes=num_classes,
                                   mode=mode,
                                   d_model=self.hidden_size,
                                   d_hidden=self.intermediate_size,
                                   fusion=fusion)
        self.extra_layer_num = extra_layer_num


class Bert(nn.Module):
    def __init__(self, config: BertConfig, pretrained_checkpoint=None):
        super(Bert, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained('/data/models/bert-base-uncase',
                                              num_labels=config.num_labels)
        self.text_classifier = nn.Linear(config.hidden_size, config.num_labels)

        if pretrained_checkpoint is not None:
            self.bert.load_state_dict(torch.load(pretrained_checkpoint))
        if self.config.fusion == "mean":
            self.pooler = BertPooler(config)
        if self.config.extra_layer_num:
            self.bert_layers = BertLayers(config)
        else:
            self.bert_layers = None
        self.bert.requires_grad = False

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            output_attentions=True,
                            output_hidden_states=True)
        if self.bert_layers is not None:
            extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.shape)
            outputs = self.bert_layers(hidden_states=outputs.last_hidden_state,
                                       attention_mask=extended_attention_mask,
                                       return_dict=True,
                                       output_attentions=True,
                                       output_hidden_states=True)
        if self.config.fusion == "first":
            pooled_output = outputs.pooler_output
        elif self.config.fusion == "mean":
            pooled_output = self.pooler(outputs.last_hidden_state)
        else:
            raise
        logits = self.text_classifier(pooled_output)
        return TransformerOutPuts(logits=logits,
                                  text_state=pooled_output,
                                  hidden_state=outputs.last_hidden_state)


class RoBertaConfig(RobertaBaseConfig, TextEncoderConfig):
    def __init__(self,
                 mode='supervised',
                 class_weight=None,
                 num_classes=2,
                 fusion="first",
                 extra_layer_num=0
                 ):
        super().__init__(num_labels=num_classes)
        TextEncoderConfig.__init__(self, class_weight=class_weight,
                                   num_classes=num_classes,
                                   mode=mode,
                                   d_model=self.hidden_size,
                                   d_hidden=self.intermediate_size,
                                   fusion=fusion)
        self.extra_layer_num = extra_layer_num


class RoBerta(nn.Module):
    def __init__(self, config: RoBertaConfig, pretrained_checkpoint=None):
        super(RoBerta, self).__init__()
        self.config = config
        self.roberta = RobertaModel.from_pretrained('/data/models/roberta-base', num_labels=3)
        self.text_classifier = RobertaClassificationHead(config)

        if pretrained_checkpoint is not None:
            self.roberta.load_state_dict(torch.load(pretrained_checkpoint))
        if self.config.fusion == "mean":
            self.pooler = BertPooler(config)
        if self.config.extra_layer_num:
            self.roberta_layers = RoBertaLayers(config)
        else:
            self.roberta_layers = None
        self.roberta.requires_grad = False

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               return_dict=True,
                               output_attentions=True,
                               output_hidden_states=True)
        if self.roberta_layers is not None:
            extended_attention_mask = self.roberta.get_extended_attention_mask(attention_mask, input_ids.shape)
            outputs = self.roberta_layers(hidden_states=outputs.last_hidden_state,
                                          attention_mask=extended_attention_mask,
                                          return_dict=True,
                                          output_attentions=True,
                                          output_hidden_states=True)
        if self.config.fusion == "first":
            pooled_output = outputs.last_hidden_state
        elif self.config.fusion == "mean":
            pooled_output = self.pooler(outputs.last_hidden_state)
            pooled_output = pooled_output.unsqueeze(1)
        else:
            raise
        logits = self.text_classifier(pooled_output)
        return TransformerOutPuts(logits=logits,
                                  text_state=pooled_output.squeeze(1),
                                  hidden_state=outputs.last_hidden_state)


class BartConfig(BartBaseConfig, TextEncoderConfig):
    def __init__(self,
                 mode='supervised',
                 class_weight=None,
                 num_classes=2,
                 fusion="last",
                 extra_layer_num=0):
        c = BartBaseConfig.from_pretrained('/data/models/bart-base')
        c.num_labels = num_classes
        c.torch_dtype = None
        super().__init__(**c.to_dict())
        # self.from_pretrained('/data/models/bart-base')
        TextEncoderConfig.__init__(self, class_weight=class_weight,
                                   num_classes=num_classes,
                                   mode=mode,
                                   d_model=self.hidden_size,
                                   d_hidden=self.decoder_ffn_dim,
                                   fusion=fusion)
        self.extra_layer_num = extra_layer_num


class Bart(nn.Module):
    def __init__(self, config: BartConfig, pretrained_checkpoint=None):
        super(Bart, self).__init__()
        self.config = config
        self.bart = BartModel.from_pretrained('/data/models/bart-base', num_labels=3)
        self.text_classifier = BartClassificationHead(config.d_model,
                                                      config.d_model,
                                                      config.num_labels,
                                                      config.classifier_dropout)
        if pretrained_checkpoint is not None:
            self.bart.load_state_dict(torch.load(pretrained_checkpoint))
        self.bart.requires_grad = False

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bart(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            output_attentions=True,
                            output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                                  :, -1, :
                                  ]
        logits = self.text_classifier(sentence_representation)
        return TransformerOutPuts(logits=logits,
                                  text_state=sentence_representation,
                                  hidden_state=outputs.last_hidden_state)

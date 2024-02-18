from .NAVF_trainer import *


class CLIPTrainer(NAVFTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super().__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_pretrain_inputs_kwargs(self, inputs):
        tokens = inputs["tokens"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        if self.args.text_encoder in ["MTAMTextEncoder"]:
            with torch.no_grad():
                outputs = self.text_encoder(tokens, attention_mask)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

            embeddings = zscore(embeddings)
            text_embedding = torch.tensor(embeddings, dtype=torch.float32)
            if self.args.brain_encoder in ["MTAMEEGEncoder"]:
                return {"brain_signals": inputs["mean_time_series"].to(self.device),
                        "text": text_embedding.to(self.device)
                        }
            elif self.args.brain_encoder in ["BNT", "BrainNetGNN", "Graphormer"]:
                return {"text": text_embedding.to(self.device),
                        "brain_signals": inputs["correlation"].to(self.device)
                        }
            else:
                return {"text": text_embedding.to(self.device),
                        "brain_signals": inputs["time_series"].to(self.device)
                        }
        else:
            if self.args.brain_encoder in ["MTAMEEGEncoder"]:
                return {"brain_signals": inputs["mean_time_series"].to(self.device),
                        "text": tokens.to(self.device),
                        "attention_mask": attention_mask.to(self.device)
                        }
            elif self.args.brain_encoder in ["BNT", "BrainNetGNN", "Graphormer"]:
                return {"brain_signals": inputs["correlation"].to(self.device),
                        "text": tokens.to(self.device),
                        "attention_mask": attention_mask.to(self.device)
                        }
            else:
                return {"brain_signals": inputs["time_series"].to(self.device),
                        "text": tokens.to(self.device),
                        "attention_mask": attention_mask.to(self.device)
                        }

    def prepare_inputs_kwargs(self, inputs):
        if self.model_config.modality == "text":
            tokens = inputs["tokens"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            if self.args.text_encoder in ["Bert", "Bart", "RoBerta"]:
                return {"text": tokens.to(self.device),
                        "attention_mask": attention_mask,
                        "labels": inputs["labels"].to(self.device)
                        }
            elif self.args.text_encoder == "MTAMEEGEncoder":
                with torch.no_grad():
                    outputs = self.text_encoder(tokens, attention_mask)
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

                embeddings = zscore(embeddings)
                text_embedding = torch.tensor(embeddings, dtype=torch.float32)
                return {"text": text_embedding.to(self.device),
                        "labels": inputs["labels"].to(self.device)
                            }
            else:
                return {"text": tokens.to(self.device),
                        "attention_mask": attention_mask.to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }
        elif self.model_config.modality == "brain":
            if self.args.brain_encoder in ["MTAMEEGEncoder"]:
                return {"brain_signals": inputs["mean_time_series"].to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }
            elif self.args.brain_encoder in ["BNT", "BrainNetGNN", "Graphormer"]:
                return {"brain_signals": inputs["correlation"].to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }
            else:
                return {"brain_signals": inputs["time_series"].to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }

    def init_model_config(self):
        if self.args.modality == 'vision-brain':
            brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
            vision_encoder, vision_config = init_vision_encoder_config(self.args, self.data_config)
            clip_config = CLIPConfig(brain_config=brain_config,
                                     vision_config=vision_config,
                                     modality=self.args.modality)
            model = PretrainedCLIP(config=clip_config,
                                   brain_encoder=brain_encoder,
                                   vision_encoder=vision_encoder)
            return model, clip_config
        elif self.args.modality == 'text-brain':
            brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
            text_encoder, text_config = init_text_encoder_config(self.args, self.data_config)
            clip_config = CLIPConfig(brain_config=brain_config,
                                     text_config=text_config,
                                     d_model=self.args.d_model,
                                     d_hidden=self.args.d_hidden,
                                     modality=self.args.modality)
            model = PretrainedCLIP(config=clip_config,
                                   brain_encoder=brain_encoder,
                                   text_encoder=text_encoder)
            return model, clip_config
        else:
            raise "Modality not supported"

    def reinit_model_config(self, finetune_modality="text"):
        self.load_model(pretrain=True)
        if self.args.modality == 'vision-brain':
            clip_config = CLIPConfig(brain_config=self.model_config.brain_config,
                                     vision_config=self.model_config.vision_config,
                                     modality=finetune_modality,
                                     num_classes=self.data_config.num_classes,
                                     class_weight=self.data_config.class_weight,
                                     label_smoothing=self.args.label_smoothing)
            model = ModifiedCLIPFewShot(config=clip_config,
                                        brain_encoder=self.model.brain_encoder,
                                        brain_projector=self.model.brain_projector,
                                        vision_encoder=self.model.vision_encoder,
                                        vision_projector=self.model.vision_projector,
                                        )
            self.model = model
            self.model_config = clip_config
        elif self.args.modality == 'text-brain':
            clip_config = CLIPConfig(brain_config=self.model_config.brain_config,
                                     text_config=self.model_config.text_config,
                                     d_model=self.args.d_model,
                                     d_hidden=self.args.d_hidden,
                                     num_classes=self.data_config.num_classes,
                                     modality=finetune_modality,
                                     class_weight=self.data_config.class_weight,
                                     label_smoothing=self.args.label_smoothing)
            model = ModifiedCLIPFewShot(config=clip_config,
                                        brain_encoder=self.model.brain_encoder,
                                        brain_projector=self.model.brain_projector,
                                        text_encoder=self.model.text_encoder,
                                        text_projector=self.model.text_projector)
            self.model = model
            self.model_config = clip_config
        else:
            raise "Modality not supported"
        if self.args.do_parallel:
            self.device = f'cuda:{self.local_rank}' \
                if self.args.device != 'cpu' and torch.cuda.is_available() else self.args.device
            self.model = model.to(self.args.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   find_unused_parameters=True)
        else:
            self.device = f'cuda' \
                if self.args.device != 'cpu' and torch.cuda.is_available() else self.args.device
            self.model = model.to(self.args.device)
        # self.model = torch.compile(model, dynamic=True)

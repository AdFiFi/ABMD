from timeit import default_timer as timer

import wandb
from scipy.stats import zscore
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import roc_auc_score, accuracy_score

from .trainer import *


class NAVFTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super().__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_pretrain_inputs_kwargs(self, inputs):
        tokens = inputs["tokens"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        idx = list(range(1, tokens.shape[0])) + [0]
        if self.args.text_encoder in ["MTAMTextEncoder"]:
            with torch.no_grad():
                outputs = self.text_encoder(tokens, attention_mask)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

            embeddings = zscore(embeddings)
            positive_text_embedding = torch.tensor(embeddings, dtype=torch.float32)
            negative_text_embedding = deepcopy(positive_text_embedding)[idx]
            if self.args.brain_encoder in ["MTAMEEGEncoder"]:
                return {"anchor_brain_signals": inputs["mean_time_series"].to(self.device),
                        "positive_text": positive_text_embedding.to(self.device),
                        "negative_text": negative_text_embedding.to(self.device)
                        }
            elif self.args.brain_encoder in ["BNT", "BrainNetGNN", "Graphormer"]:
                return {"anchor_brain_signals": inputs["correlation"].to(self.device),
                        "positive_text": positive_text_embedding.to(self.device),
                        "negative_text": negative_text_embedding.to(self.device)
                        }
            else:
                return {"anchor_brain_signals": inputs["time_series"].to(self.device),
                        "positive_text": positive_text_embedding.to(self.device),
                        "negative_text": negative_text_embedding.to(self.device)
                        }
        else:
            if self.args.brain_encoder in ["MTAMEEGEncoder"]:
                return {"anchor_brain_signals": inputs["mean_time_series"].to(self.device),
                        "positive_text": tokens.to(self.device),
                        "negative_text": deepcopy(tokens)[idx].to(self.device),
                        "positive_attention_mask": attention_mask.to(self.device),
                        "negative_attention_mask": deepcopy(attention_mask)[idx].to(self.device)
                        }
            elif self.args.brain_encoder in ["BNT", "BrainNetCNN", "Graphormer"]:
                return {"anchor_brain_signals": inputs["correlation"].to(self.device),
                        "positive_text": tokens.to(self.device),
                        "negative_text": deepcopy(tokens)[idx].to(self.device),
                        "positive_attention_mask": attention_mask.to(self.device),
                        "negative_attention_mask": deepcopy(attention_mask)[idx].to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }
            else:
                return {"anchor_brain_signals": inputs["time_series"].to(self.device),
                        "positive_text": tokens.to(self.device),
                        "negative_text": deepcopy(tokens)[idx].to(self.device),
                        "positive_attention_mask": attention_mask.to(self.device),
                        "negative_attention_mask": deepcopy(attention_mask)[idx].to(self.device),
                        "labels": inputs["labels"].to(self.device)
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
            navf_config = NAVFConfig(brain_config=brain_config,
                                     vision_config=vision_config,
                                     modality=self.args.modality,
                                     num_classes=self.data_config.num_classes,
                                     class_weight=self.data_config.class_weight,
                                     label_smoothing=self.args.label_smoothing,
                                     )
            model = PretrainedNAVF(config=navf_config,
                                   brain_encoder=brain_encoder,
                                   vision_encoder=vision_encoder)
            return model, navf_config
        elif self.args.modality == 'text-brain':
            brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
            text_encoder, text_config = init_text_encoder_config(self.args, self.data_config)
            navf_config = NAVFConfig(brain_config=brain_config,
                                     text_config=text_config,
                                     d_model=self.args.d_model,
                                     d_hidden=self.args.d_hidden,
                                     modality=self.args.modality,
                                     num_classes=self.data_config.num_classes,
                                     class_weight=self.data_config.class_weight,
                                     label_smoothing=self.args.label_smoothing,)
            model = PretrainedNAVF(config=navf_config,
                                   brain_encoder=brain_encoder,
                                   text_encoder=text_encoder)
            return model, navf_config
        else:
            raise "Modality not supported"

    def reinit_model_config(self, finetune_modality="text"):
        self.load_model(pretrain=True)
        if self.args.modality == 'vision-brain':
            navf_config = NAVFConfig(brain_config=self.model_config.brain_config,
                                     vision_config=self.model_config.vision_config,
                                     modality=finetune_modality,
                                     num_classes=self.data_config.num_classes,
                                     class_weight=self.data_config.class_weight,
                                     label_smoothing=self.args.label_smoothing)
            model = ModifiedNAVF(config=navf_config,
                                 brain_encoder=self.model.brain_encoder,
                                 brain_projector=self.model.brain_projector,
                                 vision_encoder=self.model.vision_encoder,
                                 vision_projector=self.model.vision_projector,
                                 )
            self.model = model
            self.model_config = navf_config
        elif self.args.modality == 'text-brain':
            navf_config = NAVFConfig(brain_config=self.model_config.brain_config,
                                     text_config=self.model_config.text_config,
                                     d_model=self.args.d_model,
                                     d_hidden=self.args.d_hidden,
                                     num_classes=self.data_config.num_classes,
                                     modality=finetune_modality,
                                     class_weight=self.data_config.class_weight,
                                     label_smoothing=self.args.label_smoothing)
            model = ModifiedNAVF(config=navf_config,
                                 brain_encoder=self.model.brain_encoder,
                                 brain_projector=self.model.brain_projector,
                                 text_encoder=self.model.text_encoder,
                                 text_projector=self.model.text_projector)
            self.model = model
            self.model_config = navf_config
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

    def pre_train_epoch(self, step_one=False):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_pretrain_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs, step_one=step_one)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            wandb.log({'PreTraining loss': loss.item(),
                       'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            wandb.log({f'{self.model_config.modality} Training loss': loss.item(),
                       f'{self.model_config.modality} Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

    def pre_train(self):
        total = self.args.num_pretrain_epochs * len(self.data_loaders['train'])
        logger.info("***** Running PreTraining *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_pretrain_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components(pretrain=True)
        for epoch in tqdm(range(1, 6), desc="epoch", ncols=0):
            start_time = timer()
            train_loss = self.pre_train_epoch(step_one=True)
            end_time = timer()

            msg = f" PreTrain loss: {train_loss:.5f}" \
                  f" Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)

        for epoch in tqdm(range(1, self.args.num_pretrain_epochs + 1), desc="epoch", ncols=0):
            start_time = timer()
            train_loss = self.pre_train_epoch()
            end_time = timer()

            msg = f" PreTrain loss: {train_loss:.5f}" \
                  f" Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)
            self.save_model(pretrain=True)  # todo checkpoint?

    def fine_tune(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running FineTuning *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.best_tva_result = None
        self.init_components()
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch", ncols=0):
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()
            if self.model_config.modality != "brain":
                self.tva_result, y_true, y_pred, y_probas = self.evaluate()
                msg = f" Train loss: {train_loss:.5f}," \
                      f" Test loss: {self.tva_result[f'{self.model_config.modality} Testing Loss']:.5f}" \
                      f" Epoch time = {(end_time - start_time):.3f}s"
                print(msg)
                logger.info(msg)
                if self.best_tva_result is None or self.best_tva_result[f'{self.model_config.modality} Accuracy'] <= \
                        self.tva_result[f'{self.model_config.modality} Accuracy']:
                    self.best_tva_result = self.tva_result
                    self.best_tva_true = y_true
                    self.best_tva_pred = y_pred
                    self.best_tva_probas = y_probas
                    self.save_model(pretrain=False)
            else:
                self.brain_result, y_true, y_pred, y_probas = self.evaluate()
                msg = f" Train loss: {train_loss:.5f}," \
                      f" Test loss: {self.brain_result['brain Testing Loss']:.5f}" \
                      f" Epoch time = {(end_time - start_time):.3f}s"
                print(msg)
                logger.info(msg)
                if self.best_brain_result is None or self.best_brain_result['brain Accuracy'] <= \
                        self.brain_result['brain Accuracy']:
                    self.best_brain_result = self.brain_result
                    self.best_brain_true = y_true
                    self.best_brain_pred = y_pred
                    self.best_brain_probas = y_probas
                    self.save_model(pretrain=False)
        if self.model_config.modality != "brain":
            wandb.log({f"Best {k}": v for k, v in self.best_tva_result.items()})
            self.plot_confusion_matrix(modalities="tva")
            self.plot_precision_recall(modalities="tva")
            if self.data_config.num_classes == 2:
                self.plot_roc(modalities="tva")
        else:
            wandb.log({f"Best {k}": v for k, v in self.best_brain_result.items()})
            self.plot_confusion_matrix(modalities="brain")
            self.plot_precision_recall(modalities="brain")
            if self.data_config.num_classes == 2:
                self.plot_roc(modalities="brain")

    def train(self):
        self.pre_train()
        finetune_modalities = self.args.modality.split("-")
        self.reinit_model_config(finetune_modalities[0])
        self.fine_tune()
        self.reinit_model_config(finetune_modalities[1])
        self.fine_tune()

    def binary_evaluate(self):
        logger.info(f"***** Running evaluation on test{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders['test']
        losses = 0
        loss_list = []
        labels = []
        result = {}
        preds = None
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self.model(**input_kwargs)
                loss = outputs.loss
                losses += loss.item()
                loss_list.append(loss.item())

                if preds is None:
                    preds = F.softmax(outputs.logits, dim=1).cpu().numpy()
                else:
                    preds = np.append(preds, F.softmax(outputs.logits, dim=1).cpu().numpy(), axis=0)
                labels += input_kwargs['labels'].argmax(dim=-1).tolist()
            y_probas = deepcopy(preds)
            result['AUC'] = roc_auc_score(labels, preds[:, 1])
            preds = preds.argmax(axis=1).tolist()
            result['Accuracy'] = accuracy_score(labels, preds)
            preds, labels = np.array(preds), np.array(labels)
            metric = precision_recall_fscore_support(
                labels, preds, average='macro')
            result['Precision'] = metric[0]
            result['Recall'] = metric[1]
            result['F_score'] = metric[2]

            report = classification_report(
                labels, preds, output_dict=True, zero_division=0)

            result['Specificity'] = report['0']['recall']
            result['Sensitivity'] = report['1']['recall']
            result['Testing Loss'] = losses / len(loss_list)
        if self.args.within_subject:
            print(f'Test{self.subject_id}-{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}, '
                  f'Specificity:{result["Specificity"]:.5f}, Sensitivity:{result["Sensitivity"]:.5f}', end=',')
        else:
            print(f'Test{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}, '
                  f'Specificity:{result["Specificity"]:.5f}, Sensitivity:{result["Sensitivity"]:.5f}', end=',')
        for k, v in result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")

        if "-" in self.args.modality and "-" not in self.model_config.modality:
            result = {f"{self.model_config.modality} {k}": v for k, v in result.items()}
        wandb.log(result)
        return result, labels, preds, y_probas

    def multiple_evaluate(self):
        logger.info(f"***** Running evaluation on test{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders['test']
        losses = 0
        loss_list = []
        labels = []
        result = {}
        preds = None
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self.model(**input_kwargs)
                loss = outputs.loss
                losses += loss.item()
                loss_list.append(loss.item())
                # print(f"Evaluate loss: {loss.item():.5f}")
                if preds is None:
                    preds = F.softmax(outputs.logits, dim=1).cpu().numpy()
                else:
                    preds = np.append(preds, F.softmax(outputs.logits, dim=1).cpu().numpy(), axis=0)
                labels += input_kwargs['labels'].argmax(dim=-1).tolist()
            y_probas = deepcopy(preds)
            result['AUC'] = roc_auc_score(labels, preds, multi_class='ovo')
            preds = preds.argmax(axis=1).tolist()
            result['Accuracy'] = accuracy_score(labels, preds)
            preds, labels = np.array(preds), np.array(labels)
            metric = precision_recall_fscore_support(
                labels, preds, average='macro')
            result['Precision'] = metric[0]
            result['Recall'] = metric[1]
            result['F_score'] = metric[2]
            result['Testing Loss'] = losses / len(loss_list)
        if self.args.within_subject:
            print(f'Test{self.subject_id}-{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}',
                  end=',')
        else:
            print(f'Test{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}', end=',')
        for k, v in result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        if "-" in self.args.modality and "-" not in self.model_config.modality:
            result = {f"{self.model_config.modality} {k}": v for k, v in result.items()}
        wandb.log(result)
        return result, labels, preds, y_probas

    def get_model_name_and_path(self, pretrain=False):
        model = f"{self.args.framework}_{self.args.modality}_{self.args.brain_encoder}" \
                f"-{self.args.text_encoder}" if self.args.text_encoder else "" \
                                                                            f"-{self.args.vision_encoder}" if self.args.vision_encoder else "" \
                                                                                                                                            f"-{self.args.audio_encoder}" if self.args.audio_encoder else ""
        path = os.path.join(self.args.model_dir,
                            self.args.modality,
                            model)
        if pretrain:
            path = os.path.join(path, "pretrain")
        else:
            path = os.path.join(path, f"finetune-{self.model_config.modality}")

        model_path = os.path.join(path, f'{model}-{self.subject_id}-{self.task_id}.bin')
        config_path = os.path.join(path, 'config.json')

        return model, path, model_path, config_path

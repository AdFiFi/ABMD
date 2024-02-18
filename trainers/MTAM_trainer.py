from timeit import default_timer as timer

import wandb
from scipy.stats import zscore
from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score, accuracy_score

from .trainer import *


class MTAMTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super().__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        tokens = inputs["tokens"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        if self.args.text_encoder in ["MTAMTextEncoder"]:
            with torch.no_grad():
                outputs = self.text_encoder(tokens, attention_mask)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

            embeddings = zscore(embeddings)
            text_embedding = torch.tensor(embeddings, dtype=torch.float32)
            if self.args.brain_encoder in ["MTAMEEGEncoder"]:
                return {"text": text_embedding.to(self.device),
                        "brain_signals": inputs["mean_time_series"].to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }
            else:
                return {"text": text_embedding.to(self.device),
                        "brain_signals": inputs["time_series"].to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }
        else:
            if self.args.brain_encoder in ["MTAMEEGEncoder"]:
                return {"text": tokens.to(self.device),
                        "attention_mask": attention_mask,
                        "brain_signals": inputs["mean_time_series"].to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }
            else:
                return {"text": tokens.to(self.device),
                        "attention_mask": attention_mask,
                        "brain_signals": inputs["time_series"].to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }

    def init_model_config(self):
        brain_encoder = brain_config = text_encoder = text_config = None
        if "-" in self.args.modality:
            if 'text' in self.args.modality:
                brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
                text_encoder, text_config = init_text_encoder_config(self.args, self.data_config)
            else:
                raise "Modality not supported"
        else:
            if self.args.modality == 'text':
                text_encoder, text_config = init_text_encoder_config(self.args, self.data_config)
            elif self.args.modality == 'brain':
                brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
            else:
                raise "Modality not supported"
        config = MTAMConfig(brain_config=brain_config, text_config=text_config,
                            modality=self.args.modality,
                            class_weight=self.data_config.class_weight,
                            label_smoothing=self.args.label_smoothing,
                            num_classes=self.data_config.num_classes)
        model = ModifiedMTAM(config=config,
                             brain_encoder=brain_encoder,
                             text_encoder=text_encoder)
        return model, config

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_cca = 0
        loss_wd = 0
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

            loss_cca += outputs.cca_loss.item() if outputs.cca_loss is not None else 0
            loss_wd += outputs.wd_loss.item() if outputs.wd_loss is not None else 0
            wandb.log({'Training loss': loss.item(),
                       'Training CCA Loss': outputs.cca_loss.item() if outputs.cca_loss is not None else 0,
                       'Training WD Loss': outputs.wd_loss.item() if outputs.wd_loss is not None else 0,
                       'Learning rate': self.optimizer.param_groups[0]['lr']})
        return losses / len(loss_list), loss_cca / len(loss_list), loss_wd / len(loss_list)

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch", ncols=0):
            start_time = timer()
            train_loss, cca_loss, wd_loss = self.train_epoch()
            end_time = timer()

            self.joint_result, y_true, y_pred, y_probas = self.evaluate()
            msg = f" Train loss: {train_loss:.5f}, CCA: {cca_loss:.5f}, WD: {wd_loss:.5f}, " \
                  f"Test loss: {self.joint_result['Testing Loss']:.5f}, CCA: {self.joint_result['Testing CCA Loss']:.5f}," \
                  f" WD: {self.joint_result['Testing WD Loss']:.5f}, " \
                  f"Epoch time = {(end_time - start_time):.3f}s"

            print(msg)
            logger.info(msg)
            if self.best_joint_result is None \
                    or self.best_joint_result['joint Accuracy'] <= self.joint_result['joint Accuracy']:
                self.best_joint_result = self.joint_result
                self.best_joint_true = y_true
                self.best_joint_pred = y_pred
                self.best_joint_probas = y_probas
                self.save_model()
        wandb.log({f"Best {k}": v for k, v in self.best_joint_result.items()})
        self.plot_confusion_matrix(modalities="joint")
        self.plot_precision_recall(modalities="joint")
        if self.data_config.num_classes == 2:
            self.plot_roc(modalities="joint")

    def binary_evaluate(self):
        logger.info(f"***** Running evaluation on test{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders['test']
        losses = 0
        loss_cca = 0
        loss_wd = 0
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
                loss_cca += outputs.cca_loss.item() if outputs.cca_loss is not None else 0
                loss_wd += outputs.wd_loss.item() if outputs.wd_loss is not None else 0
                loss_list.append(loss.item())
                # print(f"Evaluate loss: {loss.item():.5f}")

                if preds is None:
                    preds = F.softmax(outputs.logits, dim=1).cpu().numpy()
                else:
                    preds = np.append(preds, F.softmax(outputs.logits, dim=1).cpu().numpy(), axis=0)
                labels += input_kwargs['labels'].argmax(dim=-1).tolist()
            y_probas = deepcopy(preds)
            result['joint AUC'] = roc_auc_score(labels, preds[:, 1])
            preds = preds.argmax(axis=1).tolist()
            result['joint Accuracy'] = accuracy_score(labels, preds)
            preds, labels = np.array(preds), np.array(labels)
            metric = precision_recall_fscore_support(
                labels, preds, average='macro')
            result['joint Precision'] = metric[0]
            result['joint Recall'] = metric[1]
            result['joint F_score'] = metric[2]

            report = classification_report(
                labels, preds, output_dict=True, zero_division=0)

            result['joint Specificity'] = report['0']['recall']
            result['joint Sensitivity'] = report['1']['recall']
            result['Testing Loss'] = losses / len(loss_list)
            result['Testing CCA Loss'] = loss_cca / len(loss_list)
            result['Testing WD Loss'] = loss_wd / len(loss_list)
        if self.args.within_subject:
            print(f'Test{self.subject_id}-{self.task_id} : '
                  f'Accuracy:{result["joint Accuracy"]:.5f}, '
                  f'AUC:{result["joint AUC"]:.5f}, '
                  f'Specificity:{result["joint Specificity"]:.5f}, '
                  f'Sensitivity:{result["joint Sensitivity"]:.5f}', end=',')
        else:
            print(f'Test{self.task_id} : '
                  f'Accuracy:{result["joint Accuracy"]:.5f}, '
                  f'AUC:{result["joint AUC"]:.5f}, '
                  f'Specificity:{result["joint Specificity"]:.5f}, '
                  f'Sensitivity:{result["joint Sensitivity"]:.5f}', end=',')
        for k, v in result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        wandb.log(result)
        return result, labels, preds, y_probas

    def multiple_evaluate(self):
        logger.info(f"***** Running evaluation on test{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders['test']
        losses = 0
        loss_cca = 0
        loss_wd = 0
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
                loss_cca += outputs.cca_loss.item() if outputs.cca_loss is not None else 0
                loss_wd += outputs.wd_loss.item() if outputs.wd_loss is not None else 0
                loss_list.append(loss.item())
                # print(f"Evaluate loss: {loss.item():.5f}")
                if preds is None:
                    preds = F.softmax(outputs.logits, dim=1).cpu().numpy()
                else:
                    preds = np.append(preds, F.softmax(outputs.logits, dim=1).cpu().numpy(), axis=0)
                labels += input_kwargs['labels'].argmax(dim=-1).tolist()
            y_probas = deepcopy(preds)
            result['joint AUC'] = roc_auc_score(labels, preds, multi_class='ovo')
            preds = preds.argmax(axis=1).tolist()
            result['joint Accuracy'] = accuracy_score(labels, preds)
            preds, labels = np.array(preds), np.array(labels)
            metric = precision_recall_fscore_support(
                labels, preds, average='macro')
            result['joint Precision'] = metric[0]
            result['joint Recall'] = metric[1]
            result['joint F_score'] = metric[2]
            result['Testing Loss'] = losses / len(loss_list)
            result['Testing CCA Loss'] = loss_cca / len(loss_list)
            result['Testing WD Loss'] = loss_wd / len(loss_list)
        if self.args.within_subject:
            print(f'Test{self.subject_id}-{self.task_id} : '
                  f'Accuracy:{result["joint Accuracy"]:.5f}, '
                  f'AUC:{result["joint AUC"]:.5f}',
                  end=',')
        else:
            print(f'Test{self.task_id} : '
                  f'Accuracy:{result["joint Accuracy"]:.5f}, '
                  f'AUC:{result["joint AUC"]:.5f}', end=',')
        for k, v in result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        wandb.log(result)
        return result, labels, preds, y_probas

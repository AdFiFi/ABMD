from .BMD_trainer import *


class HMAVTrainer(BMDTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super().__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def init_model_config(self):
        if self.args.modality == 'vision-brain':
            brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
            vision_encoder, vision_config = init_vision_encoder_config(self.args, self.data_config)
            hmav_config = HMAVConfig(brain_config=brain_config,
                                   vision_config=vision_config,
                                   modality=self.args.modality,
                                   num_classes=self.data_config.num_classes,
                                   class_weight=self.data_config.class_weight,
                                   label_smoothing=self.args.label_smoothing,
                                   use_temporal=self.args.use_temporal)
            model = HMAV(config=hmav_config,
                         brain_encoder=brain_encoder,
                         vision_encoder=vision_encoder)
            return model, hmav_config
        elif self.args.modality == 'text-brain':
            brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
            text_encoder, text_config = init_text_encoder_config(self.args, self.data_config)
            hmav_config = HMAVConfig(brain_config=brain_config,
                                   text_config=text_config,
                                   d_model=self.args.d_model,
                                   d_hidden=self.args.d_hidden,
                                   num_classes=self.data_config.num_classes,
                                   modality=self.args.modality,
                                   class_weight=self.data_config.class_weight,
                                   label_smoothing=self.args.label_smoothing,
                                   use_temporal=self.args.use_temporal)
            model = HMAV(config=hmav_config,
                         brain_encoder=brain_encoder,
                         text_encoder=text_encoder)
            return model, hmav_config
        else:
            raise "Modality not supported"

    def train_epoch(self, step_one=False):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_regression = 0
        # stop_flag = False
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            # self.model.brain_encoder.set_gradient_logger()
            outputs = self.model(**input_kwargs, step_one=step_one)

            # mean, var = self.model.brain_encoder.get_gradients()
            # stop_flag = (var > mean) and step_one
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            loss_regression += outputs.loss_regression.item() if outputs.loss_regression is not None else 0
            wandb.log({'Loss Main/Training loss': loss.item(),
                       'Loss/Training Regression Loss': outputs.loss_regression.item() if outputs.loss_regression is not None else 0,
                       'Loss Main/Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list), loss_regression / len(loss_list)

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
            train_loss, loss_regression = self.train_epoch()
            end_time = timer()

            self.tva_result, self.brain_result, labels, tva_preds, tva_probas, brain_preds, brain_probas = self.evaluate()
            msg = f" Train loss: {train_loss:.5f}, " \
                  f"Regression Loss: {loss_regression:.5f}, " \
                  f"Test loss: {self.tva_result['Loss Main/Testing Loss']:.5f}, " \
                  f"Regression: {self.tva_result['Loss/Testing Regression Loss']:.5f}, " \
                  f" Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)
            if self.best_tva_result is None or \
                    self.best_tva_result[f'{self.model_config.modality[:-6]} Accuracy'] <= \
                    self.tva_result[f'{self.model_config.modality[:-6]} Accuracy']:
                self.best_tva_result = self.tva_result
                self.best_tva_true = labels
                self.best_tva_pred = tva_preds
                self.best_tva_probas = tva_probas
                self.save_model()
            if self.best_brain_result is None or \
                    self.best_brain_result['brain Accuracy'] <= \
                    self.brain_result['brain Accuracy']:
                self.best_brain_result = self.brain_result
                self.best_brain_true = labels
                self.best_brain_pred = brain_preds
                self.best_brain_probas = brain_probas
                self.save_model()
        wandb.log({f"Best/Best {k}": v for k, v in self.best_tva_result.items()})
        wandb.log({f"Best/Best {k}": v for k, v in self.best_brain_result.items()})
        self.plot_confusion_matrix(modalities="all")
        self.plot_precision_recall(modalities="all")
        if self.data_config.num_classes == 2:
            self.plot_roc(modalities="all")

    def two_step_train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        for epoch in tqdm(range(1, self.args.num_first_step_epochs + 1), desc="first step epoch", ncols=0):
            start_time = timer()
            train_loss, loss_regression = self.train_epoch(step_one=True)
            end_time = timer()

            self.tva_result, self.brain_result, labels, tva_preds, tva_probas, brain_preds, brain_probas = self.evaluate()
            msg = f" Train loss: {train_loss:.5f}, " \
                  f"Regression: {loss_regression:.5f}, " \
                  f"Test loss: {self.tva_result['Loss Main/Testing Loss']:.5f}, " \
                  f"Regression: {self.tva_result['Loss/Testing Regression Loss']:.5f}, " \
                  f" Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)
            if self.best_tva_result is None or \
                    self.best_tva_result[f'{self.model_config.modality[:-6]} Accuracy'] <= \
                    self.tva_result[f'{self.model_config.modality[:-6]} Accuracy']:
                self.best_tva_result = self.tva_result
                self.best_tva_true = labels
                self.best_tva_pred = tva_preds
                self.best_tva_probas = tva_probas
                self.save_model()
            if self.best_brain_result is None or \
                    self.best_brain_result['brain Accuracy'] <= \
                    self.brain_result['brain Accuracy']:
                self.best_brain_result = self.brain_result
                self.best_brain_true = labels
                self.best_brain_pred = brain_preds
                self.best_brain_probas = brain_probas
                self.save_model()

        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="second step epoch", ncols=0):
            start_time = timer()
            train_loss, loss_regression = self.train_epoch()
            end_time = timer()

            self.tva_result, self.brain_result, labels, tva_preds, tva_probas, brain_preds, brain_probas = self.evaluate()
            msg = f" Train loss: {train_loss:.5f}, " \
                  f"Regression: {loss_regression:.5f}, " \
                  f"Test loss: {self.tva_result['Loss Main/Testing Loss']:.5f}, " \
                  f"Regression: {self.tva_result['Loss/Testing Regression Loss']:.5f}, " \
                  f" Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)
            if self.best_tva_result is None or \
                    self.best_tva_result[f'{self.model_config.modality[:-6]} Accuracy'] <= \
                    self.tva_result[f'{self.model_config.modality[:-6]} Accuracy']:
                self.best_tva_result = self.tva_result
                self.best_tva_true = labels
                self.best_tva_pred = tva_preds
                self.best_tva_probas = tva_probas
                self.save_model()
            # if self.best_brain_result is None or \
            #         self.best_brain_result['brain Accuracy'] <= \
            #         self.brain_result['brain Accuracy']:
            #     self.best_brain_result = self.brain_result
            #     self.best_brain_true = labels
            #     self.best_brain_pred = brain_preds
            #     self.best_brain_probas = brain_probas
            #     self.save_model()
        wandb.log({f"Best/Best {k}": v for k, v in self.best_tva_result.items()})
        wandb.log({f"Best/Best {k}": v for k, v in self.best_brain_result.items()})
        self.plot_confusion_matrix(modalities="all")
        self.plot_precision_recall(modalities="all")
        if self.data_config.num_classes == 2:
            self.plot_roc(modalities="all")

    def binary_evaluate(self):
        logger.info(f"***** Running evaluation on test{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders['test']
        losses = 0
        loss_regression = 0
        loss_list = []
        labels = []
        tva_result = {}
        brain_result = {}
        tva_preds = None
        brain_preds = None
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self.model(**input_kwargs)
                loss = outputs.loss
                losses += loss.item()
                loss_regression += outputs.loss_regression.item()
                loss_list.append(loss.item())
                if "text" in self.model_config.modality:
                    tva_logits = outputs.text_logits
                elif "vision" in self.model_config.modality:
                    tva_logits = outputs.vision_logits
                else:
                    tva_logits = outputs.audio_logits
                if tva_preds is None:
                    tva_preds = F.softmax(tva_logits, dim=1).cpu().numpy()
                    brain_preds = F.softmax(outputs.brain_logits, dim=1).cpu().numpy()
                else:
                    tva_preds = np.append(tva_preds, F.softmax(tva_logits, dim=1).cpu().numpy(), axis=0)
                    brain_preds = np.append(brain_preds, F.softmax(outputs.brain_logits, dim=1).cpu().numpy(), axis=0)
                labels += input_kwargs['labels'].argmax(dim=-1).tolist()
            tva_probas = deepcopy(tva_preds)
            brain_probas = deepcopy(brain_preds)
            tva_result[f'{self.model_config.modality[:-6]} AUC'] = roc_auc_score(labels, tva_preds[:, 1])
            tva_preds = tva_preds.argmax(axis=1).tolist()
            tva_result[f'{self.model_config.modality[:-6]} Accuracy'] = accuracy_score(labels, tva_preds)
            tva_preds, labels = np.array(tva_preds), np.array(labels)
            tva_metric = precision_recall_fscore_support(
                labels, tva_preds, average='micro')
            tva_result[f'{self.model_config.modality[:-6]} Precision'] = tva_metric[0]
            tva_result[f'{self.model_config.modality[:-6]} Recall'] = tva_metric[1]
            tva_result[f'{self.model_config.modality[:-6]} F_score'] = tva_metric[2]
            tva_report = classification_report(
                labels, tva_preds, output_dict=True, zero_division=0)
            tva_result[f'{self.model_config.modality[:-6]} Specificity'] = tva_report['0']['recall']
            tva_result[f'{self.model_config.modality[:-6]} Sensitivity'] = tva_report['1']['recall']

            brain_result['brain AUC'] = roc_auc_score(labels, brain_preds[:, 1])
            brain_preds = brain_preds.argmax(axis=1).tolist()
            brain_result[f'brain Accuracy'] = accuracy_score(labels, brain_preds)
            brain_preds, labels = np.array(brain_preds), np.array(labels)
            brain_metric = precision_recall_fscore_support(
                labels, brain_preds, average='micro')
            brain_result['brain Precision'] = brain_metric[0]
            brain_result['brain Recall'] = brain_metric[1]
            brain_result['brain F_score'] = brain_metric[2]
            brain_report = classification_report(
                labels, brain_preds, output_dict=True, zero_division=0)
            brain_result['brain Specificity'] = brain_report['0']['recall']
            brain_result['brain Sensitivity'] = brain_report['1']['recall']

            tva_result['Loss Main/Testing Loss'] = losses / len(loss_list)
            tva_result['Loss/Testing Regression Loss'] = loss_regression / len(loss_list)

        if self.args.within_subject:
            print(f'{self.model_config.modality[:-6]} Test{self.subject_id}-{self.task_id} : '
                  f'Accuracy:{tva_result[f"{self.model_config.modality[:-6]} Accuracy"]:.5f}, '
                  f'AUC:{tva_result[f"{self.model_config.modality[:-6]} AUC"]:.5f}, '
                  f'Specificity:{tva_result[f"{self.model_config.modality[:-6]} Specificity"]:.5f}, '
                  f'Sensitivity:{tva_result[f"{self.model_config.modality[:-6]} Sensitivity"]:.5f}')
            print(f'brain Test{self.subject_id}-{self.task_id} : '
                  f'Accuracy:{brain_result["brain Accuracy"]:.5f}, '
                  f'AUC:{brain_result["brain AUC"]:.5f}, '
                  f'Specificity:{brain_result["brain Specificity"]:.5f}, '
                  f'Sensitivity:{brain_result["brain Sensitivity"]:.5f}', end=',')
        else:
            print(f'{self.model_config.modality[:-6]} Test{self.task_id} : '
                  f'Accuracy:{tva_result[f"{self.model_config.modality[:-6]} Accuracy"]:.5f}, '
                  f'AUC:{tva_result[f"{self.model_config.modality[:-6]} AUC"]:.5f}, '
                  f'Specificity:{tva_result[f"{self.model_config.modality[:-6]} Specificity"]:.5f}, '
                  f'Sensitivity:{tva_result[f"{self.model_config.modality[:-6]} Sensitivity"]:.5f}')
            print(f'brain Test{self.task_id} : '
                  f'Accuracy:{brain_result["brain Accuracy"]:.5f}, AUC:{brain_result["brain AUC"]:.5f}, '
                  f'Specificity:{brain_result["brain Specificity"]:.5f}, Sensitivity:{brain_result["brain Sensitivity"]:.5f}',
                  end=',')
        for k, v in tva_result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        for k, v in brain_result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        wandb.log(tva_result)
        wandb.log(brain_result)
        return tva_result, brain_result, labels, tva_preds, tva_probas, brain_preds, brain_probas

    def multiple_evaluate(self, do_print=False):
        logger.info(f"***** Running evaluation on test{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders['test']
        losses = 0
        loss_regression = 0
        loss_list = []
        labels = []
        tva_result = {}
        brain_result = {}
        tva_preds = None
        brain_preds = None
        brain_hidden_state = []
        tva_hidden_state = []
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self.model(**input_kwargs)
                loss = outputs.loss
                losses += loss.item()
                loss_regression += outputs.loss_regression.item()
                loss_list.append(loss.item())
                brain_hidden_state += outputs.brain_state.detach().cpu().tolist()
                if "text" in self.model_config.modality:
                    tva_logits = outputs.text_logits
                    tva_hidden_state += outputs.text_state.detach().cpu().tolist()
                elif "vision" in self.model_config.modality:
                    tva_logits = outputs.vision_logits
                    tva_hidden_state += outputs.vision_state.detach().cpu().tolist()
                else:
                    tva_logits = outputs.audio_logits
                    tva_hidden_state += outputs.audio_state.detach().cpu().tolist()
                if tva_preds is None:
                    tva_preds = F.softmax(tva_logits, dim=1).cpu().numpy()
                    brain_preds = F.softmax(outputs.brain_logits, dim=1).cpu().numpy()
                else:
                    tva_preds = np.append(tva_preds, F.softmax(tva_logits, dim=1).cpu().numpy(), axis=0)
                    brain_preds = np.append(brain_preds, F.softmax(outputs.brain_logits, dim=1).cpu().numpy(), axis=0)
                labels += input_kwargs['labels'].argmax(dim=-1).tolist()
                if do_print:
                    print('预测: ', input_kwargs['labels'].argmax(dim=-1).tolist())
                    print('真值: ', tva_logits.argmax(dim=-1).tolist())
                    print()
            tva_probas = deepcopy(tva_preds)
            brain_probas = deepcopy(brain_preds)
            tva_result[f'{self.model_config.modality[:-6]} AUC'] = roc_auc_score(labels, tva_preds, multi_class='ovo')
            tva_preds = tva_preds.argmax(axis=1).tolist()
            tva_result[f'{self.model_config.modality[:-6]} Accuracy'] = accuracy_score(labels, tva_preds)
            tva_preds, labels = np.array(tva_preds), np.array(labels)
            self.hidden_state = {"brain_state": np.array(brain_hidden_state),
                                 "tva_state": np.array(tva_hidden_state),
                                 "labels": labels}
            tva_metric = precision_recall_fscore_support(
                # labels, preds, average='micro')
                labels, tva_preds, average='macro', zero_division=.0)
            tva_result[f'{self.model_config.modality[:-6]} Precision'] = tva_metric[0]
            tva_result[f'{self.model_config.modality[:-6]} Recall'] = tva_metric[1]
            tva_result[f'{self.model_config.modality[:-6]} F_score'] = tva_metric[2]

            brain_result['brain AUC'] = roc_auc_score(labels, brain_preds, multi_class='ovo')
            brain_preds = brain_preds.argmax(axis=1).tolist()
            brain_result['brain Accuracy'] = accuracy_score(labels, brain_preds)
            brain_preds, labels = np.array(brain_preds), np.array(labels)
            brain_metric = precision_recall_fscore_support(
                # labels, preds, average='micro')
                labels, brain_preds, average='macro', zero_division=.0)
            brain_result['brain Precision'] = brain_metric[0]
            brain_result['brain Recall'] = brain_metric[1]
            brain_result['brain F_score'] = brain_metric[2]

            tva_result['Loss Main/Testing Loss'] = losses / len(loss_list)
            tva_result['Loss/Testing Regression Loss'] = loss_regression / len(loss_list)

        if self.args.within_subject:
            print(f'{self.model_config.modality[:-6]} Test{self.subject_id}-{self.task_id} : '
                  f'Accuracy:{tva_result[f"{self.model_config.modality[:-6]} Accuracy"]:.5f}, '
                  f'AUC:{tva_result[f"{self.model_config.modality[:-6]} AUC"]:.5f}')
            print(f'{self.model_config.modality[:-6]} Test{self.subject_id}-{self.task_id} : '
                  f'Accuracy:{tva_result[f"{self.model_config.modality[:-6]} Accuracy"]:.5f}, '
                  f'AUC:{tva_result[f"{self.model_config.modality[:-6]} AUC"]:.5f}',
                  end=',')
        else:
            print(f'brain Test{self.task_id} : '
                  f'Accuracy:{tva_result[f"{self.model_config.modality[:-6]} Accuracy"]:.5f}, '
                  f'AUC:{tva_result[f"{self.model_config.modality[:-6]} AUC"]:.5f}')
            print(f'brain Test{self.task_id} : '
                  f'Accuracy:{brain_result["brain Accuracy"]:.5f}, '
                  f'AUC:{brain_result["brain AUC"]:.5f}', end=',')
        for k, v in tva_result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        for k, v in brain_result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        wandb.log(tva_result)
        wandb.log(brain_result)
        return tva_result, brain_result, labels, tva_preds, tva_probas, brain_preds, brain_probas


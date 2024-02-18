from timeit import default_timer as timer

from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

import wandb

from .trainer import *


class HBDTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super().__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        return {k: v.to(self.device) for k, v in inputs.items()}

    def init_model_config(self):
        brain_encoder = brain_config = vision_encoder = vision_config = None
        if self.args.mode == 'coupled':
            brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
            vision_encoder, vision_config = init_vision_encoder_config(self.args, self.data_config)
            bmcl_config = BMCLConfig(brain_config=brain_config, vision_config=vision_config,
                                     mode=self.args.mode,
                                     class_weight=self.data_config.class_weight,
                                     label_smoothing=self.args.label_smoothing,
                                     alpha=self.args.alpha,
                                     beta=self.args.beta)
            model = BMCL(config=bmcl_config,
                         brain_encoder=brain_encoder,
                         vision_encoder=vision_encoder)
            return model, bmcl_config
        else:
            if self.args.mode == 'vision':
                vision_encoder, vision_config = init_vision_encoder_config(self.args, self.data_config)
            elif self.args.mode == 'brain':
                brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
            elif self.args.mode == 'distill':
                brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
                # brain_encoder, brain_config = self.load_brain()

                vision_encoder, vision_config = init_vision_encoder_config(self.args, self.data_config)
                # vision_encoder, vision_config = self.load_vision()

            hbd_config = HBDConfig(mode=self.args.mode,
                                   class_weight=self.data_config.class_weight,
                                   label_smoothing=self.args.label_smoothing,
                                   lam=self.args.lam,
                                   t=self.args.t,
                                   brain_config=brain_config,
                                   vision_config=vision_config)
            model = HBD(config=hbd_config,
                        brain_encoder=brain_encoder,
                        vision_encoder=vision_encoder)
            return model, hbd_config

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_kl = 0
        loss_sim = 0
        loss_diff = 0
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
            if self.args.mode == 'coupled':
                loss_sim += outputs.loss_sim.item() if outputs.loss_sim is not None else 0
                loss_diff += outputs.loss_diff.item() if outputs.loss_diff is not None else 0
                wandb.log({'Training loss': loss.item(),
                           'Sim loss': outputs.loss_sim.item() if outputs.loss_sim is not None else 0,
                           'Diff loss': outputs.loss_diff.item() if outputs.loss_diff is not None else 0,
                           'Learning rate': self.optimizer.param_groups[0]['lr']})
            else:
                loss_kl += outputs.loss_kl.item() if outputs.loss_kl is not None else 0
                wandb.log({'Training loss': loss.item(),
                           'KL loss': outputs.loss_kl.item() if outputs.loss_kl is not None else 0,
                           'Learning rate': self.optimizer.param_groups[0]['lr']})
        if self.args.mode == 'coupled':
            return losses / len(loss_list), (loss_sim / len(loss_list), loss_diff / len(loss_list))
        else:
            return losses / len(loss_list), loss_kl / len(loss_list)

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
            train_loss, train_loss2 = self.train_epoch()
            end_time = timer()

            self.test_result = self.evaluate()
            if self.args.mode == 'coupled':
                msg = f" Train loss: {train_loss:.5f}, Sim loss: {train_loss2[0]:.5f}, Diff loss: {train_loss2[1]:.5f}, " \
                      f"Test loss: {self.test_result['Loss']:.5f}, Epoch time = {(end_time - start_time):.3f}s"
            else:
                msg = f" Train loss: {train_loss:.5f}, KL loss: {train_loss2:.5f}, Test loss: {self.test_result['Loss']:.5f}," \
                      f"Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)
            if self.best_result is None or self.best_result['Accuracy'] <= self.test_result['Accuracy']:
                self.best_result = self.test_result
                self.save_model()
        wandb.log({f"Best {k}": v for k, v in self.best_result.items()})

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
                # print(f"Evaluate loss: {loss.item():.5f}")

                if preds is None:
                    preds = F.softmax(outputs.logits, dim=1).cpu().numpy()
                else:
                    preds = np.append(preds, F.softmax(outputs.logits, dim=1).cpu().numpy(), axis=0)
                labels += input_kwargs['labels'].argmax(dim=-1).tolist()
            result['AUC'] = roc_auc_score(labels, preds[:, 1])
            preds = preds.argmax(axis=1).tolist()
            result['Accuracy'] = accuracy_score(labels, preds)
            preds, labels = np.array(preds), np.array(labels)
            metric = precision_recall_fscore_support(
                labels, preds, average='micro')
            result['Precision'] = metric[0]
            result['Recall'] = metric[1]
            result['F_score'] = metric[2]

            report = classification_report(
                labels, preds, output_dict=True, zero_division=0)

            result['Specificity'] = report['0']['recall']
            result['Sensitivity'] = report['1']['recall']
            result['Loss'] = losses / len(loss_list)
        if self.args.within_subject:
            print(f'Test{self.subject_id}-{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}, '
                  f'Specificity:{result["Specificity"]:.5f}, Sensitivity:{result["Sensitivity"]:.5f}', end=',')
        else:
            print(f'Test{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}, '
                  f'Specificity:{result["Specificity"]:.5f}, Sensitivity:{result["Sensitivity"]:.5f}', end=',')
        for k, v in result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        wandb.log(result)
        return result

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
            result['AUC'] = roc_auc_score(labels, preds, multi_class='ovo')
            preds = preds.argmax(axis=1).tolist()
            result['Accuracy'] = accuracy_score(labels, preds)
            preds, labels = np.array(preds), np.array(labels)
            metric = precision_recall_fscore_support(
                labels, preds, average='micro')
            result['Precision'] = metric[0]
            result['Recall'] = metric[1]
            result['F_score'] = metric[2]
            result['Loss'] = losses / len(loss_list)
        if self.args.within_subject:
            print(f'Test{self.subject_id}-{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}',
                  end=',')
        else:
            print(f'Test{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}', end=',')
        for k, v in result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        wandb.log(result)
        return result

    def load_brain(self):
        path = os.path.join(self.args.model_dir, 'brain', self.args.brain_encoder)
        brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
        brain_encoder = torch.load(
            os.path.join(path, f'{self.args.brain_encoder}-{self.subject_id}-{self.task_id}.bin')).brain_encoder
        brain_config.load(os.path.join(path, f"{self.args.brain_encoder}-config.json"))
        return brain_encoder, brain_config

    def load_vision(self):
        path = os.path.join(self.args.model_dir, 'vision', self.args.vision_encoder)
        vision_encoder, vision_config = init_vision_encoder_config(self.args, self.data_config)
        vision_encoder = torch.load(os.path.join(path, f'{self.args.vision_encoder}-0-{self.task_id}.bin')).vision_encoder
        vision_config.load(os.path.join(path, f"{self.args.vision_encoder}-config.json"))
        return vision_encoder, vision_config

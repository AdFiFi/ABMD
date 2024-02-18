import json
import os
import wandb
import scipy
from transformers import BertModel
from scipy.stats import zscore
from timeit import default_timer as timer
import wandb

from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import roc_auc_score, accuracy_score

from config import *
from data import *
from models import *
from utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        self.task_id = task_id
        self.args = args
        self.local_rank = local_rank
        self.subject_id = subject_id
        self.data_config = DataConfig(args)
        self.data_loaders = self.load_datasets()

        model, self.model_config = self.init_model_config()
        if args.do_parallel:
            self.device = f'cuda:{self.local_rank}' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            self.model = model.to(args.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   find_unused_parameters=True)
        else:
            self.device = f'cuda' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            self.model = model.to(args.device)
        # self.model = torch.compile(model, dynamic=True)

        self.optimizer = None
        self.scheduler = None

        self.best_tva_result = None
        self.best_brain_result = None
        self.tva_result = None
        self.brain_result = None
        self.best_tva_true = None
        self.best_brain_true = None
        self.best_tva_pred = None
        self.best_brain_pred = None
        self.best_tva_probas = None
        self.best_brain_probas = None

        self.joint_result = None
        self.best_joint_result = None
        self.best_joint_true = None
        self.best_joint_pred = None
        self.best_joint_probas = None

        self.hidden_state = {}

        if self.args.text_encoder == "MTAMTextEncoder":
            self.text_encoder = BertModel.from_pretrained('/data/models/bert-base-uncase').to(self.device)
        else:
            self.text_encoder = None

    def prepare_inputs_kwargs(self, inputs):
        if self.args.modality == "text":
            if self.args.text_encoder == "MTAMTextEncoder":
                tokens = inputs["tokens"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                with torch.no_grad():
                    outputs = self.text_encoder(tokens, attention_mask)
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

                embeddings = zscore(embeddings)
                text_embedding = torch.tensor(embeddings, dtype=torch.float32)
                return {"text": text_embedding.to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }
            else:  # todo
                tokens = inputs["tokens"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                return {"text": tokens,
                        "attention_mask": attention_mask,
                        "labels": inputs["labels"].to(self.device)
                        }
        elif self.args.modality == "vision":
            pass
        elif self.args.modality == "audio":
            pass
        elif self.args.modality == "brain":
            if self.args.brain_encoder == "MTAMEEGEncoder":
                return {"brain_signals": inputs["mean_time_series"].to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }
            elif self.args.brain_encoder in ["BNT", "BrainNetGNN", "Graphormer"]:
                return {"brain_signals": inputs["correlation"].to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }
            else:
                return {"brain_signals": inputs["time_series"].to(self.device),
                        "brain_signals_mask": inputs["time_series_mask"].to(self.device),
                        "labels": inputs["labels"].to(self.device)
                        }
        else:
            raise "MultiModality not supported!"

    def load_datasets(self):
        datasets = eval(
            f"{self.args.dataset}Dataset")(self.args, self.data_config, k=self.task_id, subject_id=self.subject_id)

        if self.args.do_parallel:
            data_loaders = init_distributed_dataloader(self.data_config, datasets)
        else:
            data_loaders = init_StratifiedKFold_dataloader(self.data_config, datasets)
        return data_loaders

    def init_components(self, pretrain=False):
        if pretrain:
            # total = self.args.num_pretrain_epochs * len(self.data_loaders['train'])
            total = 50 * len(self.data_loaders['train'])
            self.optimizer = init_optimizer(self.model, self.args)
            self.scheduler = init_schedule(self.optimizer, self.args, total)
        else:
            # total = self.args.num_epochs * len(self.data_loaders['train'])
            total = 15 * len(self.data_loaders['train'])
            # total = len(self.data_loaders['train'])
            self.optimizer = init_optimizer(self.model, self.args)
            self.scheduler = init_schedule(self.optimizer, self.args, total)

    def init_model_config(self):
        if self.args.modality == 'text':
            text_encoder, text_config = init_text_encoder_config(self.args, self.data_config)
            model_config = ModelConfig(text_config=text_config,
                                       modality=self.args.modality)
            model = BaseModel(model_config,
                              text_encoder=text_encoder,)
            return model, model_config
        elif self.args.modality == 'vision':
            vision_encoder, vision_config = init_vision_encoder_config(self.args, self.data_config)
            model_config = ModelConfig(vision_config=vision_config,
                                       modality=self.args.modality)
            model = BaseModel(model_config,
                              vision_encoder=vision_encoder)
            return model, model_config
        elif self.args.modality == 'audio':
            audio_encoder, audio_config = init_audio_encoder_config(self.args, self.data_config)
            model_config = ModelConfig(audio_config=audio_config,
                                       modality=self.args.modality)
            model = BaseModel(model_config,
                              audio_encoder=audio_encoder)
            return model, model_config
        elif self.args.modality == 'brain':
            brain_encoder, brain_config = init_brain_encoder_config(self.args, self.data_config)
            model_config = ModelConfig(brain_config=brain_config,
                                       modality=self.args.modality)
            model = BaseModel(model_config,
                              brain_encoder=brain_encoder)
            return model, model_config
        else:
            return None, None

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            input_kwargs = self.prepare_inputs_kwargs(inputs)

            # self.model.set_gradient_logger()
            outputs = self.model(**input_kwargs)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            wandb.log({'Training loss': loss.item(),
                       'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running FineTuning *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

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
                    self.save_hidden_state()
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
                    self.save_hidden_state()
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

    def evaluate(self):
        if self.data_config.num_classes == 2:
            result = self.binary_evaluate()
        else:
            result = self.multiple_evaluate()
        return result

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
                labels, preds, average='macro', zero_division=.0)
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
        brain_hidden_state = []
        tva_hidden_state = []
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self.model(**input_kwargs)
                loss = outputs.loss
                losses += loss.item()
                loss_list.append(loss.item())
                if self.model_config.modality == "brain":
                    brain_hidden_state += outputs.brain_state.detach().cpu().tolist()
                else:
                    tva_hidden_state += outputs.text_state.detach().cpu().tolist()
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
            if self.model_config.modality == "brain":
                self.hidden_state = {"brain_state": np.array(brain_hidden_state),
                                     "labels": labels}
            else:
                self.hidden_state = {"tva_state": np.array(tva_hidden_state),
                                     "labels": labels}
            metric = precision_recall_fscore_support(
                labels, preds, average='macro', zero_division=.0)
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
        result = {f"{self.model_config.modality} {k}": v for k, v in result.items()}
        wandb.log(result)
        return result, labels, preds, y_probas

    def save_model(self, pretrain=False):
        model, path, model_path, config_path = self.get_model_name_and_path(pretrain=pretrain)

        model_to_save = self.model.module if self.args.do_parallel else self.model
        torch.save(model_to_save, model_path)

        # Save training arguments together with the trained model
        args_dict = self.model_config.dict()
        with open(config_path, 'w') as f:
            f.write(json.dumps(args_dict))
        logger.info("Saving model checkpoint to %s", path)

    def get_model_name_and_path(self, pretrain=False):
        model = f"{self.args.framework if self.args.framework else 'SM'}_{self.args.modality}" \
                f"{f'_{self.args.brain_encoder}_{self.args.brain_fusion}' if self.args.brain_encoder else ''}" \
                f"{f'-{self.args.text_encoder}_{self.args.text_fusion}' if self.args.text_encoder else ''}" \
                f"{f'-{self.args.vision_encoder}_{self.args.vision_fusion}' if self.args.vision_encoder else ''}" \
                f"{f'-{self.args.audio_encoder}_{self.args.audio_fusion}' if self.args.audio_encoder else ''}"
        path = os.path.join(self.args.model_dir,
                            self.args.modality,
                            model)
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = os.path.join(path, f'{model}-{self.subject_id}-{self.task_id}.bin')
        config_path = os.path.join(path, 'config.json')

        return model, path, model_path, config_path

    def load_model(self, pretrain=False):
        model, path, model_path, config_path = self.get_model_name_and_path(pretrain=pretrain)
        if not os.path.exists(model_path):
            raise "Model doesn't exists! Train first!"
        if self.args.do_parallel:
            self.model = torch.load(os.path.join(model_path))
            self.model = self.model.to(self.args.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank])
        else:
            self.model = torch.load(os.path.join(model_path))
            self.model.to(self.device)
        logger.info("***** Model Loaded *****")

    def plot_confusion_matrix(self, modalities="all"):
        def plot(y_true, y_pred, label_names, modality):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            not_missing = wandb.sklearn.utils.test_missing(y_true=y_true, y_pred=y_pred)
            correct_types = wandb.sklearn.utils.test_types(y_true=y_true, y_pred=y_pred)

            if not_missing and correct_types:
                confusion_matrix_chart = wandb.sklearn.calculate.confusion_matrix(
                    y_true,
                    y_pred,
                    label_names
                )

                wandb.log({f"Best Confusion_Matrix {modality}": confusion_matrix_chart})
        if self.data_config.num_classes == 2:
            return
        if modalities == "tva":
            plot(self.best_tva_true, self.best_tva_pred, self.data_config.label_names, self.model_config.modality)
        elif modalities == "brain":
            plot(self.best_brain_true, self.best_brain_pred, self.data_config.label_names, "brain")
        elif modalities == "joint":
            plot(self.best_joint_true, self.best_joint_pred, self.data_config.label_names, "joint")
        else:
            plot(self.best_tva_true, self.best_tva_pred, self.data_config.label_names, self.model_config.modality)
            plot(self.best_brain_true, self.best_brain_pred, self.data_config.label_names, "brain")

    def plot_precision_recall(self, plot_micro=True, classes_to_plot=None, modalities="all"):
        def plot(y_true, y_probas, label_names, modality):
            precision_recall_chart = wandb.plots.precision_recall(
                y_true, y_probas, label_names, plot_micro, classes_to_plot
            )

            wandb.log({f"Best Precision_Recall {modality}": precision_recall_chart})

        if self.data_config.num_classes == 2:
            return
        if modalities == "tva":
            plot(self.best_tva_true, self.best_tva_probas, self.data_config.label_names, self.model_config.modality)
        elif modalities == "brain":
            plot(self.best_brain_true, self.best_brain_probas, self.data_config.label_names, "brain")
        elif modalities == "joint":
            plot(self.best_joint_true, self.best_joint_probas, self.data_config.label_names, "joint")
        else:
            plot(self.best_tva_true, self.best_tva_probas, self.data_config.label_names, self.model_config.modality)
            plot(self.best_brain_true, self.best_brain_probas, self.data_config.label_names, "brain")

    def plot_roc(self, plot_micro=True, plot_macro=True, classes_to_plot=None, modalities="all"):
        def plot(y_true, y_probas, label_names, modality):
            roc_chart = wandb.plots.roc.roc(
                y_true, y_probas, label_names, plot_micro, plot_macro, classes_to_plot
            )
            wandb.log({f"Best ROC {modality}": roc_chart})
        if self.data_config.num_classes == 2:
            return
        if modalities == "tva":
            plot(self.best_tva_true, self.best_tva_probas, self.data_config.label_names, self.model_config.modality)
        elif modalities == "brain":
            plot(self.best_brain_true, self.best_brain_probas, self.data_config.label_names, "brain")
        elif modalities == "joint":
            plot(self.best_joint_true, self.best_joint_probas, self.data_config.label_names, "joint")
        else:
            plot(self.best_tva_true, self.best_tva_probas, self.data_config.label_names, self.model_config.modality)
            plot(self.best_brain_true, self.best_brain_probas, self.data_config.label_names, "brain")

    def get_best(self):
        best_tva_result = self.best_tva_result if self.best_tva_result is not None else {}
        best_brain_result = self.best_brain_result if self.best_brain_result is not None else {}
        best_joint_result = self.best_joint_result if self.best_joint_result is not None else {}
        best_tva_result.update(best_brain_result)
        best_tva_result.update(best_joint_result)
        return best_tva_result

    def save_hidden_state(self):
        model, path, model_path, config_path = self.get_model_name_and_path()
        path = os.path.join(path, f"hidden_state-{self.task_id}.mat")
        scipy.io.savemat(path, self.hidden_state)

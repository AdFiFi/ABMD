import os
import math
import mne
import numpy as np
import pandas as pd
import scipy.io as io
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import BertTokenizer, RobertaTokenizer, BartTokenizer

from .data_config import DataConfig
from .dataset import BaseDataset

names = ["ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH"]
MIN_TEXT_LEN = 5
MAX_TEXT_LEN = 32
MIN_EEG_LEN = 200
MAX_EEG_LEN = 2000


class ZuCoDataset(BaseDataset):
    def __init__(self, args, data_config: DataConfig, k=0, train=True, subject_id=0):
        self.tokenizer = BertTokenizer.from_pretrained('/data/models/bert-base-uncase')
        # self.tokenizer = RobertaTokenizer.from_pretrained('/data/models/roberta-base')
        # self.tokenizer = BartTokenizer.from_pretrained('/data/models/bart-base')
        super(ZuCoDataset, self).__init__(args, data_config, k, train, subject_id=subject_id)

    def load_data(self):
        self.all_data = np.load(self.data_config.data_dir, allow_pickle=True).item()

        self.data_config.node_size = self.data_config.node_feature_size = 104
        self.truncating()
        self.tokenize()

        if "TSR" in self.data_config.data_dir:
            label_map = {k: v for v, k in enumerate(sorted(list(set(self.all_data['labels']))))}
            self.all_data['labels'] = np.array([label_map[l] for l in self.all_data['labels']])
            self.data_config.num_classes = 9
            self.data_config.label_names = list(label_map.keys())
            groups = np.array([label_map[l] for l in self.all_data['labels_list']])
        elif "SR" in self.data_config.data_dir:
            self.data_config.num_classes = 3
            self.all_data['labels'] = self.all_data['labels'] + 1
            self.data_config.label_names = [-1, 0, 1]
            groups = self.all_data["labels_list"] + 1
        else:
            self.all_data['labels'] = (self.all_data['labels'] != "NO-RELATION") * 1
            self.data_config.num_classes = 2
            self.data_config.label_names = ["NO-RELATION", "RELATION"]
            groups = (self.all_data["labels_list"] == "RELATION") * 1

        self.data_config.class_weight = [1] * self.data_config.num_classes
        if self.subject_id:
            self.select_subject()
        sentence_ids = self.all_data["id_list"]

        train_ids, test_ids = list(self.k_fold.split(sentence_ids, groups))[self.k]
        train_ids = sentence_ids[train_ids]
        test_ids = sentence_ids[test_ids]
        self.train_index = [i for i, j in enumerate(self.all_data['sentence_ids']) if j in train_ids]
        self.test_index = [i for i, j in enumerate(self.all_data['sentence_ids']) if j in test_ids]
        self.all_data['labels'] = F.one_hot(torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()

        # label_count = np.array([(self.all_data['labels'][self.train_index] == i).sum()
        #                         for i in range(self.data_config.num_classes)])
        # label_count = label_count.sum() - label_count
        # self.data_config.class_weight = (label_count / label_count.sum()).tolist()

    def __len__(self):
        return len(self.train_index) if self.train else len(self.test_index)

    def __getitem__(self, item):
        idx = self.train_index if self.train else self.test_index
        sentences_time_series = torch.from_numpy(self.all_data['sentences_time_series'][idx[item]]).float()
        words_time_series = torch.from_numpy(self.all_data['words_time_series'][idx[item]]).float()
        mean_time_series = torch.from_numpy(self.all_data['mean_time_series'][idx[item]]).float()
        # sentences = self.all_data['sentences'][idx[item]]
        tokens = self.all_data['tokens'][idx[item]]
        attention_mask = self.all_data['attention_mask'][idx[item]]
        labels = torch.from_numpy(self.all_data['labels'][idx[item]]).float()

        correlation = self.connectivity(sentences_time_series, activate=False)
        words_time_series = rearrange(words_time_series, "c f n w -> w (c f n)")
        mean_time_series = rearrange(mean_time_series, "f n -> (f n)")
        # words_time_series = self.norm(words_time_series)
        # sentences_time_series = self.norm(sentences_time_series)
        # correlation = self.correlation(time_series)
        # correlation = torch.from_numpy(self.all_data['correlation'][idx[item]]).float()

        if self.data_config.augmentation:
            pass

        # padding and truncating
        N, T = sentences_time_series.shape
        max_padding = MAX_EEG_LEN // self.args.p1 // self.args.p2
        t = math.ceil(T / self.args.p1 / self.args.p2)
        if T >= MAX_EEG_LEN:
            sentences_time_series = sentences_time_series[:, :MAX_EEG_LEN]
            sentences_time_series_mask = torch.ones(max_padding)
        else:
            sentences_time_series = torch.concat([sentences_time_series, torch.zeros((N, MAX_EEG_LEN - T))], dim=-1)
            sentences_time_series_mask = torch.concat([torch.ones(min(t, max_padding)),
                                                       torch.zeros(max(0, max_padding - t))], dim=-1)
        S, E = words_time_series.shape
        if S >= MAX_TEXT_LEN:
            words_time_series = words_time_series[:MAX_TEXT_LEN]
            words_time_series_mask = torch.ones(MAX_TEXT_LEN)
        else:
            words_time_series = torch.concat([words_time_series,
                                              torch.zeros((MAX_TEXT_LEN - S, E))], dim=0)
            words_time_series_mask = torch.concat([torch.ones(S), torch.zeros(MAX_TEXT_LEN - S)], dim=-1)

        return {'time_series': sentences_time_series,
                'time_series_mask': sentences_time_series_mask,
                'words_time_series': words_time_series,
                'words_time_series_mask': words_time_series_mask,
                'mean_time_series': mean_time_series,
                'correlation': correlation,
                "tokens": tokens,
                "attention_mask": attention_mask,
                'labels': labels}

    def select_subject(self):
        self.selected = [self.subject_id]
        index = np.sum(self.all_data["subject_id"] == i for i in self.selected) == 1
        self.all_data['words_time_series'] = self.all_data['words_time_series'][index]
        # self.all_data['sentences_time_series'] = self.all_data['sentences_time_series'][index]
        self.all_data['sentences_time_series'] = [self.all_data['sentences_time_series'][i]
                                                  for i, j in enumerate(index) if j]
        self.all_data['labels'] = self.all_data['labels'][index]
        self.all_data['subject_id'] = self.all_data['subject_id'][index]
        self.all_data['sentences'] = self.all_data['sentences'][index]

    def truncating(self):
        all_data = {}
        w_lens = np.array([s.shape[-1] for s in self.all_data["words_time_series"]])
        s_lens = np.array([s.shape[-1] for s in self.all_data["sentences_time_series"]])
        # idx = np.logical_and(w_lens >= MIN_TEXT_LEN, s_lens <= MAX_EEG_LEN)
        idx = np.logical_and(w_lens >= MIN_TEXT_LEN, s_lens >= MIN_EEG_LEN)
        idx = np.logical_and(idx, self.all_data["labels"] != "CONTROL")
        all_data['labels'] = self.all_data['labels'][idx]
        all_data['subject_id'] = self.all_data['subject_id'][idx]
        all_data['mean_time_series'] = self.all_data['mean_time_series'][idx]
        all_data['sentence_ids'] = self.all_data['sentence_ids'][idx]
        all_data['labels_list'] = self.all_data['labels_list'][self.all_data["labels_list"] != "CONTROL"]
        all_data['id_list'] = np.array(list(range(len(self.all_data["labels_list"]))))[self.all_data["labels_list"] != "CONTROL"]
        sentences = []
        words_time_series = []
        sentences_time_series = []
        for i, j in enumerate(idx):
            if j:
                sentences.append(self.all_data['sentences'][i])
                sentences_time_series.append(self.all_data['sentences_time_series'][i])
                words_time_series.append(self.all_data['words_time_series'][i])
        all_data['sentences'] = np.array(sentences)
        all_data['words_time_series'] = words_time_series
        all_data['sentences_time_series'] = sentences_time_series
        self.all_data = all_data
        self.data_config.time_series_size = MAX_EEG_LEN

    def tokenize(self):
        tokenized = self.tokenizer(self.all_data["sentences"].tolist(), max_length=MAX_TEXT_LEN, padding='max_length',
                                   truncation=True, return_tensors='pt')
        self.all_data["tokens"] = tokenized["input_ids"]
        self.all_data["attention_mask"] = tokenized["attention_mask"]


def zuco_preprocess_SR(path=r"D:\data\ZuCo"):
    subject_ids = []
    labels = []
    sentences_time_series = []
    sentence_ids = []
    mean_time_series = []
    words_time_series = []
    sentences = []
    label_path = os.path.join(path, "osfstorage-archive", "task_materials", "sentiment_labels_task1.csv")
    label_data = pd.read_csv(label_path, sep=";")
    # label = label_data[["sentiment_label"]].values
    # sentence = label_data[["sentence"]].values
    sentence_list = label_data.sentence.tolist()
    labels_list = label_data.sentiment_label.tolist()
    sentence_ids_list = label_data.sentence_id.tolist()
    channels = [f"eeg{i}" for i in range(104)]
    channel_types = ["eeg"] * 104
    info = mne.create_info(ch_names=channels,
                           sfreq=500,
                           ch_types=channel_types)

    for subject_id, name in enumerate(names):
        file_path = os.path.join(path, "osfstorage-archive", "task1-SR/Matlab files", f"results{name}_SR.mat")
        data = io.loadmat(file_path, squeeze_me=True, struct_as_record=False)['sentenceData']

        for i in range(len(data)):
            flag = True
            if isinstance(data[i].rawData, float) or data[i].rawData.shape[-1] < 500:
                continue
            ts = data[i].rawData[:104]
            ts[np.isnan(ts)] = 0

            raw = mne.io.RawArray(ts, info)
            raw.resample(128)
            raw.filter(4, 38)
            ts = raw.get_data()

            freq = ["t1", "t2", "a1", "a2", "g1", "g2", "b1", "b2"]
            mts = []
            for f in freq:
                d_f = eval(f"data[i].mean_{f}")[:104]
                d_f[np.isnan(d_f)] = 0
                mts.append(d_f)
            if len(mts) == 0:
                continue
            mts = np.stack(mts, axis=0)

            word = data[i].word
            ffd = []
            for f in freq:
                d_f = [eval(f"w.FFD_{f}")[:104] for w in word if w.nFixations > 0]
                if len(d_f) == 0:
                    flag = False
                    break
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                ffd.append(d_f)
            if not flag:
                continue
            ffd = np.stack(ffd, axis=0)
            trt = []
            for f in freq:
                d_f = [eval(f"w.TRT_{f}")[:104] for w in word if w.nFixations > 0]
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                trt.append(d_f)
            trt = np.stack(trt, axis=0)
            gd = []
            for f in freq:
                d_f = [eval(f"w.GD_{f}")[:104] for w in word if w.nFixations > 0]
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                gd.append(d_f)
            gd = np.stack(gd, axis=0)

            words_time_series.append(np.stack([ffd, trt, gd], axis=0))
            sentences_time_series.append(ts)
            mean_time_series.append(mts)
            sentence = data[i].content
            if sentence == 'Ultimately feels emp11111ty and unsatisfying, like swallowing a Communion wafer without the wine.':
                sentence = 'Ultimately feels empty and unsatisfying, like swallowing a Communion wafer without the wine.'
            elif sentence == "Bullock's complete lack of focus and ability quickly derails the film.1":
                sentence = "Bullock's complete lack of focus and ability quickly derails the film."
            sentences.append(sentence)

            sentence_idx = sentence_list.index(sentence)
            label = labels_list[sentence_idx]
            labels.append(label)
            sentence_ids.append(sentence_idx)
            subject_ids.append(subject_id + 1)

    mean_time_series = np.stack(mean_time_series, axis=0)
    labels = np.array(labels)
    subject_ids = np.array(subject_ids)
    sentence_ids = np.array(sentence_ids)
    labels_list = np.array(labels_list)
    np.save(os.path.join(path, f"ZuCo-SR.npy"), {"labels": labels,
                                                 "labels_list": labels_list,
                                                 "sentences": sentences,
                                                 "sentence_ids": sentence_ids,
                                                 "words_time_series": words_time_series,
                                                 "mean_time_series": mean_time_series,
                                                 "sentences_time_series": sentences_time_series,
                                                 "subject_id": subject_ids})


def zuco_preprocess_NR(path=r"D:\data\ZuCo"):
    subject_ids = []
    labels = []
    sentences_time_series = []
    mean_time_series = []
    words_time_series = []
    sentences = []
    sentence_ids = []
    label_path = os.path.join(path, "osfstorage-archive", "task_materials", "relations_labels_task2.csv")
    label_data = pd.read_csv(label_path, sep=",")
    sentence_list = label_data.sentence.tolist()
    labels_list = label_data.relation_types.tolist()
    channels = [f"eeg{i}" for i in range(104)]
    channel_types = ["eeg"] * 104
    info = mne.create_info(ch_names=channels,
                           sfreq=500,
                           ch_types=channel_types)

    for subject_id, name in enumerate(names):
        file_path = os.path.join(path, "osfstorage-archive", "task2-NR/Matlab files", f"results{name}_NR.mat")
        data = io.loadmat(file_path, squeeze_me=True, struct_as_record=False)['sentenceData']

        for i in range(len(data)):
            flag = True
            if isinstance(data[i].rawData, float) or data[i].rawData.shape[-1] < 500:
                continue
            ts = data[i].rawData[:104]
            ts[np.isnan(ts)] = 0

            raw = mne.io.RawArray(ts, info)
            raw.resample(128)
            raw.filter(4, 38)
            ts = raw.get_data()

            freq = ["t1", "t2", "a1", "a2", "g1", "g2", "b1", "b2"]
            mts = []
            for f in freq:
                d_f = eval(f"data[i].mean_{f}")[:104]
                d_f[np.isnan(d_f)] = 0
                mts.append(d_f)
            if len(mts) == 0:
                continue
            mts = np.stack(mts, axis=0)

            word = data[i].word
            ffd = []
            for f in freq:
                d_f = [eval(f"w.FFD_{f}")[:104] for w in word if w.nFixations > 0]
                if len(d_f) == 0:
                    flag = False
                    break
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                ffd.append(d_f)
            if not flag:
                continue
            ffd = np.stack(ffd, axis=0)
            trt = []
            for f in freq:
                d_f = [eval(f"w.TRT_{f}")[:104] for w in word if w.nFixations > 0]
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                trt.append(d_f)
            trt = np.stack(trt, axis=0)
            gd = []
            for f in freq:
                d_f = [eval(f"w.GD_{f}")[:104] for w in word if w.nFixations > 0]
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                gd.append(d_f)
            gd = np.stack(gd, axis=0)

            words_time_series.append(np.stack([ffd, trt, gd], axis=0))
            sentences_time_series.append(ts)
            mean_time_series.append(mts)
            sentence = data[i].content
            sentence = sentence.replace('–', "-")
            sentence = sentence.replace('�', "’")
            sentence = sentence.replace('Wuerttemberg', "Wurttemberg")
            sentence = sentence.replace('111Senator', "Senator")
            sentence = sentence.replace('1902’19', "1902-19")
            if sentence == 'In 1954 she, along with Bing Crosby, Danny Kaye, and nVera-Ellen, starred in the movie White Christmas.':
                sentence = 'In 1954 she, along with Bing Crosby, Danny Kaye, and Vera-Ellen, starred in the movie White Christmas.'
            sentences.append(sentence)

            sentence_idx = sentence_list.index(sentence)
            label = labels_list[sentence_idx]
            labels.append(label)
            sentence_ids.append(sentence_idx)
            subject_ids.append(subject_id + 1)
    mean_time_series = np.stack(mean_time_series, axis=0)
    labels = np.array(labels)
    subject_ids = np.array(subject_ids)
    sentence_ids = np.array(sentence_ids)
    labels_list = np.array(labels_list)
    np.save(os.path.join(path, f"ZuCo-NR.npy"), {"labels": labels,
                                                 "labels_list": labels_list,
                                                 "sentences": sentences,
                                                 "sentence_ids": sentence_ids,
                                                 "words_time_series": words_time_series,
                                                 "mean_time_series": mean_time_series,
                                                 "sentences_time_series": sentences_time_series,
                                                 "subject_id": subject_ids})


def zuco_preprocess_TSR(path=r"D:\data\ZuCo"):
    subject_ids = []
    labels = []
    sentences_time_series = []
    mean_time_series = []
    words_time_series = []
    sentences = []
    sentence_ids = []
    label_path = os.path.join(path, "osfstorage-archive", "task_materials", "relations_labels_task3.csv")
    label_data = pd.read_csv(label_path, sep=";")
    sentence_list = label_data.sentence.tolist()
    labels_list = label_data["relation-type"].tolist()
    channels = [f"eeg{i}" for i in range(104)]
    channel_types = ["eeg"] * 104
    info = mne.create_info(ch_names=channels,
                           sfreq=500,
                           ch_types=channel_types)

    for subject_id, name in enumerate(names):
        file_path = os.path.join(path, "osfstorage-archive", "task3-TSR/Matlab files", f"results{name}_TSR.mat")
        data = io.loadmat(file_path, squeeze_me=True, struct_as_record=False)['sentenceData']

        for i in range(len(data)):
            flag = True
            if isinstance(data[i].rawData, float) or data[i].rawData.shape[-1] < 500:
                continue
            ts = data[i].rawData[:104]
            ts[np.isnan(ts)] = 0

            raw = mne.io.RawArray(ts, info)
            raw.resample(128)
            raw.filter(4, 38)
            ts = raw.get_data()

            freq = ["t1", "t2", "a1", "a2", "g1", "g2", "b1", "b2"]
            mts = []
            for f in freq:
                d_f = eval(f"data[i].mean_{f}")[:104]
                d_f[np.isnan(d_f)] = 0
                mts.append(d_f)
            if len(mts) == 0:
                continue
            mts = np.stack(mts, axis=0)

            word = data[i].word
            ffd = []
            for f in freq:
                d_f = [eval(f"w.FFD_{f}")[:104] for w in word if w.nFixations > 0]
                if len(d_f) == 0:
                    flag = False
                    break
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                ffd.append(d_f)
            if not flag:
                continue
            ffd = np.stack(ffd, axis=0)
            trt = []
            for f in freq:
                d_f = [eval(f"w.TRT_{f}")[:104] for w in word if w.nFixations > 0]
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                trt.append(d_f)
            trt = np.stack(trt, axis=0)
            gd = []
            for f in freq:
                d_f = [eval(f"w.GD_{f}")[:104] for w in word if w.nFixations > 0]
                d_f = np.stack(d_f, axis=-1)
                d_f[np.isnan(d_f)] = 0
                gd.append(d_f)
            gd = np.stack(gd, axis=0)

            words_time_series.append(np.stack([ffd, trt, gd], axis=0))
            sentences_time_series.append(ts)
            mean_time_series.append(mts)
            sentence = data[i].content
            sentence = sentence.replace("(40 km�)", "(40 km)")
            sentences.append(sentence)

            sentence_idx = sentence_list.index(sentence)
            label = labels_list[sentence_idx]
            labels.append(label)
            sentence_ids.append(sentence_idx)
            subject_ids.append(subject_id + 1)

    mean_time_series = np.stack(mean_time_series, axis=0)
    labels = np.array(labels)
    subject_ids = np.array(subject_ids)
    sentence_ids = np.array(sentence_ids)
    labels_list = np.array(labels_list)
    np.save(os.path.join(path, f"ZuCo-TSR.npy"), {"labels": labels,
                                                  "labels_list": labels_list,
                                                  "sentences": sentences,
                                                  "sentence_ids": sentence_ids,
                                                  "words_time_series": words_time_series,
                                                  "mean_time_series": mean_time_series,
                                                  "sentences_time_series": sentences_time_series,
                                                  "subject_id": subject_ids})

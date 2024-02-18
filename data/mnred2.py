import os
from random import randrange

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
import mne
from transformers import BertTokenizer, RobertaTokenizer, BartTokenizer

from .data_config import DataConfig
from .dataset import BaseDataset

MAX_TEXT_LEN = 32


class MNRED2Dataset(BaseDataset):
    def __init__(self, args, data_config: DataConfig, k=0, train=True, subject_id=0, one_hot=True):
        # self.tokenizer = BertTokenizer.from_pretrained('/data/models/bert-base-uncase')
        # self.tokenizer = RobertaTokenizer.from_pretrained('/data/models/roberta-base')
        self.tokenizer = BartTokenizer.from_pretrained('/data/models/bart-base')
        super(MNRED2Dataset, self).__init__(args, data_config, k, train, subject_id=subject_id)

    def load_data(self, one_hot=True):
        self.all_data = np.load(self.data_config.data_dir, allow_pickle=True).item()

        self.data_config.node_size = self.data_config.node_feature_size = self.all_data['timeseries'][0].shape[0]
        self.data_config.time_series_size = self.all_data['timeseries'][0].shape[1]
        self.data_config.num_class = 2

        self.data_config.class_weight = [1.5, 3.5]
        self.tokenize()

        groups = self.all_data['labels_list']
        sentence_ids = np.arange(0, 500)
        train_ids, test_ids = list(self.k_fold.split(sentence_ids, groups))[self.k]
        train_ids = sentence_ids[train_ids]
        test_ids = sentence_ids[test_ids]
        self.train_index = [i for i, j in enumerate(self.all_data['sentence_ids']) if j in train_ids]
        self.test_index = [i for i, j in enumerate(self.all_data['sentence_ids']) if j in test_ids]
        self.all_data['labels'] = F.one_hot(torch.from_numpy(self.all_data['labels']).to(torch.int64)).numpy()

    def __len__(self):
        return len(self.train_index) if self.train else len(self.test_index)

    def __getitem__(self, item):
        idx = self.train_index if self.train else self.test_index
        time_series = torch.from_numpy(self.all_data['timeseries'][idx[item]]).float()
        # sentences = self.all_data['sentences'][idx[item]]
        tokens = self.all_data['tokens'][idx[item]]
        attention_mask = self.all_data['attention_mask'][idx[item]]
        labels = torch.from_numpy(self.all_data['labels'][idx[item]]).float()

        correlation = self.all_data['corr'][idx[item]]
        # words_time_series = self.norm(words_time_series)
        # sentences_time_series = self.norm(sentences_time_series)
        # correlation = self.correlation(time_series)
        # correlation = torch.from_numpy(self.all_data['correlation'][idx[item]]).float()

        return {'time_series': time_series,
                'correlation': correlation,
                "tokens": tokens,
                "attention_mask": attention_mask,
                'labels': labels}

    def select_subject(self):
        # self.selected = [self.subject_id]
        self.selected = [6, 7, 8, 11, 13, 14, 15, 21, 22, 24]
        index = np.sum(self.all_data["subject_id"] == i for i in self.selected) == 1
        self.all_data['time_series'] = self.all_data['time_series'][index]
        self.all_data['correlation'] = self.all_data['corr'][index]
        self.all_data['labels'] = self.all_data['labels'][index]
        self.all_data['subject_id'] = self.all_data['subject_id'][index]
        self.all_data['sentences'] = [self.all_data['sentences'][i] for i, j in enumerate(index) if j]

    def tokenize(self):
        tokenized = self.tokenizer(self.all_data["sentences"].tolist(), max_length=MAX_TEXT_LEN, padding='max_length',
                                   truncation=True, return_tensors='pt')
        self.all_data["tokens"] = tokenized["input_ids"]
        self.all_data["attention_mask"] = tokenized["attention_mask"]


def eeg_preprocess_test(path):
    time_series = []
    labels = []
    subject_ids = []
    sentences = []
    sentences_ids = []

    all_data = h5py.File('/data/datasets/MNRED2/event_eeg.mat')
    # all_data = h5py.File('D:\data\MNRED\event_eeg.mat')
    refs = [[r for r in all_data['event_eeg_struct'][f'event_{i}'][0]] for i in range(1, 501)]
    all_sentences = pd.read_excel('/data/datasets/MNRED2/materials.xlsx', sheet_name=[0, 1, 2, 3, 4])
    # all_sentences = pd.read_excel('D:\data\MNRED\materials.xlsx', sheet_name=[0, 1, 2, 3, 4])
    all_sentences = pd.concat(all_sentences)["素材语句"].values

    channels = [f"eeg{i}" for i in range(1, 33)]
    channel_types = ["eeg"] * 32
    channels_of_interest = [f"eeg{i}" for i in range(1, 16)] + [f"eeg{i}" for i in range(18, 33)]
    info = mne.create_info(ch_names=channels,
                           sfreq=1100,
                           ch_types=channel_types)
    labels_list = ([1] * 30 + [0] * 70)*5
    for sentence_id, (sentence, label, sentence_ref) in enumerate(zip(all_sentences, labels_list, refs)):
        labels += [label] * 30
        sentences += [sentence] * 30
        sentences_ids += [sentence_id+1] * 30
        subject_ids += list(range(1, 31))
        for trail_ref in sentence_ref:
            data = np.array(all_data[trail_ref]).transpose().tolist()
            time_series.append(data)
    time_series = np.array(time_series)
    labels = np.array(labels)
    subject_ids = np.array(subject_ids)
    sentences = np.array(sentences)
    sentences_ids = np.array(sentences_ids)

    new_time_series = []

    for ts in time_series:
        # ts = np.concatenate([ts[:15, :], ts[17:, :]], axis=0)
        raw = mne.io.RawArray(ts, info)
        raw = raw.pick_channels(channels_of_interest)
        ica = mne.preprocessing.ICA(n_components=None, random_state=0)
        ica.fit(raw)
        raw = ica.apply(raw)
        raw = raw.filter(0.5, 80, method='iir').resample(sfreq=220)
        preprocessed_data = raw.get_data()
        new_time_series.append(preprocessed_data)
    new_time_series = np.array(new_time_series)
    labels_list = np.array(labels_list)
    pearson = np.array([np.corrcoef(t) for t in time_series])
    np.save(os.path.join(path, "MNRED2_ICA.npy"), {"timeseries": new_time_series,
                                                   "corr": pearson,
                                                   "labels": labels,
                                                   "labels_list": labels_list,
                                                   "subject_id": subject_ids,
                                                   "sentences": sentences,
                                                   "sentence_ids": sentences_ids}
            )

if __name__ == '__main__':
    eeg_preprocess_test("../data/EEG")

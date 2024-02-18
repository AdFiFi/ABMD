from random import randrange
from abc import abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from nilearn import connectome
from sklearn.covariance import OAS
from .data_config import DataConfig


class BaseDataset(Dataset):
    def __init__(self, args, data_config: DataConfig, k=0, train=True, subject_id=0):
        super(BaseDataset, self).__init__()
        self.args = args
        self.data_config = data_config
        self.train = train
        if data_config.n_splits-1:
            self.k_fold = StratifiedKFold(n_splits=data_config.n_splits, shuffle=True, random_state=0)
        else:
            self.k_fold = None
        self.k = k
        self.subject_id = subject_id
        self.selected = []
        self.all_data = dict()
        self.ts_train_index = None
        self.ts_test_index = None
        self.vision_train_index = None
        self.vision_test_index = None
        self.train_index = None
        self.test_index = None
        self.train_data = None
        self.test_data = None
        self.channels = None
        self.channel_types = None
        self.info = None
        self.load_data()

    @abstractmethod
    def load_data(self):
        pass

    @staticmethod
    def connectivity(time_series, activate=True):
        conn_measure = connectome.ConnectivityMeasure(kind='correlation')
        # conn_measure = connectome.ConnectivityMeasure(kind='correlation', cov_estimator=OAS(store_precision=False))
        connectivity = conn_measure.fit_transform(time_series.T.unsqueeze(0).numpy())[0]
        connectivity = torch.from_numpy(connectivity)
        if activate:
            connectivity = torch.arctanh(connectivity)
            connectivity = torch.clamp(connectivity, -1.0, 1.0)
            diag = torch.diag_embed(torch.diag(connectivity))
            connectivity = connectivity - diag
        return connectivity

    @staticmethod
    def correlation(time_series, activate=True):
        feature = torch.einsum('nt, mt ->nm', time_series, time_series) / (time_series.size(1)-1)
        feature = torch.clamp(feature, -1.0, 1.0)
        if activate:
            feature = torch.arctanh(feature)
            feature = torch.clamp(feature, -1.0, 1.0)
            diag = torch.diag_embed(torch.diag(feature))
            feature = feature - diag
        return feature

    @staticmethod
    def norm(time_series):
        _, n, t = time_series.shape
        time_series -= np.repeat(np.mean(time_series, (1, 2)), t*n).reshape((-1, n, t))
        time_series /= np.repeat(np.std(time_series, (1, 2)), t*n).reshape((-1, n, t))
        return time_series

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    def augmentations(self, x, modality="text"):
        pass

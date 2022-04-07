from __future__ import print_function, division
from cgi import test
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import KFold


def kfoldsplits(X):
    """Split annotations"""
    kf = KFold(n_splits=10, shuffle=False)
    splits = []
    for trainIdx, validIdx in kf.split(X):
        splits.append((trainIdx, validIdx))
        
    print("The first index of the first split is ", splits[0][0][0])

    return splits


class RNAData(object):
    def __init__(self, data_path, test_path, fc_path):

        np.random.seed(42)

        # Preprocess data
        model_data = pd.read_csv(data_path, index_col=0).values[:, 2:].astype(np.float32)
        test_data = pd.read_csv(test_path, index_col=0).values[:, 2:].astype(np.float32)
        fc_data = pd.read_csv(fc_path, index_col=0).values[:, 1:].astype(np.float32)

        self.column_labels = pd.read_csv(test_path, index_col=0).columns[2:(2649 + 384 + 2)].values.astype(str)
        print("Model data: ", model_data.shape, type(model_data))
        print("Test data: ", test_data.shape, type(test_data))
        print("fc data: ", fc_data.shape, type(fc_data))


        # Sum up deletions and insertions to
        X = model_data[:, :(2649 + 384)]
        y = model_data[:, (2649 + 384):]

        X_test = test_data[:, :(2649 + 384)]
        y_test = test_data[:, (2649 + 384):]

        X_fc = fc_data[:, :(2649 + 384)]
        y_fc = fc_data[:, (2649 + 384):]

        print("X Shape ", X.shape, " | y shape ", y.shape)
        print("X_test Shape ", X_test.shape, " | y_test shape ", y_test.shape)
        print("X_fc Shape ", X_fc.shape, " | y_fc shape ", y_fc.shape)

        # Randomly shuffle data
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]
        test_idx = np.arange(len(y_test))
        np.random.shuffle(test_idx)
        X_test, y_test = X_test[test_idx], y_test[test_idx]
        fc_idx = np.arange(len(y_fc))
        np.random.shuffle(fc_idx)
        X_fc, y_fc = X_fc[fc_idx], y_fc[fc_idx]

        print("Now removing samples with only insertion events")
        X_deletion, y_deletion = [], []
        X_test_deletion, y_test_deletion = [], []
        X_fc_deletion, y_fc_deletion = [], []

        # Remove samples that only have insertion events:
        for i in range(model_data.shape[0]):
            if 1> sum(y[i,:536])> 0 :
                y_deletion.append(y[i,:536]/sum(y[i,:536]))
                X_deletion.append(X[i])
                
        X_deletion, y_deletion = np.array(X_deletion), np.array(y_deletion)

        for i in range(test_data.shape[0]):
            if 1> sum(y_test[i,:536])> 0 :
                y_test_deletion.append(y_test[i,:536]/sum(y_test[i,:536]))
                X_test_deletion.append(X_test[i])
                
        X_test_deletion, y_test_deletion = np.array(X_test_deletion), np.array(y_test_deletion)

        for i in range(fc_data.shape[0]):
            if 1 > sum(y_fc[i, :536]) > 0:
                y_fc_deletion.append(y_fc[i, :536] / sum(y_fc[i, :536]))
                X_fc_deletion.append(X_fc[i])

        X_fc_deletion, y_fc_deletion = np.array(X_fc_deletion), np.array(y_fc_deletion)

        print("X_deletion Shape ", X_deletion.shape, " | y_deletion shape ", y_deletion.shape)
        print("X_test_deletion Shape ", X_test_deletion.shape, " | y_test_deletion shape ", y_test_deletion.shape)
        print("X_fc_deletion Shape ", X_fc_deletion.shape, " | y_fc_deletion shape ", y_fc_deletion.shape)

        splits = kfoldsplits(X_deletion)
        print("Number of train/val splits: ", len(splits))
        train_split, val_split = splits[0]

        self.test_file = X_test_deletion
        self.train_file = X_deletion[train_split]
        self.val_file = X_deletion[val_split]
        self.fc_file = X_fc_deletion

        print("Train:", self.train_file.shape)
        print("Test:", self.test_file.shape)
        print("Val:", self.val_file.shape)
        print("FC_Test:", self.fc_file.shape)

    def get_dataset(self, split):

        if split == 'train':
            return RNADataset(self.train_file)
        elif split == 'test':
            return RNADataset(self.test_file)
        elif split == 'val':
            return RNADataset(self.val_file)
        elif split == 'testy':
            return self.column_labels
        elif split == 'trainval':
            return RNADataset(np.vstack((self.train_file, self.val_file)))
        elif split == 'fc':
            return RNADataset(self.fc_file)


class RNADataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data):
        """
        Args:
            xena_file (string): Path to the csv file with annotations.
        """
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

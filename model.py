import argparse
import numpy as np

import scipy
from scipy import io as scio

import torch
from torch.utils.data import DataLoader, Dataset


class ListDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def init_loaders(features_path, labels_path, batch_size=2, shuffle=True, ratios=[8, 1]):
    labels = scipy.io.loadmat(labels_path)['real_rgb']
    features = np.load(features_path)[()]
    voters = list(features.keys())
    size = len(labels)
    
    features_list = []
    for index in range(size):
        features_list.append(([], labels[index]))
        for voter in voters:
            features_list[-1][0].append(features[voter][index])

    train_end = int(ratios[0] * len(features_list))
    valid_end = int(ratios[1] * len(features_list))
    train_dataset = ListDataset(features_list[:train_end])
    valid_dataset = ListDataset(features_list[train_end:valid_end])
    test_dataset = ListDataset(features_list[valid_end:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader, test_loader
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--features", required=True, help="path to features file")
    parser.add_argument("-l", "--labels", required=True, help="path to labels file")
    args = vars(parser.parse_args())

    # load the image from disk
    init_loaders(args["features"], args["labels"])
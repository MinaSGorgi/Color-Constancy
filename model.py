import argparse
import numpy as np
import scipy.io as io
from torch.utils.data import DataLoader, Dataset


class ListDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def construct_loaders(features_path, labels_path, batch_size=2, shuffle=True, ratios=[0.8, 0.1]):
    features = np.load(features_path)[()]
    labels = io.loadmat(labels_path)['real_rgb']
    voters = list(features.keys())
    size = len(features[voters[0]])

    features_list = np.zeros((size, len(voters) * 3))
    for i in range(size):
        features_list[i, :3] = features[voters[0]][i]
        features_list[i, 3:6] = features[voters[1]][i]
        features_list[i, 6:9] = features[voters[2]][i]

    train_size = int(ratios[0] * len(features_list))
    valid_size = int(ratios[1] * len(features_list))
    test_size = int((1 - ratios[1] - ratios[0]) * len(features_list))
    train_end = train_size
    valid_end = valid_size + train_end
    test_end = valid_end + test_size

    features_list = list(zip(features_list, labels))

    train_dataset = ListDataset(features_list[:train_end])
    valid_dataset = ListDataset(features_list[train_end:valid_end])
    test_dataset = ListDataset(features_list[test_end:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--features", required=True, help="path to features file")
    parser.add_argument("-l", "--labels", required=True, help="path to labels file")
    args = vars(parser.parse_args())

    # load the features from disk
    train_loader, valid_loader, test_loader = construct_loaders(args["features"], args["labels"])

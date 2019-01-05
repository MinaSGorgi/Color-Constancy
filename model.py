import argparse
import numpy as np
import scipy.io as io

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch import optim


class ListDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def construct_loaders(features_path, labels_path, batch_size=64, shuffle=True, ratios=[0.8, 0.1]):
    features = np.load(features_path)[()]
    labels = io.loadmat(labels_path)['real_rgb']
    voters = list(features.keys())
    size = len(features[voters[0]])

    features_list = np.zeros((size, len(voters) * 3), dtype=np.float)
    for fi in range(size):
        for vi in range(len(voters)):
            features_list[fi, vi * 3:vi * 3 + 3] = features[voters[vi]][fi]
    dataset_list = list(zip(features_list, labels))

    train_size = int(ratios[0] * len(features_list))
    valid_size = int(ratios[1] * len(features_list))

    train_dataset = ListDataset(dataset_list[:train_size])
    valid_dataset = ListDataset(dataset_list[train_size:train_size+valid_size])
    test_dataset = ListDataset(dataset_list[train_size+valid_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle)

    return train_loader, valid_loader, test_loader


class ColorConstancyModel(nn.Module):

    def __init__(self, network_arch, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        for input_size, output_size in network_arch:
            self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.dropout(self.relu(self.layers[i](x)))
        x = self.layers[-1](x)
        return x


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_device(use_gpu=True):
    """
    TODO: add documentation here
    """
    if not use_gpu:
        print('Usage of gpu is not allowed! Using cpu instead ...')
        device = torch.device('cpu')
    elif not torch.cuda.is_available():
        print('No support for CUDA available! Using cpu instead ...')
        device = torch.device('cpu')
    else:
        print('Support for CUDA available! Using gpu ...')
        device = torch.device('cuda')

    return device


def train(n_epochs, model, optimizer, scheduler, criterion, use_gpu=True):
    loss_list = []

    device = get_device(use_gpu)
    model.to(device)

    valid_loss_min = np.Inf  # track change in validation loss
    train_error = []
    valid_error = []

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for features, labels in train_loader:
            # move tensors to GPU if CUDA is available
            features, labels = features.to(device).float(), labels.to(device).float()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(features)
            # calculate the batch loss
            loss = criterion(output, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.data.numpy().sum()
            loss_list.append(loss.data.numpy().sum())

        ######################
        # validate the model #
        ######################
        model.eval()
        for features, labels in valid_loader:
            # move tensors to GPU if CUDA is available
            features, labels = features.to(device).float(), labels.to(device).float()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(features)
            # calculate the batch loss
            loss = criterion(output, labels)
            # update average validation loss
            valid_loss += loss.data.numpy().sum()

        scheduler.step()
        print("learning rate = ", get_lr(optimizer))

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        valid_error.append(valid_loss)
        train_error.append(train_loss)

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f})'.format(
                valid_loss_min,
                valid_loss))
            print('Saving model ...')
            file_name = "best model at epoch " + str(epoch) + ".pth"
            torch.save(model.state_dict(), file_name)

            valid_loss_min = valid_loss
    for param in model.parameters():
        print(param)
    loss_list = np.array(loss_list)
    print("The loss mean {:.6f} and var {:.6f}".format(loss_list.mean(), loss_list.var()))

    return model


def sanity_check(model, test_loader, use_gpu=True):
    device = get_device(use_gpu)
    model.to(device)
    model.eval()

    for features, labels in test_loader:
        # move tensors to GPU if CUDA is available
        features, labels = features.to(device).float(), labels.to(device).float()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(features)
        original_loss = [angular_error(features.data[0, index * 3:index * 3 + 3], labels).data[0] for index in range(len(features[0])//3)]
        print("before: " + str(original_loss), "\tafter: " + str(output))


def angular_error(output, label):
    output_v = Variable(output, requires_grad=True)
    label_v = Variable(label, requires_grad=True)
    return torch.acos(torch.sum(output_v * label_v, dim=1))


class AngularLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, label):
        output_v = Variable(output, requires_grad=True)
        label_v = Variable(label, requires_grad=True)
        return torch.mean(torch.atan((torch.norm(torch.cross(output, label), dim=1)) / (torch.sum(output * label, dim=1))) * 180 / np.pi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--features", required=True, help="path to features file")
    parser.add_argument("-l", "--labels", required=True, help="path to labels file")
    args = vars(parser.parse_args())

    n_epochs = 10
    batch_size = 10

    lr = 1
    momentum = 0.1
    step_size = 2
    gamma = 0.5

    model = ColorConstancyModel([(9, 16), (16, 3)])

    # load the features from disk
    train_loader, valid_loader, test_loader = construct_loaders(args["features"], args["labels"], batch_size)
    criterion = AngularLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    model = train(n_epochs, model, optimizer, scheduler, criterion)

    sanity_check(model, test_loader)


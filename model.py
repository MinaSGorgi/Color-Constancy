import utils
import voters

import argparse
import numpy as np
import scipy.io as scio

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


class ListDataset(Dataset):
    """
    TODO: add documentation here
    """
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


class AngularLoss(torch.nn.Module):
    """
    TODO: add documentation here
    """
    def forward(self, output, label):
        return torch.mean(torch.atan2((torch.norm(torch.cross(output, label), dim=1)), (torch.sum(output * label, dim=1))) * 180 / np.pi)


class ColorConstancyModel(nn.Module):
    """
    TODO: add documentation here
    """
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
        return self.layers[-1](x)


def construct_loaders(features_path, labels_path, batch_size=64, shuffle=True, ratios=[0.8, 0.1]):
    """
    TODO: add documentation here
    """
    features = np.load(features_path)[()]
    labels = scio.loadmat(labels_path)['real_rgb']
    voters = list(features.keys())
    size = len(labels)

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


def get_lr(optimizer):
    """
    TODO: add documentation here
    """
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
    """
    TODO: add documentation here
    """
    device = get_device(use_gpu)
    model.to(device)

    valid_loss_min = np.Inf  # track change in validation loss
    train_error = []
    valid_error = []
    loss_list = []

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

    loss_list = np.array(loss_list)
    print("The loss mean {:.6f} and var {:.6f}".format(loss_list.mean(), loss_list.var()))

    return model


def sanity_check(model, test_loader, use_gpu=True):
    """
    TODO: add documentation here
    """
    device = get_device(use_gpu)
    model.to(device)
    model.eval()

    original_loss_list = []
    final_loss_list = []
    for features, labels in test_loader:
        # move tensors to GPU if CUDA is available
        features, labels = features.to(device).float(), labels.to(device).float()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(features)
        original_loss = [utils.angular_error(features.data[0, index * 3:index * 3 + 3].detach().numpy(), np.transpose(labels.detach().numpy()))[0] for index in range(len(features[0])//3)]
        
        final_loss = utils.angular_error(output.detach().numpy(), np.transpose(labels.detach().numpy()))[0][0]

        original_loss_list.append(original_loss[-1].item())
        final_loss_list.append(final_loss)

    print("Original:")
    utils.print_stats(original_loss_list)
    print("Final:")
    utils.print_stats(final_loss_list)


def predict(model, image):
    """
    TODO: add documentation here
    """
    features = [0.]*9
    features[:3] = voters.grey_edge(image, njet=0, mink_norm=1, sigma=0)
    features[3:6] = voters.grey_edge(image, njet=0, mink_norm=-1, sigma=0)
    features[6:] = voters.grey_edge(image, njet=1, mink_norm=5, sigma=2)
    
    model.eval()
    prediction = model(torch.Tensor(features))
    return prediction.numpy().tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--features", required=True, help="path to features file")
    parser.add_argument("-l", "--labels", required=True, help="path to labels file")
    parser.add_argument("-b", "--batch", help="batch size for loaders", default=10)
    parser.add_argument("-e", "--epochs", help="number of training epochs", default=10)
    parser.add_argument("-r", "--lr", help="initial learning rate", default=1)
    parser.add_argument("-m", "--momentum", help="optimizer momentum", default=0.1)
    parser.add_argument("-s", "--step", help="optimizer step size", default=2)
    parser.add_argument("-g", "--gamma", help="optimizer gamma", default=0.9)
    parser.add_argument("-d", "--dict", help="path to saved dictionary", default=None)
    parser.add_argument("-gpu", action='store_true', help="allow using gpu", default=False)
    args = vars(parser.parse_args())

    # setup model
    model = ColorConstancyModel([(9, 27), (27, 3)])
    if args["dict"] is not None:
        model.load_state_dict(torch.load(args["dict"]))

    # load the features from disk
    train_loader, valid_loader, test_loader = construct_loaders(args["features"], args["labels"], args["batch"])
    #setup hyper parameters
    criterion = AngularLoss()
    optimizer = optim.SGD(model.parameters(), lr=args["lr"], momentum=args["momentum"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args["step"], gamma=args["gamma"])

    # train
    model = train(args["epochs"], model, optimizer, scheduler, criterion, use_gpu=args["gpu"])
    # error stats check
    sanity_check(model, test_loader)

import argparse
import os
import torch
import datetime
from utils import get_accuracy, plot_results
from data.get_data import get_data_loaders
from models.orig_primary import PrimaryNetwork
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import math

def end_of_training_stats(outputs_dict, output_path):

    plot_results(outputs_dict, output_path, architecture)


def print_size_ratio(network, std, print_rat = False):

    size_rat = network.get_size_ratio(std)
    if print_rat:
        print("\t \t Size ratio", size_rat)
    return size_rat


def get_std(curr_epoch, stds, std_epochs):

    std = 0.0
    for i in range(len(std_epochs)):
        if curr_epoch >= std_epochs[i]:
            std = stds[i]
        else:
            break
    return std


def get_l1_coeff(curr_epoch, coeffs, coeffs_epochs):

    l1_coeff = 0.0
    for i in range(len(coeffs_epochs)):
        if curr_epoch >= coeffs_epochs[i]:
            l1_coeff = coeffs[i]
        else:
            break
    return l1_coeff


def optimizers_epochs(network, epoch, lr, weight_decay, momentum):

    if (epoch >= 60) and (epoch < 120):
        optimizer = optim.SGD(network.parameters(), lr=lr * 0.2, weight_decay=weight_decay, momentum=momentum)
    elif (epoch >= 120) and (epoch < 160):
        optimizer = optim.SGD(network.parameters(), lr=lr * 0.04, weight_decay=weight_decay, momentum=momentum)
    elif (epoch >= 160):
        optimizer = optim.SGD(network.parameters(), lr=lr * 0.008, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optim.SGD(network.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    return optimizer


def train(regularize, dataset_name, dropout, transform_train, transform_test, batch_size, lr, momentum, random_type,
          weight_decay, train_epochs, setup, regularize_2, architecture, stds, std_epochs, coeffs, coeffs_epochs, means,
          root=None):

    # Creating the directory in which the results and the graphs will be stored

    output_path = os.getcwd()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, "Resnet18_{}".format(dataset_name))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    t_now = datetime.datetime.now()
    output_path = os.path.join(output_path, "{}_{}_{}_{}".format(t_now.month, t_now.day, t_now.hour, t_now.minute))
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    print("\t \t", "=" * 80)
    print("\t \t Before getting data loader")
    print("\t \t", "=" * 80)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader, num_classes = get_data_loaders(dataset_name=dataset_name, transform_train_name=transform_train,
                                                            transform_test_name=transform_test, batch_size=batch_size,
                                                            root=root)

    print("\t \t", "=" * 80)
    print("\t \t After getting data loader")
    print("\t \t", "=" * 80)

    print("\t \t", "=" * 80)
    print("\t \t Before getting models")
    print("\t \t", "=" * 80)

    # Getting the main network

    network = PrimaryNetwork(num_classes, device, regularize, 1, dropout[0], random_type=random_type[0], architecture=architecture)
    network.to(device)
    criterion = nn.CrossEntropyLoss()
    network.train()

    print("\t \t", "=" * 80)
    print("\t \t After getting models")
    print("\t \t", "=" * 80)

    # setup will be COMP if we want to compare two models

    if setup == "COMP":
        losses_2 = list()
        accuracy_train_2 = list()
        accuracy_test_2 = list()
        network2 = PrimaryNetwork(num_classes, device, regularize_2, 1, dropout[1], random_type=random_type[1], architecture=architecture)
        network2.to(device)
        network2.train()

    losses = list()
    accuracy_train = list()
    accuracy_test = list()
    net_struct = list()

    net_struct.append(print_size_ratio(network, 3.0))
    for epoch in tqdm(range(train_epochs)):

        network.train()
        std = get_std(epoch, stds, std_epochs)
        l1_coeff = get_l1_coeff(epoch, coeffs, coeffs_epochs)
        optimizer = optimizers_epochs(network, epoch, lr[0], weight_decay[0], momentum[0])
        running_loss = 0.0
        between_loss = 0.0

        if setup == "COMP":
            running_loss2 = 0.0
            between_loss2 = 0.0
            optimizer2 = optimizers_epochs(network2, epoch, lr[1], weight_decay[1], momentum[1])
            network2.train()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, noise = network(inputs, std, means[0])
            loss_ce = criterion(outputs, labels)
            loss_l1 = (torch.sum(torch.abs(noise)) / noise.numel()) * l1_coeff
            loss = loss_l1 + loss_ce
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if setup == "COMP":
                optimizer2.zero_grad()
                outputs2, noise2 = network2(inputs, std, means[1])
                loss_ce_2 = criterion(outputs2, labels)
                loss_l1_2 = (torch.sum(torch.abs(noise2)) / noise2.numel()) * l1_coeff
                loss2 = loss_l1_2 + loss_ce_2
                loss2.backward()
                optimizer2.step()
                running_loss2 += loss2.item()

            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

            if i % 20 == 19:
                losses.append(running_loss - between_loss)
                net_struct.append(print_size_ratio(network, std))
                between_loss = running_loss
                if setup == "COMP":
                    losses_2.append(running_loss2 - between_loss2)
                    between_loss2 = running_loss2

        print_size_ratio(network, std, True)
        curr_accuracy_train= get_accuracy(network, trainloader, device, std)
        accuracy_train.append(get_accuracy(network, trainloader, device, std))
        print(f'Curr epoch {epoch} train accuracy: {curr_accuracy_train}')
        curr_accuracy_test = get_accuracy(network, testloader, device, std)
        accuracy_test.append(curr_accuracy_test)
        print(f'Curr epoch {epoch} test accuracy: {curr_accuracy_test}')
        if setup == "COMP":
            accuracy_train_2.append(get_accuracy(network2, trainloader, device, std))
            accuracy_test_2.append(get_accuracy(network2, testloader, device, std))

    outputs_dict = {"base_train": accuracy_train, 'base_test': accuracy_test, 'base_loss': losses, "structure": net_struct}
    if setup == "COMP":
        outputs_dict["second_train"] = accuracy_train_2
        outputs_dict["second_test"] = accuracy_test_2
        outputs_dict["second_loss"] = losses_2

    return outputs_dict, output_path, architecture



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Original training")
    parser.add_argument("--regularize", type=bool, nargs='*', help="How to regularize the network", default=
                        [True, False, True, False, True, False, True, False, True])
    parser.add_argument("--regularize_2", type=bool, nargs='*', help="How to regularize the network", default=
                        [False, False, False, False, False, False, False, False, False])
    parser.add_argument("--dataset_name", type=str, default='CIFAR100')
    parser.add_argument("--dropout", type=int, nargs='*', help="From which epoch begining dropout", default=[0., 0.])
    parser.add_argument('--transform_train', type=str, default='tensor_only', help= "Name of the transform for the training set")
    parser.add_argument("--transform_test", type=str, default='tensor_only', help="Name of the transform for the training set")
    parser.add_argument("--batch_size", type = int, default=16)
    parser.add_argument("--optimizer_name", type=str, default="AdamW", help="Which optimizer to use")
    parser.add_argument("--lr", type=float, default=[0.1, 0.1], help="Learning rate of the optimizer", nargs="*")
    parser.add_argument("--momentum", type=float, default=[0.9, 0.9], help="Momentum in case of SGD")
    parser.add_argument("--weight_decay", type=float, default=[5e-4, 5e-4], nargs='*', help="Weight_decay")
    parser.add_argument("--setup", type=str, default="SINGLE", help = "Set to COMP if you want to compare two models")
    parser.add_argument("--train_epochs", type=int,  default=50)
    parser.add_argument("--architecture", type = str, default="A", help="Which hypernet architecture to choose")
    parser.add_argument("--stds", type=float, nargs='*', default=[1.0, 2.0, 3.0], help="Which standard deviations to use")
    parser.add_argument("--std_epochs", type=int, nargs='*', default=[15, 30, 70], help="From which epoch to run it")
    parser.add_argument("--coeffs", type=float, nargs='*', default=[0.002, 0.001], help="Which standard deviations to use")
    parser.add_argument("--coeffs_epochs", type=int, nargs='*', default=[15, 30], help="From which epoch to run it")
    parser.add_argument("--random_type", type=str, nargs='*', default=["normal", "uniform"], help="What distribution to draw the random samples from")
    parser.add_argument("--means", type=str, nargs='*', default=[0, 0], help="What distribution to draw the random samples from")
    args = parser.parse_args()
    outputs_dict, output_path, architecture = train(**vars(args))
    end_of_training_stats(outputs_dict, output_path, architecture)
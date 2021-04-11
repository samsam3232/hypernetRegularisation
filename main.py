import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models.primaryNetwork import get_resnet
import argparse
from data.get_data import get_data_loaders
import os
from utils import get_optimizer, get_accuracy, plot_results, print_network_structure
from tqdm import tqdm
import math
import torch.optim as optim
import datetime
import matplotlib.pyplot as plt

# transform = transforms.ToTensor()
#
# trainset = torchvision.datasets.CIFAR10(root='/Users/samuelamouyal/PycharmProjects/hypernetRegularisation', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='/Users/samuelamouyal/PycharmProjects/hypernetRegularisation', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=16,
#                                          shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# network = PrimaryNetwork(10, 'cpu')
# optimizer = torch.optim.AdamW(network.parameters())
# criterion = nn.CrossEntropyLoss()
# network.train()
# for epoch in range(5):
#     running_loss = 0.0
#     for i, data in tqdm(enumerate(trainloader, 0)):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = network(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if i % 1000 == 999:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 1000))
#             running_loss = 0.0
def end_of_training_stats(outputs_dict, network_base, output_path):

    plot_results(outputs_dict, output_path)
    print("\t \t", "="*80)
    print("\t \t Outputting base network's structure")
    print("\t \t", "="*80)
    print_network_structure(network_base)


def transform_to_options(dropout = [0.0], dropout_hyper = [0.0], relu = [False], setup = "SINGLE"):


    options_base = {}
    if dropout[0] != 0:
        options_base['dropout'] = dropout
    if dropout_hyper[0] != 0:
        options_base['dropout_hype'] = dropout_hyper
    if relu[0]:
        options_base['relu'] = True

    if setup != 'SINGLE':
        options_second = {}
        if dropout[1] != 0:
            options_second['dropout'] = dropout
        if dropout_hyper[1] != 0:
            options_second['dropout_hype'] = dropout_hyper
        if relu[1]:
            options_second['relu'] = True

        return {'1': options_base, '2': options_second}

    return options_base


def get_models(size, type, device, regularize, num_classes, opts, setup = 'SINGLE', regularize_2 = None):

    network_second = None
    if '1' in opts:
        opts_base = opts['1']
    else:
        opts_base = opts
    network_base = get_resnet(size=size, type=type, num_classes=num_classes, device=device, regularize=regularize, options=opts_base)
    if setup == "COMPARE":
        opts_second = opts['2']
        network_second = get_resnet(size=size, type=type, num_classes=num_classes, device=device, regularize=regularize_2,
                                   options=opts_second)

    return network_base, network_second


def translate_regularize(regularize_delay, regularize_2_delay):

    regularize = torch.tensor(regularize_delay) == 0
    regularize_2 = torch.tensor(regularize_2_delay) == 0

    return regularize, regularize_2


def train(network_base, network_second, optimizer_base, optimizer_second, dropout_epoch, dropout_hyper_epoch,
          relu_epoch, regularize_delay, regularize_2_delay, train_epochs, trainloader, testloader, l1):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    losses_base = list()
    accuracy_train_base = list()
    accuracy_test_base = list()
    network_base.to(device)
    scheduler_base = optim.lr_scheduler.ReduceLROnPlateau(optimizer_base, mode="min", factor=0.2, patience=10)

    losses_second = list()
    accuracy_train_second = list()
    accuracy_test_second = list()

    if network_second is not None:
        network_second.to(device)
        scheduler_second = optim.lr_scheduler.ReduceLROnPlateau(optimizer_second, mode="min", factor=0.2, patience=10)

    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(train_epochs)):

        if dropout_epoch[0] == epoch:
            network_base.set_dropout(True)
        if dropout_hyper_epoch[0] == epoch:
            network_base.set_hyper_dropout(True)
        if relu_epoch[0] == epoch:
            network_base.set_hyper_relu(True)
        for i in range(len(regularize_delay)):
            if regularize_delay[i] == epoch:
                network_base.res_net[i].regularize()
        network_base.train()
        between_loss_base = 0
        running_loss_base = 0.0

        if network_second is not None:

            network_second.train()
            between_loss_second = 0
            running_loss_second = 0.0
            if dropout_epoch[1] == epoch:
                network_second.set_dropout(True)
            if dropout_hyper_epoch[1] == epoch:
                network_second.set_hyper_dropout(True)
            if relu_epoch[1] == epoch:
                network_second.set_hyper_relu(True)
            for i in range(len(regularize_2_delay)):
                if regularize_2_delay[i] == epoch:
                    network_second.res_net[i].regularize()

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_base.zero_grad()
            outputs, noise = network_base(inputs)
            if l1[0]:
                loss_ce = criterion(outputs, labels)
                loss_l1 = (torch.norm(noise, 1) * math.sqrt(math.sqrt(epoch)) / 300000)
                loss = loss_ce + loss_l1
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer_base.step()
            running_loss_base += loss.item()
            scheduler_base.step(loss.item())

            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss_base / 1000))
                running_loss_base = 0.0

            if i % 20 == 19:
                losses_base.append(running_loss_base - between_loss_base)
                between_loss_base = running_loss_base


            if network_second is not None:
                optimizer_second.zero_grad()
                outputs, noise = network_second(inputs)
                if l1[1]:
                    loss_ce = criterion(outputs, labels)
                    loss_l1 = (torch.norm(noise, 1) * math.sqrt(math.sqrt(epoch)) / 300000)
                    loss = loss_ce + loss_l1
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer_second.step()
                running_loss_second += loss.item()
                scheduler_second.step(loss.item())

                if i % 20 == 19:
                    losses_second.append(running_loss_second - between_loss_second)
                    between_loss_second = running_loss_second

        accuracy_train_base.append(get_accuracy(network_base, trainloader, device))
        accuracy_test_base.append(get_accuracy(network_base, testloader, device))

        if network_second is not None:
            accuracy_train_second.append(get_accuracy(network_second, trainloader, device))
            accuracy_test_second.append(get_accuracy(network_second, testloader, device))

    outputs_dict = {"base_train": accuracy_train_base, 'base_test': accuracy_test_base, 'base_loss': losses_base,
                    "second_train": accuracy_train_second, 'second_test': accuracy_test_second, 'second_loss': losses_second}

    return outputs_dict


def main(size = 18, dataset = "CIFAR10", dropout =  [0.0, 0.0], dropout_epoch = [-1, -1], dropout_hyper = [0.0,0.0],
         dropout_hyper_epoch = [-1, -1], relu = [True, False], relu_epoch = [-1, -1],
         regularize_delay = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], transform_train = 'tensor_only',
         transform_test = 'tensor_only', batch_size = 16,  optimizer_name = "AdamW", lr= [0.001, 0.001], momentum = [0.7, 0.7],
         weight_decay = [None, None], setup = "SINGLE", regularize_delay_2 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
         root = None, output_path = None, train_epochs = 50, l1 = [False, False]):

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, "Resnet{}_{}".format(size, dataset))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    t_now = datetime.datetime.now()
    output_path = os.path.join(output_path, "{}_{}_{}_{}".format(t_now.month, t_now.day, t_now.hour, t_now.minute))

    print("\t \t", "=" * 80)
    print("\t \t Before everything")
    print("\t \t", "=" * 80)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader, num_classes = get_data_loaders(dataset_name=dataset, transform_train_name=transform_train,
                                                            transform_test_name=transform_test, batch_size=batch_size,
                                                            root=root)

    print("\t \t", "=" * 80)
    print("\t \t After loading data. There are {} classes.".format(num_classes))
    print("\t \t", "=" * 80)

    type = 'CIFAR' if "CIFAR" in dataset else "REG"
    options = transform_to_options(dropout=dropout, dropout_hyper=dropout_hyper, relu=relu)

    print("\t \t", "=" * 80)
    print("\t \t After options transformations")
    print("\t \t", "=" * 80)

    regularize, regularize_2 = translate_regularize(regularize_delay=regularize_delay, regularize_2_delay=regularize_delay_2)

    print("\t \t", "=" * 80)
    print("\t \t Regularization translation")
    print("\t \t", "=" * 80)

    network_base, network_second = get_models(size=size, type=type, device= device, regularize=regularize, num_classes=num_classes,
                                              opts = options, setup=setup, regularize_2=regularize_2)

    print("\t \t", "=" * 80)
    print("\t \t After networks creation")
    print("\t \t", "=" * 80)

    optimizer_base = get_optimizer(network=network_base, optim_name=optimizer_name, lr=lr[0], momentum=momentum[0], weight_decay=weight_decay[0])
    optimizer_second = None
    if network_second is not None:
        optimizer_second = get_optimizer(network=network_second, optim_name=optimizer_name, lr=lr[1], momentum=momentum[1],
                                       weight_decay=weight_decay[1])

    print("\t \t", "=" * 80)
    print("\t \t After optimizer creation")
    print("\t \t", "=" * 80)

    ouputs = train(network_base, network_second, optimizer_base, optimizer_second, dropout_epoch, dropout_hyper_epoch,
                   relu_epoch, regularize_delay, regularize_delay_2, train_epochs, trainloader, testloader, l1)

    print("\t \t", "=" * 80)
    print("\t \t After training")
    print("\t \t", "=" * 80)

    end_of_training_stats(ouputs, network_base, output_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Regularization comparison")
    parser.add_argument("--size", type = int, help = "Size of the model we want to use (only resnet supported for now", default = 18)
    parser.add_argument("--dataset", type = str, help = "Name of the dataset you want to use", default="CIFAR10")
    parser.add_argument("--dropout", type=int, nargs='*', help="From which epoch begining dropout", default=[0., 0.])
    parser.add_argument("--dropout_hyper", type=int, nargs='*', help="From which epoch do dropout in hyper", default=[0., 0.])
    parser.add_argument("--dropout_epoch", type=int, nargs='*', help="From which epoch begining dropout", default=[-1., -1.])
    parser.add_argument("--dropout_hyper_epoch", type=int, nargs='*', help="From which epoch do dropout in hyper", default=[-1., -1.])
    parser.add_argument("--relu", type=int, nargs='*', help="From which epoch to add layer relu_on top", default=[True, False])
    parser.add_argument("--relu_epoch", type=int, nargs='*', help="From which epoch to add layer relu_on top", default=[-1, -1])
    parser.add_argument("--regularize_delay", type=int, help="When do you want to begin to regularize the layer", nargs='*',
                        default=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    parser.add_argument('--transform_train', type=str, default='tensor_only', help= "Name of the transform for the training set")
    parser.add_argument("--transform_test", type=str, default='tensor_only', help="Name of the transform for the training set")
    parser.add_argument("--batch_size", type = int, default=16)
    parser.add_argument("--optimizer_name", type=str, default="AdamW", help="Which optimizer to use")
    parser.add_argument("--lr", type=float, default=[0.001, 0.001], help="Learning rate of the optimizer", nargs="*")
    parser.add_argument("--momentum", type=float, default=[0.7, 0.7], help="Momentum in case of SGD")
    parser.add_argument("--weight_decay", type=float, default=[None, None], nargs='*', help="Weight_decay")
    parser.add_argument('--setup', type=str, help="COMPARE if you want to compare", default="SINGLE")
    parser.add_argument("--regularize_delay_2", type=int, help="When do you want to begin to regularize the layer in second model", nargs='*',
                        default=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    parser.add_argument("--root", type=str, help="Path to where you want to keep your data")
    parser.add_argument("--output_path", type=str, help="Path to where to output the data", default=os.path.join(os.getcwd(), 'outputs'))
    parser.add_argument("--train_epochs", type=int, default=50, help="Num of training epochs")
    args = parser.parse_args()
    main(**vars(args))
import argparse
import numpy as np
import os
import torch
import datetime
from utils import get_optimizer, get_accuracy, plot_results, print_network_structure
from data.get_data import get_data_loaders
from models.orig_primary import PrimaryNetwork
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn

def end_of_training_stats(outputs_dict, output_path):

    plot_results(outputs_dict, output_path)


def train(regularize, dataset_name, dropout, transform_train, transform_test, batch_size, optimizer_name, lr, momentum,
          weight_decay, train_epochs, root=None):

    device  = "cuda" if torch.cuda.is_available() else 'cpu'
    regularize=np.array(regularize).reshape((3, 3))
    output_path = os.getcwd()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, "Resnet18_{}".format(dataset_name))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    t_now = datetime.datetime.now()
    output_path = os.path.join(output_path, "{}_{}_{}_{}".format(t_now.month, t_now.day, t_now.hour, t_now.minute))

    print("\t \t", "=" * 80)
    print("\t \t Before everything")
    print("\t \t", "=" * 80)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader, num_classes = get_data_loaders(dataset_name=dataset_name, transform_train_name=transform_train,
                                                            transform_test_name=transform_test, batch_size=batch_size,
                                                            root=root)

    losses = list()
    accuracy_train = list()
    accuracy_test = list()
    network = PrimaryNetwork(num_classes, device, regularize, 1, dropout[0])

    network.to(device)

    optimizer = get_optimizer(network=network, optim_name=optimizer_name, lr=lr[0], momentum=momentum[0], weight_decay=weight_decay[0])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)
    network.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(train_epochs)):
        network.train()
        running_loss = 0.0
        between_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, noise = network(inputs)
            loss = criterion(outputs, labels)
            #        loss_l1 = torch.norm(noise, 1)* math.sqrt(math.sqrt(epoch)) / 300000
            #        loss = loss_l1 + loss_ce
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 1000 == 999:
                scheduler.step(running_loss)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

            if i % 20 == 19:
                losses.append(running_loss - between_loss)
                between_loss = running_loss

        accuracy_train.append(get_accuracy(network, trainloader, device))
        accuracy_test.append(get_accuracy(network, testloader, device))

    outputs_dict = {"base_train": accuracy_train, 'base_test': accuracy_test, 'base_loss': losses}

    return outputs_dict, output_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Original training")
    parser.add_argument("--regularize", type=  bool, nargs='*', help="How to regularize the network", default=
                        [True, True, False, False, False, False, True, True, True])
    parser.add_argument("--dataset_name", type=str, default='CIFAR100')
    parser.add_argument("--dropout", type=int, nargs='*', help="From which epoch begining dropout", default=[0., 0.])
    parser.add_argument('--transform_train', type=str, default='tensor_only', help= "Name of the transform for the training set")
    parser.add_argument("--transform_test", type=str, default='tensor_only', help="Name of the transform for the training set")
    parser.add_argument("--batch_size", type = int, default=16)
    parser.add_argument("--optimizer_name", type=str, default="AdamW", help="Which optimizer to use")
    parser.add_argument("--lr", type=float, default=[0.001, 0.001], help="Learning rate of the optimizer", nargs="*")
    parser.add_argument("--momentum", type=float, default=[0.7, 0.7], help="Momentum in case of SGD")
    parser.add_argument("--weight_decay", type=float, default=[None, None], nargs='*', help="Weight_decay")
    parser.add_argument("--train_epochs", type=int,  default=50)
    args = parser.parse_args()
    output_dict, output_path = train(**vars(args))
    end_of_training_stats(outputs_dict, output_path)
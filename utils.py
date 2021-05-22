import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np

OPTIM_MAPPING = {"SGD": optim.SGD, "AdamW": optim.AdamW}


def get_optimizer(network, optim_name, lr, momentum=0.7, weight_decay=None):
    if optim_name == "SGD":
        if weight_decay is not None:
            return OPTIM_MAPPING["SGD"](network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        return OPTIM_MAPPING["SGD"](network.parameters(), lr=lr, momentum=momentum)

    if weight_decay is not None:
        return OPTIM_MAPPING[optim_name](network.parameters(), lr=lr, weight_decay=weight_decay)
    return OPTIM_MAPPING[optim_name](network.parameters(), lr=lr)


def get_accuracy(network, loader, device, std):
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, noise = network(inputs, std)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def plot_results(output_dicts, outpath):
    plt.figure()
    plt.plot(output_dicts['base_train'], color='r', label='ours')
    if 'second_train' in output_dicts:
        plt.plot(output_dicts['second_train'], color='b', alpha=0.5, label="baseline")
    plt.ylabel('train_accuracy')
    plt.xlabel('epoch number')
    plt.title("Train accuracy according to epoch")
    plt.legend()
    plt.savefig(outpath + '/train_accuracy.png')
    plt.show()

    plt.figure()
    plt.plot(output_dicts['base_test'], color='r', label='ours')
    if 'second_test' in output_dicts:
        plt.plot(output_dicts['second_test'], color='b', alpha=0.5, label="baseline")
    plt.ylabel('test_accuracy')
    plt.xlabel('epoch number')
    plt.legend()
    plt.title("Test accuracy according to epoch")
    plt.savefig(outpath + '/test_accuracy.png')
    plt.show()

    base_losses = output_dicts['base_loss']
    for i in base_losses:
        if i < 0:
            base_losses.remove(i)
    if 'second_loss' in output_dicts:
        second_losses = output_dicts['second_loss']
        for i in second_losses:
            if i < 0:
                second_losses.remove(i)
    plt.figure()
    plt.plot(base_losses, color='r', label="ours")
    if 'second_loss' in output_dicts:
        plt.plot(second_losses, color='b', alpha=0.5, label="baseline")
    plt.legend()
    plt.ylabel('Loss')
    plt.title("Loss every 50 batches")
    plt.savefig(outpath + '/loss.png')
    plt.show()

    plt.figure()
    plt.plot(np.array(output_dicts['base_train']) - np.array(output_dicts['base_test']), color='r', label="ours")
    if 'second_train' in output_dicts:
        plt.plot(np.array(output_dicts['second_train']) - np.array(output_dicts['second_test']), color='b', alpha=0.5,
                 label="baseline")
    plt.legend()
    plt.xlabel('epoch number')
    plt.ylabel('Accuracies differences')
    plt.title("Accuracies differences")
    plt.savefig(outpath + '/diff_accuracies.png')
    plt.show()

    plt.figure()
    plt.plot(np.array(output_dicts['structure']), color='r', label="ours")
    plt.legend()
    plt.xlabel('epoch number')
    plt.ylabel('Non_zero proportion')
    plt.title("Regularized_struct")
    plt.savefig(outpath + '/regularized_struct.png')
    plt.show()


def print_network_structure(network):
    diff_proportions = list()
    for i in range(5):
        non_zero = 0
        noise = network.hyper_net()
        curr = 0
        for index in range(len(network.mod_sizes)):
            if network.regularize[np.ceil(index / 2)]:
                non_zero += np.count_nonzero(noise[curr: curr + network.mod_sizes[index]])
            else:
                if index == 0:
                    non_zero += np.count_nonzero(network.conv1.weights)
                elif index != (len(network.mod_sizes) - 1):
                    if index % 2 == 1:
                        non_zero += np.count_nonzero(network.res_net[int(index / 2)].conv1.weight)
                    else:
                        non_zero += np.count_nonzero(network.res_net[int(index / 2)].conv2.weight)
                else:
                    non_zero += np.count_nonzero(network.final.weight)
            curr += network.mod_sizes[index]

        print("\t \t Current proportion of zeros for batch {}: {}".format(i + 1, (
                    1 - (non_zero / torch.sum(network.mod_sizes)))))

        diff_proportions.append((1 - (non_zero / torch.sum(network.mod_sizes))))

    print("\t \t", '=' * 80)
    print("Final average proportion: {}".format(np.sum(diff_proportions) / 5))
    print("\t \t", "=" * 80)


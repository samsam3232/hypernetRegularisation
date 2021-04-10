import torch
import torchvision.transforms as transforms
import data.utils as utils
from torch.utils.data import DataLoader
import os

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

TRANSFORM_MAPPING = {'tensor_only': transforms.ToTensor(), 'standard': transforms.Compose([transforms.ToTensor(), NORMALIZE]),
                     'augs_cifar': transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),
                                                        transforms.ToTensor(), NORMALIZE])}


def get_data_loaders(dataset_name, transform_train_name, transform_test_name, batch_size, root = None):

    transform_train, transform_test = TRANSFORM_MAPPING[transform_train_name], TRANSFORM_MAPPING[transform_test_name]

    if root is None:
        current_dir = os.getcwd()
        root = os.path.join(current_dir, 'data', dataset_name)
    trainset, testset, num_classes = utils.get_dataset(dataset_name, transform_train, transform_test, root)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return trainloader, testloader, num_classes
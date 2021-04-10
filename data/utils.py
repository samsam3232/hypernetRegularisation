import torchvision
import torchvision.transforms as transforms

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

DATASET_MAPPING = {'CIFAR10' : torchvision.datasets.CIFAR10, 'CIFAR100': torchvision.datasets.CIFAR100,
                   'LSUN': torchvision.datasets.LSUN, 'IMAGENET': torchvision.datasets.imagenet}

TRANSFORM_MAPPING = {'tensor_only': transforms.ToTensor(), 'standard': transforms.Compose([transforms.ToTensor(), NORMALIZE]),
                     'augs_cifar': transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),
                                                        transforms.ToTensor(), NORMALIZE])}

def get_dataset(dataset_name, transform_train, transform_test, root):

    dataset_train = DATASET_MAPPING[dataset_name](root=root, train=True, download=True, transform=transform_train)
    dataset_test = DATASET_MAPPING[dataset_name](root=root, train=False, download=True, transform=transform_test)

    return dataset_train, dataset_test, len(dataset_train.classes)
"""Train and Test data set downloader and allows access through PyTorch's Dataloader"""

import torch
import CONSTANTS as const
from torchvision import datasets, transforms
from utility import get_config_details, check_gpu_availability

config = get_config_details()


def define_train_test_transformers():
    # Train data transformation

    train_transforms = transforms.Compose([transforms.RandomRotation((-9.0, 9.0), fill=(1,)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                                           ])

    # Test transform

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    return train_transforms, test_transforms


def download_data(*, train_transforms, test_transforms):
    """Downloads and returns train test MNIST data after the mandatory train and test transforms respectively"""

    if not (isinstance(train_transforms, transforms.Compose) and isinstance(test_transforms, transforms.Compose)):
        raise Exception("\n The train and test transformers passed are invalid.")

    train = datasets.MNIST(root='../data', train=True, download=True, transform=train_transforms)  # Train data

    test = datasets.MNIST(root='../data', train=False, download=True, transform=test_transforms)  # Test data

    return train, test


dataloader_args = dict(shuffle=bool(config[const.MODEL_CONFIG][const.SHUFFLE]),
                       batch_size=int(config[const.MODEL_CONFIG][const.BATCH_SIZE]),
                       num_workers=int(config[const.MODEL_CONFIG][const.WORKERS]),
                       pin_memory=bool(config[const.MODEL_CONFIG][const.PIN_MEMORY])) if check_gpu_availability() \
    else dict(shuffle=bool(config[const.MODEL_CONFIG][const.SHUFFLE]),
              batch_size=int(config[const.MODEL_CONFIG][const.BATCH_SIZE]))


def get_train_test_dataloaders(*, train_data, test_data, data_loader_args):
    """Generates and returns data loaders for train and test data sets"""

    if not (isinstance(train_data, datasets.MNIST) and isinstance(test_data, datasets.MNIST)):
        raise Exception("\n Wrong data type passed >> Expected MNIST dataset")

    '''train dataloader'''
    train_loader = torch.utils.data.DataLoader(dataset=train_data, **data_loader_args)

    '''test dataloader'''
    test_loader = torch.utils.data.DataLoader(dataset=test_data, **data_loader_args)

    return train_loader, test_loader

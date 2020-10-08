"""Train and Test data set downloader and allows access through PyTorch's Dataloader"""

import torch
from wrapper import CONSTANTS as const
import albumentations as A
from torchtoolbox.transform import Cutout
# from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms
from wrapper.utility import get_config_details, check_gpu_availability, AlbumentationTransforms

config = get_config_details()

cifar10 = ["s7", "s8"]


def define_train_test_transformers(dataset_name, use_album_library, cutout):
    if not isinstance(dataset_name, str):
        raise TypeError("\n Data set name must be a string")

    if "mnist" in dataset_name.lower():

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

    elif "cifar10" in dataset_name.lower():
        mean = tuple([125.30691805 / 255, 122.95039414 / 255, 113.86538318 / 255])
        standard_deviation = tuple([62.99321928 / 255, 62.08870764 / 255, 66.70489964 / 255])

        if not use_album_library:
            if cutout:
                # Train Phase transformations
                train_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=32, padding=1),
                    Cutout(p=0.25, scale=(0.02, 0.10)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=standard_deviation)

                ])
            else:
                train_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=32, padding=1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=standard_deviation)

                ])
        elif use_album_library:
            if cutout:
                train_transforms = AlbumentationTransforms([A.HorizontalFlip(p=0.25),
                                                            A.OneOf([
                                                                A.IAAAdditiveGaussianNoise(),
                                                                A.GaussNoise(),
                                                            ], p=0.2),
                                                            A.OneOf([
                                                                A.MotionBlur(p=0.2),
                                                                A.MedianBlur(blur_limit=3, p=0.1),
                                                                A.Blur(blur_limit=3, p=0.1),
                                                            ], p=0.2),
                                                            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
                                                                               rotate_limit=35, p=0.2),
                                                            # Cutout is deprecated in latest Albumentations library
                                                            # A.Cutout(num_holes=2, max_h_size=8, max_w_size=8, fill_value=list(mean), p=0.3),
                                                            A.CoarseDropout(max_holes=2, min_holes=1, max_height=12,
                                                                            max_width=12, min_height=6,
                                                                            min_width=6, fill_value=list(mean)),
                                                            A.RandomBrightnessContrast(p=0.15),
                                                            A.HueSaturationValue(p=0.21),
                                                            A.Normalize(mean=mean, std=standard_deviation)])
            else:
                train_transforms = AlbumentationTransforms([A.HorizontalFlip(p=0.25),
                                                            A.OneOf([
                                                                A.IAAAdditiveGaussianNoise(),
                                                                A.GaussNoise(),
                                                            ], p=0.2),
                                                            A.OneOf([
                                                                A.MotionBlur(p=0.2),
                                                                A.MedianBlur(blur_limit=3, p=0.1),
                                                                A.Blur(blur_limit=3, p=0.1),
                                                            ], p=0.2),
                                                            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
                                                                               rotate_limit=35, p=0.2),
                                                            A.RandomBrightnessContrast(p=0.15),
                                                            A.HueSaturationValue(p=0.21),
                                                            A.Normalize(mean=mean, std=standard_deviation)])

        test_transforms = AlbumentationTransforms([A.Normalize(mean=mean, std=standard_deviation)])

    else:
        train_transforms = transforms.Compose([transforms.ToTensor()])
        test_transforms = transforms.Compose([transforms.ToTensor()])

    return train_transforms, test_transforms


def download_data(*, dataset_name, train_transforms, test_transforms):
    """Downloads and returns train test dataset after the mandatory train and test transforms respectively"""

    if not ((isinstance(train_transforms, transforms.Compose) and isinstance(test_transforms,
                                                                             transforms.Compose)) or isinstance(
        dataset_name, str)):
        raise Exception("\n The train and test transformers passed are invalid. Or the data set is not valid string")

    if dataset_name:
        if "mnist" in dataset_name.lower():

            train = datasets.MNIST(root='../data', train=True, download=True, transform=train_transforms)  # Train data

            test = datasets.MNIST(root='../data', train=False, download=True, transform=test_transforms)  # Test data

        elif "cifar10" in dataset_name.lower():
            train = datasets.CIFAR10(root='../data', train=True,
                                     download=True, transform=train_transforms)

            test = datasets.CIFAR10(root='../data', train=False,
                                    download=True, transform=test_transforms)

    return train, test


dataloader_args = dict(shuffle=bool(config[const.MODEL_CONFIG][const.SHUFFLE]),
                       batch_size=int(config[const.MODEL_CONFIG][const.BATCH_SIZE]),
                       num_workers=int(config[const.MODEL_CONFIG][const.WORKERS]),
                       pin_memory=bool(config[const.MODEL_CONFIG][const.PIN_MEMORY])) if check_gpu_availability() \
    else dict(shuffle=bool(config[const.MODEL_CONFIG][const.SHUFFLE]),
              batch_size=int(config[const.MODEL_CONFIG][const.BATCH_SIZE]))


def get_train_test_dataloaders(*, train_data, test_data, data_loader_args):
    """Generates and returns data loaders for train and test data sets"""

    '''train dataloader'''
    train_loader = torch.utils.data.DataLoader(dataset=train_data, **data_loader_args)

    '''test dataloader'''
    test_loader = torch.utils.data.DataLoader(dataset=test_data, **data_loader_args)

    return train_loader, test_loader

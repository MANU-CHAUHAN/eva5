from wrapper import CONSTANTS
import os
import torch
import collections
import configparser
from torch import nn
from copy import deepcopy
import matplotlib.pyplot as plt
import torch.nn.functional as f
from torchsummary import summary
import copy
from torchvision.utils import make_grid
import PIL
from torchvision import transforms
import sys
from wrapper import show_images, GradCAM
import albumentations as A
import albumentations.pytorch as AP
import numpy as np

config = configparser.RawConfigParser()
current_path = os.getcwd()

if not os.path.isfile(os.path.join(current_path, "config.cfg")):
    raise FileNotFoundError('\n Configuration file "config.cfg" not found in the root directory of the project.')

config.read(os.path.join(current_path, "config.cfg"))


def get_config_details():
    """
    Reads the configuration file and sets the attribute for this function the first time so subsequent calls avoid re-reading config file

    :return: 2 level dict, where outer key is section name and outer value is a dict,
             inner key is parameter name corresponding to outer key and value is the corresponding value for inner key.
    """
    if not hasattr(get_config_details, 'config_dict'):
        get_config_details.config_dict = collections.defaultdict(dict)

        for section in config.sections():
            get_config_details.config_dict[section] = dict(config.items(section))

    return get_config_details.config_dict


def get_combos_and_trackers():
    """Makes a tracker for every combination given in config file.
    :returns a tuple of list of all combinations in string and tracker dict with key as combination name and value as
            dictionaries with 'misclassified', 'train_losses','test_losses','train_accuracy', 'test_accuracy'
            for each combo key"""

    d = {
        'misclassified': [],
        'train_losses': [],
        'test_losses': [],
        'train_accuracy': [],
        'test_accuracy': []
    }

    all_combo_list = get_config_details()[CONSTANTS.MODEL_CONFIG][CONSTANTS.COMBOS].split(',')
    tracker = {}
    for item in all_combo_list:
        tracker[item] = deepcopy(d)
    del d
    return all_combo_list, tracker


def check_gpu_availability(seed=101):
    """
    Checks if a GPU is available. Uses :param seed to set seed value with torch if GPU is available.
    :param seed: the seeding value
    :return: cuda flag, type: bool
    """
    cuda = torch.cuda.is_available()
    if cuda:
        print('\n CUDA is available')
        torch.cuda.manual_seed(seed)
    else:
        print("\n No GPU")
    return cuda


def get_device():
    """ :returns the device type available"""
    return 'cuda' if check_gpu_availability() else 'cpu'


class GhostBatchNorm(nn.BatchNorm2d):
    """ from https://github.com/apple/ml-cifar-10-faster/blob/master/utils.py
        Implements Ghost Batch Normalization"""

    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return f.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return f.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


def print_summary(*, model: torch.nn.Module, input_size: tuple):
    """Will utilize torchsummary to print summary of the model"""
    if not (isinstance(model, torch.nn.Module) and (isinstance(input_size, tuple) and len(input_size) == 3)):
        raise Exception("\nCheck the model passed or the input size(must be a tuple of c, h, w)")

    print(summary(model=model, input_size=input_size))


def get_optimizer(*, model):
    """
    Checks for the given Optimizer type, learning rate, momentum and weight decay in configuration and returns the optim
    :param model: the model for which the optimizer will be used
    :returns the optimizer after going through the configuration file for essential parameters for the given optimizer to use
    """

    lr = 0.01
    momentum = wd = 0.0
    nesterov = False
    optim_dict = config[CONSTANTS.OPTIMIZER]
    regul_dict = config[CONSTANTS.REGULARIZATION]

    if CONSTANTS.LR in optim_dict.keys():
        lr = float(optim_dict[CONSTANTS.LR])
    if CONSTANTS.L2 in regul_dict.keys():
        wd = float(regul_dict[CONSTANTS.L2])
    if CONSTANTS.MOMENTUM in optim_dict.keys():
        momentum = float(optim_dict[CONSTANTS.MOMENTUM])
    if CONSTANTS.NESTEROV in optim_dict.keys():
        nesterov = True

    if CONSTANTS.SGD in optim_dict[CONSTANTS.OPTIM_TYPE].split()[0].lower():
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)

    if CONSTANTS.ADAM in optim_dict[CONSTANTS.OPTIM_TYPE].split()[0].lower():
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def get_scheduler(*, optimizer):
    """
    Gets the scheduler type and other parameters from config and returns the corresponding scheduler
    :param optimizer: the optimizer on which scheduler will run
    :return: scheduler, type either torch.optim.lr_scheduler or None in case no values are given in config
    """
    schdlr_dict = config[CONSTANTS.SCHEDULER]
    step, gamma = 5, 0.001
    if len(schdlr_dict.keys()) > 0:
        if CONSTANTS.SCHEDULER_TYPE in schdlr_dict.keys():
            scheduler = schdlr_dict[CONSTANTS.SCHEDULER_TYPE]

        if CONSTANTS.STEP in schdlr_dict.keys():
            step = int(schdlr_dict[CONSTANTS.STEP])

        if CONSTANTS.GAMMA in schdlr_dict.keys():
            gamma = float(schdlr_dict[CONSTANTS.GAMMA])

        if CONSTANTS.STEP_LR in scheduler.lower():
            return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step, gamma=gamma)

        if scheduler.lower() == CONSTANTS.MULTI_STEP_LR and CONSTANTS.MILESTONES in schdlr_dict.keys():
            return torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=list(map(int, schdlr_dict[
                                                            CONSTANTS.MILESTONES].strip().split(','))),
                                                        gamma=gamma)

    else:
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1)


def plot(*, title, x_label, y_label, tracker, category):
    """
    Plots and saves the plot with given params. Uses tracker for getting the data and category is for the target.
    :param title: string describing the title of the plot
    :param x_label: str, xlabel
    :param y_label: str, ylabel
    :param tracker: dict, containing data, where outer key is the type and inner key is the category and inner v holds the values
    :param category: str, one of the inner keys for tracker
    :return: None
    """

    for type_, d in tracker.items():
        for k, v in d.items():
            if k.lower() == category:
                x = [*range(len(v))]
                plt.plot(x, v, label=type_)
                break

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(title + ".png", bbox_inches='tight')


def plot_misclassified(*, tracker, main_category, color_map, interpolate):
    if main_category:
        for type_, d in tracker.items():
            for k, v in d.items():
                if type_.lower() == main_category and k.lower() == CONSTANTS.MISCLASSIFIED:
                    fig = plt.figure(figsize=(10, 10))
                    for i in range(25):
                        sub = fig.add_subplot(5, 5, i + 1)
                        plt.imshow(v[i][0].cpu().numpy().squeeze(), cmap=color_map, interpolation=interpolate)
                        sub.set_title(
                            "Pred={}, Act={}".format(str(v[i][1].data.cpu().numpy()[0]),
                                                     str(v[i][2].data.cpu().numpy())))
                    plt.tight_layout()
                    plt.show()
                    plt.savefig(type_ + main_category + "_misclassified.png")
    else:
        for type_, d in tracker.items():
            for k, v in d.items():
                if k.lower() == CONSTANTS.MISCLASSIFIED:
                    fig = plt.figure(figsize=(10, 10))
                    for i in range(25):
                        sub = fig.add_subplot(5, 5, i + 1)
                        plt.imshow(v[i][0].cpu().numpy().squeeze(), cmap=color_map, interpolation=interpolate)
                        sub.set_title(
                            "Pred={}, Act={}".format(str(v[i][1].data.cpu().numpy()[0]),
                                                     str(v[i][2].data.cpu().numpy())))
                    plt.tight_layout()
                    plt.show()
                    plt.savefig(type_ + "_misclassified.png")


def get_dataloader_args():
    dataloader_args = dict(shuffle=bool(config[CONSTANTS.MODEL_CONFIG][CONSTANTS.SHUFFLE]),
                           batch_size=int(config[CONSTANTS.MODEL_CONFIG][CONSTANTS.BATCH_SIZE]),
                           num_workers=int(config[CONSTANTS.MODEL_CONFIG][CONSTANTS.WORKERS]),
                           pin_memory=bool(
                               config[CONSTANTS.MODEL_CONFIG][CONSTANTS.PIN_MEMORY])) if check_gpu_availability() \
        else dict(shuffle=bool(config[CONSTANTS.MODEL_CONFIG][CONSTANTS.SHUFFLE]),
                  batch_size=int(config[CONSTANTS.MODEL_CONFIG][CONSTANTS.BATCH_SIZE]))
    return dataloader_args


def get_all_models_summary():
    """ Collects all models from file `models.py` and uses torchsummary to print the summary"""
    import inspect
    from models import model_dummy
    for i in [m[0] for m in inspect.getmembers(model_dummy, inspect.isclass) if 'Net' in m[0]]:
        print(f'\nModel name: {i}')
        print_summary(model=model_dummy.str_to_class(i)().to(get_device()), input_size=(1, 28, 28))


def show_model_summary(title=None, *, model, input_size):
    """
    Calls `summary` method from torchsummary for the passed model and input size
    :param: title: title to show before printing summary
    :param model: the model to show the summary, detailed layers and parameters
    :param input_size: the input data size
    """
    if title:
        print(title)
    print(summary(model=model, input_size=input_size))


def get_dataset_name(*, session):
    return {"s6": "mnist", "s7": "cifar10", "s8": "cifar10"}.get(session.lower(), "mnist")


def get_input_size(*, dataset):
    return {"mnist": (1, 28, 28),
            "cifar10": (3, 32, 32)}.get(dataset.lower(), "mnist")


def get_accuracy_by_class_type(*, model, device, classes, test_loader):
    correct_dict = collections.defaultdict(int)
    total_dict = collections.defaultdict(int)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).squeeze()
            for i in range(len(correct)):
                label = labels[i]
                correct_dict[label] += correct[i].item()
                total_dict[label] += 1

    print("\n\n Class level accuracies")
    for class_ in classes:
        print(f'Accuracy for class: {class_}  =  {(100 * correct_dict[class_] / total_dict[class_]):0.3f}')


def plot_curve(curves, title, Figsize=(7, 7)):
    fig = plt.figure(figsize=Figsize)
    ax = plt.subplot()
    for curve in curves:
        ax.plot(curve[0], label=curve[1])
        plt.title(title)
    ax.legend()
    plt.show()


# images = [sys.path[-1] + '/images/cat.jpeg', sys.path[-1] + '/images/dog.jpeg', sys.path[-1] + '/images/bird.jpeg']


def grad_cam_showtime(images, device, model, classes, layers, size=(32, 32), display_figsize=(10, 10)):
    pil_image = []
    for i, img in enumerate(images):
        pil_image.append(PIL.Image.open(img))

    normed_torch_img = []
    torch_img_list = []

    for i in pil_image:
        torch_img = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])(i).to(device)
        torch_img_list.append(torch_img)
        normed_torch_img.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(torch_img)[None])
    for i, k in enumerate(normed_torch_img):
        images1 = [torch_img_list[i].cpu()]
        images2 = [torch_img_list[i].cpu()]
        b = copy.deepcopy(model.to(device))
        output = model(normed_torch_img[i])
        _, predicted = torch.max(output.data, 1)
        for j in layers:
            g = GradCAM.GradCAM(b, j)
            mask, _ = g(normed_torch_img[i])
            heatmap, result = GradCAM.visualize_cam(mask, torch_img_list[i])
            images1.extend([heatmap])
            images2.extend([result])
        grid_image = make_grid(images1 + images2, nrow=5)
        show_images.show_image(grid_image, title=classes[int(predicted)], figsize=display_figsize)


class AlbumentationTransforms:
    """
    Helper class to create test and train transforms using Albumentations
    """

    def __init__(self, transforms_list=[]):
        transforms_list.append(AP.ToTensor())

        self.transforms = A.Compose(transforms_list)

    def __call__(self, img):
        img = np.array(img)
        return self.transforms(image=img)['image']

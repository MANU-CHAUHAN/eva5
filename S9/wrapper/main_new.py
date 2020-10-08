import time
import traceback
import torch.nn as nn
from wrapper import utility, CONSTANTS, show_images, train_test, train_test_dataloader
from models import basic_mnist, cifar10_groups_dws_s7_model, ResNet


def run_model_run(*, dataset, model, cutout=False, use_albumentation=False):
    try:
        if not (dataset and model) or not (isinstance(dataset, str) and isinstance(model, str)):
            raise ValueError("\n Don't you think you should be providing the program some dataset and model names")

        train_transforms, test_transforms = train_test_dataloader.define_train_test_transformers(dataset_name=dataset,
                                                                                                 use_album_library=use_albumentation,
                                                                                                 cutout=cutout)
        train_data, test_data = train_test_dataloader.download_data(
            dataset_name=dataset,
            train_transforms=train_transforms,
            test_transforms=test_transforms)

        train_loader, test_loader = train_test_dataloader.get_train_test_dataloaders(train_data=train_data,
                                                                                     test_data=test_data,
                                                                                     data_loader_args=utility.get_dataloader_args())

        all_regularizations_list, tracker = utility.get_combos_and_trackers()
        device = utility.get_device()
        # utility.get_all_models_summary()
        loss_fn = nn.functional.nll_loss

        if dataset.lower() == "mnist":
            if CONSTANTS.GBN in utility.get_config_details()[CONSTANTS.REGULARIZATION].keys():
                model = basic_mnist.GBNNet().to(device)
            else:
                model = basic_mnist.S6_MNIST().to(device)

        elif "s7" in model.lower() and dataset.lower() == "cifar10":
            model = cifar10_groups_dws_s7_model.S7_CIFAR10()
            model = model.to(device)
            loss_fn = nn.CrossEntropyLoss()

        elif ("resnet18" in model.lower() or "s8" in model.lower()) and dataset.lower() == "cifar10":
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            show_images.show_training_data(dataset=train_loader, classes=classes)
            time.sleep(5)
            model = ResNet.ResNet18()
            model = model.to(device)
            loss_fn = nn.CrossEntropyLoss()

        optimizer = utility.get_optimizer(model=model)
        scheduler = utility.get_scheduler(optimizer=optimizer)
        utility.show_model_summary(title=model.__doc__, model=model, input_size=utility.get_input_size(
            dataset=dataset))

        train_test.train_test(model=model, device=device, train_loader=train_loader, optimizer=optimizer,
                              epochs=int(utility.get_config_details()[CONSTANTS.MODEL_CONFIG][CONSTANTS.EPOCHS]),
                              scheduler=scheduler, test=True, test_loader=test_loader, tracker=tracker, loss_fn=loss_fn)

        # for plot_type in utility.get_config_details()[CONSTANTS.PLOTS][CONSTANTS.TO_PLOT].strip().split(','):
        #     utility.plot(title="Plot is for:" + plot_type, x_label='Epochs', y_label=plot_type.lower(),
        #                  tracker=tracker, category=plot_type)
        return tracker, model, test_loader

    except Exception as e:
        print(traceback.format_exc(e))


if __name__ == '__main__':
    metrics, model, test_data = run_model_run(model="resnet18", dataset="cifar10", cutout=True, use_albumentation=True)
    utility.get_accuracy_by_class_type(model=model, device=utility.get_device(), classes=(
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'), test_loader=test_data)

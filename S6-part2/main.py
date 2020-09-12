import CONSTANTS, model, train_test, train_test_dataloader, utility
from model import Net, GBNNet


def run_model_run():
    train_transforms, test_transforms = train_test_dataloader.define_train_test_transformers()
    train_data, test_data = train_test_dataloader.download_data(train_transforms=train_transforms,
                                                                test_transforms=test_transforms)

    train_loader, test_loader = train_test_dataloader.get_train_test_dataloaders(train_data=train_data,
                                                                                 test_data=test_data,
                                                                                 data_loader_args=utility.dataloader_args)

    all_regularizations_list, tracker = utility.get_combos_and_trackers()
    device = utility.get_device()
    # utility.get_all_models_summary()

    for combo in all_regularizations_list:
        print("\nRunning for: ", combo)

        if CONSTANTS.GBN in combo.lower():
            model = GBNNet().to(device)
        else:
            model = Net().to(device)

        optimizer = utility.get_optimizer(model=model)
        scheduler = utility.get_scheduler(optimizer=optimizer)

        train_test.train(model=model, device=device, train_loader=train_loader, optimizer=optimizer,
                         epochs=int(utility.get_config_details()[CONSTANTS.MODEL_CONFIG][CONSTANTS.EPOCHS]),
                         scheduler=scheduler,
                         test=True, test_loader=test_loader, type_=combo, tracker=tracker,
                         l1_lambda=float(utility.get_config_details()[CONSTANTS.REGULARIZATION][CONSTANTS.L1]),
                         l2_lambda=float(utility.get_config_details()[CONSTANTS.REGULARIZATION][CONSTANTS.L2]))

    for plot_type in utility.get_config_details()[CONSTANTS.PLOTS][CONSTANTS.TO_PLOT].strip().split(','):
        utility.plot(title="Plot is for:" + plot_type, x_label='Epochs', y_label=plot_type.lower(),
                     tracker=tracker, category=plot_type)


if __name__ == '__main__':
    run_model_run()

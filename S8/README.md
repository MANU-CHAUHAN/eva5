# S8 EVA5

The session was about deeper look under the hood for fundamental ideas behind receptive field.

VGG architecture, Inception and ResNet variations were covered. 

The concept of having GRF bigger than the image and compensating for different object sizes are really fundamental to have deeper understanding for CV and what exactly the network is trying to figure out.

The approach of having shortcut paths in a network for easy transfer of information during forward pass as well as gradients helps to go deeper with number of layers in a network while maintaining high accuracy. 

The entry point for .py files in `main.py` which imports other .py files. The idea behind running pipeline using `configuration.cfg` was to allow changing of metrics without user modifying the code. The `.cfg` file is read only once, the first time and thne a dictionary is set as an attribute of the fucntion and utilized several times in entire program flow.

(for values that should be interpreted as boolean used 0 for False and any value for True, so for shuffle only specifying 0 will make it False, True otherwise, I have stuck to using 1 here)
##### `config.cfg` has following strucrure for  S7:


  **[model_config]**

    workers = 1
    batch_size = 128
    pin_memory = 1
    shuffle = 1
    epochs = 25
    combinations = L1+BN, L2+BN, L1+L2+BN, GBN, L1+L2+GBN

  **[optimizer]**

    lr = 0.001
    optimizer_type = sgd # either sgd or adam or any other from `torch.optim`
    momentum = 0.9

  **[scheduler]**

    scheduler_type = steplr # any one of the available from `torch.optim.lr_scheduler`
    step = 3
    gamma = 0.379
    milestones = 5,10,15

  **[regularization]**

    l1 = 0.001
    l2 = 4e-4

  **[plots]**

    to_plot = train_losses, test_losses, train_accuracy, test_accuracy

*These values are read from config file and utilized in the program. This can be further enhanced to include other Learning Rate schedulers. Currently only 2 optimizers are supported SGD and Adam*



## L1

1. L1 penalizes the sum of absolute weights
2. It pushes weights more towards 0
3. L1 can be thought as **reducing the number of features** in the model i.e. if there are multiple features, L1 may push the factors for some to very low values thus rendering them not useful
4. Thus helping in reduction of model's complexity
5. L1 is more sparse and robust to outliers



## L2

1. L2 penalizes the sum of squared weights
2. Less sparse
3. Not robust to outliers
4. Punishes large weights more (squared)

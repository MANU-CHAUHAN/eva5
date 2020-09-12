# S7 EVA5

The session was about modularizing the previous code for Batch Normalization and Regularization effects and topics covered were:
>Normal Convolutions

>Pointwise Convolutions

>Concept of Channels

>Receptive Field

>Checkerboard Issue

>Atrous or Dilated Convolutions

>Dense Problems

>Deconvolutions or Transpose Convolutions

>Pixel Shuffle Algorithm (SKIPPED)

>Depthwise Separable Convolution

>Spatially Separable Convolution (SKIPPED)

>Grouped Convolutions 

**Overfitting** is a phenomenon that occurs when a machine learning or statistics model is tailored to a particular dataset and is unable to generalise to other datasets. This usually happens in complex models, like deep neural networks.

**Regularisation** is a process of introducing additional information in order to prevent overfitting. 

Training efficiently requires weights to be in around same scale of distribution otherwise we get ellipses or contours for errors which make convergence really difficult bcoz one feature would easily update while other would be still stuck in weight space to try to move towards minima. Thus, two different features with different scales would keep trying to update themselves and may never achieve good minima due to different scales of distribution.

*The different types of convolutions allowed one to peak into the inner working at deeper levels and understand how various types of convolutions can be helpful in achieving the task at hand.*

> Dilation convolution helps to increase the receptive field of a kernel according to the dilation rate, this helps to provide a wider context from the input of previous layer. These, however, are not good feature extractors and should be followed by a normal 3x3 Conv2d.

> Depthwise separable convolutions helps us to carry out the convolution process in 2 steps, first is to have same number of kernels as input channels. These kernels convolve on each and every channel input separately and then the output is combined together to get feature map of depth= number of kernels. Then second step involves a convolution by Nx(1x1xd) convolution where `d` is depth of feature map=number of output kernels and `N` is the number of 1x1 kernels used after 1st step.
> This convolution helps to dramatically reduce the number of parameters in a model with the flexibility of reducing or creating channels at 1x1 step.

> Spatially Separable Convolutions aim to break down a feature detector into two parts of convolution process 1st-> Kx1 followed by 2nd-> 1xK convolution.
> This, like depthwise reduces the number of parameters but are often ignored due to lack of good breakdown of features into 2 kernels except few common edge detectors.

> Grouped convolutions is simply the idea of combining multiple convolutions from same incoming input with different kernel sizes and topology. Good example is Inception model.

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

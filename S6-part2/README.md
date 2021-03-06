# S7 EVA5

The session was about modularizing the previous code for Batch Normalization and Regularization effects. 

**Overfitting** is a phenomenon that occurs when a machine learning or statistics model is tailored to a particular dataset and is unable to generalise to other datasets. This usually happens in complex models, like deep neural networks.

**Regularisation** is a process of introducing additional information in order to prevent overfitting. 

Training efficiently requires weights to be in around same scale of distribution otherwise we get ellipses or contours for errors which make convergence really difficult bcoz one feature would easily update while other would be still stuck in weight space to try to move towards minima. Thus, two different features with different scales would keep trying to update themselves and may never achieve good minima due to different scales of distribution.

Colab link for the .ipynb file in repo: https://colab.research.google.com/drive/1qrJw9AaVtTYqh_EfOaSw8sXXiImvL_dI?usp=sharing

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

## Batch Normalization

![BN](https://kharshit.github.io/img/batch_normalization.png)

> Helps to reduce internal covariate shift, that is the distribution of inputs for each layer of the network.

> Helps to speed up layers learn independently

> Speeds up learning process

> Has a regularization effect (Each mini-batch is scaled using its mean and standard deviation. This introduces some noise to each layer, providing a regularization effect)


Ghost Batch Normalization helps in regularization after creating smaller virtual ('ghost') batches using the 'num_split' factor while still having larger batch in memory. The calculations for batch-wise mean and standard deviation are split according to ghost batch size which helps to have smaller batch size and more stochasticity hence more weight updates as **larger batch/smaller batch** divides the epochs further and allowing more passes due to virtual smaller batches over the entire training dataset, while still having large or medium sized total sample batch in memory. Therefore, the benfits of bigger batch size and randomness of smaller in-memory 'virtual' batch.


#### from paper: Four things everyone should know to improve Batch Normalization

> Why might Ghost Batch Normalization be useful? One reason is its power as a regularizer: due to
the stochasticity in normalization statistics caused by the random selection of minibatches during
training, Batch Normalization causes the representation of a training example to randomly change
every time it appears in a different batch of data. Ghost Batch Normalization, by decreasing the
number of examples that the normalization statistics are calculated over, increases the strength of this
stochasticity, thereby increasing the amount of regularization. Based on this hypothesis, we would
expect to see a unimodal effect of the Ghost Batch Normalization size B0 on model performance—
a large value of B0 would offer somewhat diminished performance as a weaker regularizer, a very
low value of B0 would have excess regularization and lead to poor performance, and an intermediate
value would offer the best tradeoff of regularization strength.


-----------
![](S7_Test_loss_change.png)

----------
![](S7_Test_loss_accuracy.png)

----------
![](S7_Train_loss_change.png)

-------------
![](S7_gbn_mis.PNG)

-------------
![](S7_l1_l2_bn_mis.PNG)

--------------
![](S7_l1_bn_mis.PNG)

-------------
![](S7_l2_bn_mis.PNG)

------------------
![](S7_l1_l2_gbn_mis.PNG)
-----------------------------------


![](https://miro.medium.com/max/1328/1*l6E7S7S36mPPwZ2yMlU_og@2x.png)



![](https://miro.medium.com/max/1400/1*hC68XigZjhYVCEJbwuVZrQ@2x.png)







![](https://miro.medium.com/max/1400/1*bV49pcaBBMW79adaN5gp_w@2x.png)





- #### *η* = 1,

- #### *H = 2x*(*wx+b-y*)

  

![](https://miro.medium.com/max/976/1*k9fevtlt4GIrkIao9fli1g@2x.png)



![](https://miro.medium.com/max/1400/1*cX7OClIl2O-ZLTVqfB47-A@2x.png)



![](https://miro.medium.com/max/1400/1*pU47ApyQYzt5Yj_jTpVNFw@2x.png)



![](https://miro.medium.com/max/1400/1*abHVX1SUuzdiAWcpDq4TzA@2x.png)





![](https://miro.medium.com/max/1400/1*5etBmH3PZk7dR0e_H3zA8g@2x.png)





## [With vs. Without Regularisation](https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261) 

Observe the differences between the weight updates with the regularisation parameter *λ* and without it. Here are some intuitions.

**Intuition A:**

Let’s say with Equation 0, calculating *w-H* gives us a *w* value that leads to overfitting. Then, intuitively, Equations {1.1, 1.2 and 2} will reduce the chances of overfitting because introducing *λ* makes us shift *away* from the very *w* that was going to cause us overfitting problems in the previous sentence.

**Intuition B:**

Let’s say an overfitted model means that we have a *w* value that is **perfect**for our model. ‘Perfect’ meaning if we substituted the data (*x*) back in the model, our prediction *ŷ* will be very, very close to the true *y*. Sure, it’s good, but we don’t want perfect. Why? Because this means our model is only meant for the dataset which we trained on. This means our model will produce predictions that are far off from the true value for *other* datasets. So we settle for **less than perfect**, with the hope that our model can also get close predictions with other data. To do this, we ‘taint’ this perfect *w* in Equation 0 with a penalty term *λ.* This gives us Equations {1.1, 1.2 and 2}.

**Intuition C:**

Notice that *H*  is **dependent** on the model (*w* and *b*) and the data (*x* and *y*). Updating the weights based *only* on the model and data in Equation 0 can lead to overfitting, which leads to poor generalisation. On the other hand, in Equations {1.1, 1.2 and 2}, the final value of *w* is not only influenced by the model and data, but *also* by a predefined parameter *λ* which is **independent** of the model and data. Thus, we can prevent overfitting if we set an appropriate value of *λ*, though too large a value will cause the model to be severely underfitted.

**Intuition D:**

1. Weights for different potential training sets will be more similar — which means that the model variance is reduced (in contrast, if we shifted our weights randomly each time just to move away from the overfitted solution, the variance would not change).
2. We will have a smaller weight for each feature (and/or less features if using L1 reg.). Why does this decrease overfitting? The way I find it easy to think about is that in a typical case we will have a small number of simple features that will explain most of the variance (e.g. most of y will be explained by y_hat = ax+b); but if our model is not regularized, we can add as many more features we want that explain the residual variance of the dataset (e.g. y_at = ax + bx² +cx³ + dx⁴ + e), which would naturally overfit the trainin set. Inroducing a penalty to the sum of the weights means that the model has to “distribute” its weights optimally, so naturally most of this “resource” will go to the simple features that explain most of the variance, with complex features getting small or zero weights.





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

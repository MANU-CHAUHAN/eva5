# eva5


## S4 was about covering architectural basics and topics covered were:
#### Fully Connected Layers:
> Fully connected layers are NO LONGER used until and unless required, large number of parameters, not good for converting 2D information to 1D as that leads to loss of valuable spatial information including loss of translational, rotational, skew invariance in comparison to operations or Convolutions on 2D data.

#### VGG - The Last awesome network based on OLD architecture - Pre-2014:
> The importance VGG holds, despite not winning the competition, use of FC and the addition of large number of parameters with it. 

#### Modern Architectures - Post-2014:
> Modern acrchitectures removed using FC and instead develop Fully Convolutional Networks. Although 1x1 is used after Global Average Pooling if need be. 
> Major Points(from content):

>     ResNet is the latest among the above. You can clearly note 4 major blocks. 
>      The total number of kernels increases from 64 > 128 > 256 > 512 as we proceed from the first block to the last (unlike what we discussed where at each  block we expand to 512. Both architectures are correct, but 64  ... 512 would lead in lesser computation and parameters. 
>     Only the most advanced networks have ditched the FC layer for the GAP layer. In TSAI we will only use GAP Layers.
>     Nearly every network starts from 56x56 resolution!


#### SoftMax:
> Scales the inputs to be the sum=1 but is NOT PROBABILITY, it's more likelihood in terms of interpretation.The softmax function is often used in the final layer of a neural network-based classifier, one advantage of using the softmax at the output layer is that it improves the interpretability of the neural network.. Now we use `log_softmax` as it is Negative Log of Likelihood which is better than normal `softmax` as it scales the correct class to lower loss value thus allowing network to force fine tuning of weights towards making correct class scores higher. `log(softmax)` is slower and mathematically unstable hence in-built `log_softmax` should be used. To stabalize, the max of the entire vector is subtracted from the vector and then softamx is carried out to avoid over or underflow of values.


#### MaxPooling:
>    used when we need to reduce channel dimensions
>    not used close to each other
>    used far off from the final layer
>    used when a block has finished its work (edges-gradients/textures-patterns/parts-of-objects/objects
>    nn.MaxPool2D()

#### Batch Normalization:
Normalizes the incoming bathc at every layer of the network to make amplitudes more prominent by scaling the values of the channel so that next immediate layer can figure out features with more clarity and confidence in order to make better decision by combination of features. It's an essential ingredient of modern DNN architectures and allows having deeper layers and higher learning rates for better training of networks.
>    used after every layer
>    never used before last layer
>    indirectly you have sort of already used it!
>    nn.BatchNorm2d()

#### DropOut:
>    A regularization technique that randomly drops weights during traoining to avoid over-dependence on specific features and helps network to focus on other more relevant features for better decision making.
>    used after every layer
>    used with small values
>    is a kind of regularization
>    not used at a specific location
>    nn.Dropout2d()

#### Learning Rate:
>     The factor by which the parameters of the network must be updated, this is usally less than 1 and helps to take fraction of the gradinet wrt. loss while update rule.

#### Batch Size:
>    Signifies the total number of samples to be processed for forward pass -> loss calculation -> gradient calculation -> gradient update. The idea is to have batch size more or equal to number of classes as we want our network to look at all classes and give equal importance while learning parameters and be biased towards few or one class and have loss refelcting only one or few classes. The loss should not be stagnant and must be calculated for all classes simultaneously in order for effective learning.


##### S4, task: to achieve accuracy >= 99.4% on MNIST test data with less than 20k params within 20 epochs in PyTorch.

>    Since MNIST is an easy dataset, the ideaology of increasing kernels and then decreasing with values ranging from 32 to 512, is irrelevant as that would be overkill in this scenario. The aproach is to force backprop to utilize only small number of kernels to pick important features from gray scaled images and then form digits' representation at deeper layers. 


# EVA5

## S5 was about understanding architectural basics, going through multiple approaches(with code) and topics covered were:
### The steps for assignment are: 1(basic) -> 2(BN)+ 3(Dropout) -> 4(parameter decrease)+ 5(Global Average Pooling) 6(Image Augmentation)+ 7(LR Scheduler).
- Code 1 - Set up

  > The ideas was to first have a basic understanding of the dataset and different images and understanding of the framwework to be used.

  

- Code 2 - Basic Skeleton

  > Here the idea was to first have a simple working skeleton of the program with basic transformers, data loaders, config kwargs etc setup and read data and process with a simple model and see if some accuracy is coming up after loss calculation.

  

- Code 3 - Lighter Model

- > Here a reduced number of  parameter were considered for the model. and accuracy was noted

- Code 4 - Batch Normalization

  > Basic overview of BN and it's usage in code and the effects

  

- Code 5 - Regularization

- > What is regularization and how to apply it and when to apply it was covered with code's output

  

- Code 6 - Gobal Average Pooling

  > GAP was shown in code and the idea was explained behind it's usage. Also, the conceot of increasing model's capacity after GAP and usage with 1x1xN was discussed

  

- Code 7 - Increasing Capacity

  > How to increase model's capacity, usually after GAP. FC could be used but 1x1 offers more elegant and efficient solution

  

- Code 8 - Correct MaxPooling Location

  > MaxPooling should be added at a correct position in network, it depends upon RF and data set.

  

- Code 9 - Image Augmentation

  > Amazing technique to make model more robust and add regularization, make model more generalizable.

  

- Code 10 - Playing naively with Learning Rates

  > How learning rate affects the learning processa nd has huge importance to play in faster convergence and reaching global minima.

  

  task: to achieve accuracy >= 99.4% on MNIST test data with less than 10k params within 15 epochs using PyTorch

>    Since MNIST is an easy dataset, the ideaology of increasing kernels and then decreasing with values ranging from 32 to 512, is irrelevant as that would be overkill in this scenario. The aproach is to force backprop to utilize only small number of kernels to pick important features from gray scaled images and then form digits' representation at deeper layers. 
>
>    The target was successfully achieved.


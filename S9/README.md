**S9 DATA AUGMENTATION**

And why we should fall in love with it!

-----------------------------------------------------
PMDA - Poor Man's Data Augmentation Strategies

MMDA* - Middle-Class Man's Data Augmentation Strategies

RMDA* - Rich Man's Data Augmentation Strategies

Train Results: at 40th Epoch: **Train Accuracy=97.07%** and **Test accuracy=93.35%**

*Layers used for GradCAM results: Layer1, Layer2, Layer3, each have two parts hence both Conv layers were plotted and have size greater than 7x7*

**Config file values used:**
----------------------------

**[model_config]**

workers = 2

batch_size = 128

pin_memory = 1

shuffle = 1

epochs = 40

combinations = L2

**[optimizer]**

lr = 0.0099841

optimizer_type = sgd

momentum = 0.9

nesterov = True

**[scheduler]**

scheduler_type = steplr

step = 3

gamma = 0.689

**[regularization]**

l2 = 21e-3

--------------------------

> Relationship between Data and Validation Accuracy

> Data Augmentation

>PMDA
    
    Scale; Translation; Rotation; Blurring; Image Mirroring; Color Shifting / Whitening. 

    Simple Stuff which most probably is in-built in Pytorch. 

>MMDA - Albumentations

    Elastic Distortion

    Cutout

    MixUp

    RICAP

>RMDA
    
    [Auto Augment](https://arxiv.org/abs/1805.09501)

>Patch Gaussian

>Global Average Pooling

> **GradCam**

> Role of GradCAM in Diagnosing DNN

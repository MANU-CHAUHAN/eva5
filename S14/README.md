# Session 14

#### <u>RCNN, FastRCNN, FasterRCNN and Mask-RCNN</u>

##### *Contributors: Ashna Eldho(ashnaeldho12@gmail.com) and Manu Chauhan(manuchauhan1992@gmail.com, 001colab@gmail.com)*

**Preparation of MiDas' as well as PlaneRCNN's output on HardHat, Mask, Vest, Boots dataset.  [Google drive link](https://drive.google.com/drive/folders/14ZG4-izFkrhuCIPfxzrq5rf-o5nqETLg?usp=sharing)**

##### **Topics:**

- RCNN Family - The Beginning

- RCNN

  - Problems with RCNN

- Fast RCNN

- Faster RCNN

- Region Proposal Networks

  - Anchor Classification Layer
  - Region Proposal Layer
  - Region Proposal Network
  - Proposal Layer
  - Anchor Target Layer
  - Calculating RPM Loss
  - Classification Box Regression Loss
  - Classification Layer

- Mask R-CNN

  - RoI Align

  

## The dataset:

#### Why data? 🤨

### "Data, Data, Data... I cannot build bricks without clay." - Sherlock Holmes

Data is an indespensable part of our lives and modern technological innovation, be it healthcare, autonomous driving systems, robots, speech tranlation, text summarization or the over whelming power of GPT-3, for all those to work at an acceptable level of accuracy, one needs very large volume of data. Moreover, computers do not come with an evolving concious mind (u wish?? or better code one!). Similarly, for our next task on Monocular Depth Estimation, we had to prepare dataset via one of the most efficient approaches.


### What all we had to do? 🤔

#### Prepare depth estimated set of images... wait.... but why ??

![](https://miro.medium.com/max/2363/1*k15IG4yYYk_cgsmXMNpeBg.jpeg)

Depth in images and videos, or in computer vision in general, to date remains one of the most crucial parameters when it comes to automation, robotics, Augmented Reality, surveillance, self driving systems or even advanced photography features in latest cameras.

Depth is a key prerequisite to perform multiple tasks such as perception, navigation, and planning.

Understanding of how far an entity is from the camera has always been a challenging task. Factors such as occlusion, dynamic object in the scene and imperfect stereo correspondence, have made estimation of depth difficult task for computers. Which is the very reason for most companies and solutions integrating laser based sensors for having a 3-D reconstruction view of the world. The major drawback with this, like LiDAR, is the absence of capability of laser based sensors to actually capture the entities in view as different from one another. Eg: for LiDAR, a garbage can is an just an object and so is a human sitting on pedestal, without any knowledge of how one is supposed to be different from one another.

![](https://miro.medium.com/max/770/1*P1bTz2TsAmAjjvienVxtIw.jpeg)

Estimation of depth is not only the first challenging task in this problem domain, the precursor to this challenge is the availability of fine tuned, good quality, large data set.

##### Monocular Depth Estimation:
In simple words: Monocular Depth Estimation is the task of estimating scene depth using a single image.

![Sample image (left) and its depth annotation in RGB-D (right)](https://miro.medium.com/max/875/1*4PH2J5iGG1wgnIj4XzQEmg.jpeg)

For preparing depth data set on hardhat, mask, vest and boots, we used Intel's MiDaS github repo (https://github.com/intel-isl/MiDaS), which covers the paper `Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer`. The paper discusses the lack of availability independent good quality dataset for training and previous techniques which relied upon large data volume for the task at hand.

`We show that a model trained on a rich and diverse set of images from different sources, with an appropriate training procedure, delivers state-of-the-art results across a variety of environments. To demonstrate this, we use the experimental protocol of zero-shot cross-dataset transfer. That is, we train a model on certain datasets and then test its performance on other datasets that were never seen during training. The intuition is that zero-shot cross-dataset performance is a more faithful proxy of “real world” performance than training and testing on subsets of a single data collection that largely exhibit the same biases.` - From the paper.

For the task:
1) The dependencies were installed (pytorch, torchvision, opencv)
2) The pre-trained model weights were downloaded (model-f45da743.pt)
3) All the images from the previous data set were placed in a folder (`input folder`) and run.py was used.
4) All the inverse depth predicted images are places in `output` folder


##### Prepare dataset for Planar regions segmented on our images... wait.... but why ??
Next task involved detecting and reconstructing piecewise planar regions in RGB images. For this we utilised Nvidia's `PlaneRCNN: 3D Plane Detection and Reconstruction from a Single Image` github repo (https://github.com/NVlabs/planercnn). PlaneRCNN uses a variant of Mask R-CNN to detect planes with plane parameters and segmentation mask.
PlaneRCNN then jointly refines the segmentation masks with a new(relatively, paper was out in Jan 2019) loss function whiuch enforces consistency with a nearby view during training. PlaneRCNN helps to extract `robust` planar regions from images which drastically impact the feasibility and accuracy in Computer Vision for tasks such as robotics, augmented reality, virtual reality etc. 

***A difficult yet fundamental task is the inference of a piecewise planar structure from a single RGB image, posing two key challenges 😥:*** (from the paper)

**1)** First, 3D plane reconstruction from a single image is an ill-posed problem, requiring rich scene priors.
**2)** Second, planar structures abundant in man-made environments often lack textures, requiring global image understanding as opposed to local texture analysis.

![](https://research.nvidia.com/sites/default/files/publications/planercnn.jpg)

The Mask R-CNN model generates bounding boxes and segmentation masks for each instance of an object in the image.

For the task (on Google Colab):
1) Had to downgrade gcc version to 5.0
2) Install CUDA version less than 10.0 as the repo's code did not support 10.0+ versions (Worked fine with version Cuda)
3) For step 2, had to download the appropriate .deb file using WGET
4) Then install using the debian package manager
5) cloned Nvidia's repo
6) Installed dependencies from reqirements.txt file (cffi==1.11.5, numpy==1.15.4, opencv-python==3.4.4.19, scikit-image==0.14.1, torch==0.4.1, tqdm==4.28.1)
7) Then compile nms and roialign in PlaneRCNN for Cuda version installed (8-0), using the build.py files for NMS and ROIAlign
8) Test on given sample images: !python evaluate_adf.py --methods=f --suffix=warping_refine --dataset=inference --customDataFolder=example_images
9) Creating `camera.txt` file in our custom or user specific images folder (The camera parameters should be put under a .txt file with 6 values (fx, fy, cx, cy, image_width, image_height) separately by a space.)
10) The pre-trained model weights were downloaded from Dropbox url: https://www.dropbox.com/s/yjcg6s57n581sk0/checkpoint.zip?dl=0
11) The corresponding `.pt` file was placed under `checkpoint` dir having `planercnn_normal_warping_refine/checkpoint.pth`
12) The Google Drive was mounted for the evaluate.py file to infer on our dataset.
13) !python evaluate.py --methods=f --suffix=warping_refine --dataset=inference --numTestingImages=4000 --customDataFolder='/content/gdrive/My Drive/YoloV3/data/customdata/images/'
14) `--numTestingImages` argument's default value is 100, since our dataset has 3500+ images the argument's value was changed.
15) `--customDataFolder` is the absolute path of the dataset for which we want to generate Planar images
16) The results for all the images in our dataset were stored under `inference` directory.
17) All the resulting, planar segmented images were copied back to G drive.

**Plane representation**
-----------------------------
In this project, plane parameters are of absolute scale (in terms of meters). Each plane has three parameters, which equal to plane_normal * plane_offset. Suppose plane_normal is (a, b, c) and plane_offset is d, every point (X, Y, Z) on the plane satisfies, aX + bY + cZ = d. Then plane parameters are (a, b, c)*d. Since plane normal is a unit vector, we can extract plane_normal and plane_offset from their multiplication. (from Nvidia's repo)













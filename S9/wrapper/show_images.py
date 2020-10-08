import numpy as np
import torchvision
import matplotlib.pyplot as plt


def show_image(img, title, figsize=(7, 7), normalize=False):
    if normalize:
        img = img / 2 + 0.5  # denormalize, by reverse of scale and shift by approximate 0.5 value
    numpy_img = img.numpy()
    fig = plt.figure(figsize=figsize)
    plt.imshow(np.transpose(numpy_img, (1, 2, 0)), interpolation='none')
    plt.title(title)


def show_training_data(dataset, classes):
    dataiter = iter(dataset)
    images, labels = next(dataiter)
    # images, labels = images.to('cpu'), labels.to('cpu')
    for i in range(10):
        index = [j for j in range(len(labels)) if labels[j] == i]
        show_image(torchvision.utils.make_grid(images[index[0:5]], nrow=5, padding=2, scale_each=True), classes[i],
                   normalize=True)

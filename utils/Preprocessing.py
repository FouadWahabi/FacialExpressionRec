import numpy as np
from scipy import ndimage


def preprocessing(images):
    # Normalize the data
    images = images - images.mean(axis=1).reshape(-1, 1)
    images = np.multiply(images, 100.0 / 255.0)
    each_pixel_mean = images.mean(axis=0)
    each_pixel_std = np.std(images, axis=0)
    images = np.divide(np.subtract(images, each_pixel_mean), each_pixel_std)

    return images
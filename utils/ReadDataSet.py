import pandas as pd

from utils.Preprocessing import *


def read_dataset(path):
    data = pd.read_csv(path)
    train_data = data[data.Usage == "Training"]
    print(train_data.head())
    pixels_values = train_data.pixels.str.split(" ").tolist()
    pixels_values = pd.DataFrame(pixels_values, dtype=int)
    images = pixels_values.values
    images = images.astype(np.float)
    images = preprocessing(images)

    labels_flat = train_data["emotion"].values.ravel()
    print(labels_flat)
    num_classes = np.unique(labels_flat).shape[0]
    # flatten labels
    index_offset = np.arange(labels_flat.shape[0]) * num_classes
    labels = np.zeros((labels_flat.shape[0], num_classes))
    labels.flat[index_offset + labels_flat.ravel()] = 1
    labels = labels.astype(np.uint8)

    return images, labels

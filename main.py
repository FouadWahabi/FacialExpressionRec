from __future__ import division, print_function, absolute_import

import sys

from utils.EmotionRecognition import *
from utils.ReadDataSet import *

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("missing arguments")
        exit()

    # Load training data
    images, labels = read_dataset("data/fer2013.csv")

    network = EmotionRecognition(images, labels)
    network.build_network()
    if sys.argv[1] == 'train':
        network.start_training()
        network.save_model()

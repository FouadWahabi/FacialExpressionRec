from __future__ import division, print_function, absolute_import

from os.path import isfile, join

import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, reshape
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization


class EmotionRecognition:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.face_size = np.ceil(np.sqrt(data.shape[1])).astype(np.int8)
        self.num_emotions = labels.shape[1]

    def build_network(self):
        # This network architecture is based on the work of A. Gudi : Recognizing semantic features
        # using deep learning

        print('[+] Building CNN')

        self.network = input_data(shape=[None, self.data.shape[1]])
        self.network = reshape(self.network, new_shape=[-1, self.face_size, self.face_size, 1])
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = local_response_normalization(self.network)
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 128, 4, activation='relu')
        self.network = dropout(self.network, 0.3)
        self.network = fully_connected(self.network, 3072, activation='relu')
        self.network = fully_connected(self.network, self.num_emotions, activation='softmax')
        self.network = regression(self.network,
                                  optimizer='momentum',
                                  loss='categorical_crossentropy')
        self.model = tflearn.DNN(
            self.network,
            checkpoint_path='model/emotion_recognition2',
            max_checkpoints=1,
            tensorboard_verbose=2
        )

    def start_training(self):
        if self.network is None:
            raise Exception("Network should be built before training")

        # Training
        print('[+] Training network')
        self.model.fit(
            self.data, self.labels,
            validation_set=0.1,
            n_epoch=100,
            batch_size=50,
            shuffle=True,
            show_metric=True,
            snapshot_step=200,
            snapshot_epoch=True,
            run_id='emotion_recognition2'
        )

    def save_model(self):
        self.model.save(join("model", "emotion_recognition2"))
        print('[+] Model trained and saved at ' + "emotion_recognition2")

    def load_model(self):
        if isfile(join("model", "emotion_recognition2")):
            self.model.load(join("model", "emotion_recognition2"))
        else:
            raise Exception("You should train the model first")
        print('[+] Model loaded from ' + "emotion_recognition2")

    def predict(self, image):
        if image is None:
            return None
        image = image.reshape(-1, self.face_size * self.face_size)

        return self.model.predict(image)

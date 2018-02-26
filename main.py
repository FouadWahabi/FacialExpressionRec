import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == '__main__':
    from utils.ReadDataSet import *

    # Load traingin data
    images, labels = read_dataset("data/fer2013.csv")

    image_width = image_height = np.ceil(np.sqrt(images.shape[1])).astype(np.uint8)

    # init tf inputs
    # images
    x = tf.placeholder('float', shape=[None, images.shape[1]])
    # labels
    y_ = tf.placeholder('float', shape=[None, labels.shape[1]])

    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 64], stddev=1e-4))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))

    # (27000, 2304) => (27000,48,48,1)
    image = tf.reshape(x, [-1, image_width, image_height, 1])

    # First convolution layer
    h_conv1 = tf.nn.relu(tf.nn.conv2d(image, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)

    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # Second convolution layer
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=1e-4))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[128]))

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_norm1, W_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)

    h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    h_pool2 = tf.nn.max_pool(h_norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 128])

    # Densely connected layer 1
    W_fc1 = tf.Variable(tf.truncated_normal([12 * 12 * 128, 3072], stddev=1e-4))
    b_fc1 = tf.Variable(tf.constant(0.0, shape=[3072]))

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Densely connected layer 2
    W_fc2 = tf.Variable(tf.truncated_normal([3072, 1536], stddev=1e-4))
    b_fc2 = tf.Variable(tf.constant(0.0, shape=[1536]))

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # dropout
    keep_prob = tf.placeholder('float')
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # readout layer for deep net
    W_fc3 = tf.Variable(tf.truncated_normal([1536, labels.shape[1]], stddev=1e-4))
    b_fc3 = tf.Variable(tf.constant(0.0, shape=[labels.shape[1]]))

    # CrossEntropySoftmax output
    y = tf.nn.softmax(tf.add(tf.matmul(h_fc2_drop, W_fc3), b_fc3))

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # optimisation function
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    # prediction function
    predict = tf.argmax(y, 1)

    print_predict = tf.Print(predict, [predict], "The prediction is : ")

    # evaluation
    correct_prediction = tf.equal(print_predict, tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


    # start tf session
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()

    sess.run(init)

    # Test with a small batch of 10 images
    batch_xs = images[0:10]
    batch_ys = labels[0:10]

    # Do the training
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    train_accuracy = accuracy.eval(feed_dict={x: batch_xs,
                                              y_: batch_ys,
                                              keep_prob: 1.0})

    print('training_accuracy => %.4f' % (train_accuracy))

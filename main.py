import tensorflow as tf


# serve data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


if __name__ == '__main__':
    from utils.ReadDataSet import *

    # Load traingin data
    images, labels = read_dataset("data/fer2013.csv")

    # split data into training & validation
    VALIDATION_SIZE = 1709

    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]

    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]

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

    # (40000, 7, 7, 64) => (40000, 3136)
    h_fc2_flat = tf.reshape(h_fc1, [-1, 3072])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc2_flat, W_fc2) + b_fc2)

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
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # labels
    labels_flat = tf.argmax(y_, 1)

    # prediction function
    predict = tf.argmax(y, 1)

    #    print_labels = tf.Print(labels_flat, [labels_flat], "The labels are : ")

    #    print_predict = tf.Print(predict, [predict], "The prediction is : ")

    # evaluation
    correct_prediction = tf.equal(predict, labels_flat)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # start tf session
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()

    sess.run(init)

    # Do iterative training
    training_iterations = 3000
    batch_size = 50
    display_step = 1
    epochs_completed = 0
    index_in_epoch = 0
    num_examples = train_images.shape[0]

    for i in range(0, training_iterations):
        # get new batch
        batch_xs, batch_ys = next_batch(batch_size)

        if i % display_step == 0 or (i + 1) == training_iterations:

            train_accuracy = accuracy.eval(feed_dict={x: batch_xs,
                                                      y_: batch_ys,
                                                      keep_prob: 1.0})
            if (VALIDATION_SIZE):
                validation_accuracy = accuracy.eval(feed_dict={x: validation_images[0:batch_size],
                                                               y_: validation_labels[0:batch_size],
                                                               keep_prob: 1.0})
                print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (
                    train_accuracy, validation_accuracy, i))


            else:
                print('training_accuracy => %.4f for step %d' % (train_accuracy, i))

            # increase display_step
            if i % (display_step * 10) == 0 and i and display_step < 100:
                display_step *= 10
        # train on batch
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

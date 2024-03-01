#!/usr/bin/env python3
"""
Using tf.compact.v1
to build our lenet5 architechture model
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def lenet5(x, y):
    """
    building it in here using the:
    x is a tf.placeholder of shape (m, 28, 28, 1) containing the input images for the network
    m is the number of images
    y is a tf.placeholder of shape (m, 10) containing the one-hot labels for the network
    The model should consist of the following layers in order:
    Convolutional layer with 6 kernels of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with the he_normal initialization method: tf.keras.initializers.VarianceScaling(scale=2.0)
    All hidden layers requiring activation should use the relu activation function
    you may import tensorflow.compat.v1 as tf
    you may NOT use tf.keras only for the he_normal method.
    Returns:
    a tensor for the softmax activated output
    a training operation that utilizes Adam optimization (with default hyperparameters)
    a tensor for the loss of the netowrk
    a tensor for the accuracy of the network
    """
    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=(5, 5),
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))(x)
    pooling1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))(pooling1)
    pooling2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    flattened = tf.layers.Flatten()(pooling2)
    f1 = tf.layers.Dense(units=120, activation='relu',
                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))(flattened)
    f2 = tf.layers.Dense(units=84, activation='relu',
                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))(f1)
    logits = tf.layers.Dense(units=10,
                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))(f2)
    softmax_output = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax_output, axis=1),
                                               tf.argmax(y, axis=1)), tf.float32))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    return softmax_output, train_op, loss, accuracy

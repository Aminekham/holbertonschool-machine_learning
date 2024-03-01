#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import applications, layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

def preprocess_data(X, Y):
    X_resized = tf.image.resize(X, (32, 32))    
    X_p = (X_resized / 255.0).numpy().astype(np.float32)
    Y_p = to_categorical(Y, 10)
    return X_p, Y_p


def build_cifar10_model():
    # Load the CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Preprocess the data
    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)

    #Choosing the model
    base_model = applications.DenseNet169(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Build the classification head
    model = models.Sequential([
        layers.Rescaling(scale=1./255, input_shape=(32, 32, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.1),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=30, validation_split=0.2, batch_size=32)
    model.save('cifar10.h5')

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}, Test Loss: {test_loss}')

    return model

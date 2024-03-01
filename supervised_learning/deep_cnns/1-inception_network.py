#!/usr/bin/env python3
"""
building the architecture
needed to match the idea of the
inception neural networks
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    
    """
    X_input = K.Input(shape=(224, 224, 3))
    c1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation = 'relu')(X_input)
    p1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c1)
    c2 = K.layers.Conv2D(filters=64, strides=(1, 1), activation='relu',kernel_size=(1, 1))(p1)
    c21 = K.layers.Conv2D(filters=192, strides=(3, 3), kernel_size=(3, 3), activation='relu')(c2)
    p2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c21)
    inception1 = inception_block(p2, filters=[64, 96, 128, 16, 32, 32])
    inception2 = inception_block(inception1, filters=[128, 128, 192, 32, 96, 64])
    p3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(inception2)
    inception3 = inception_block(p3, filters=[192, 96, 208, 16, 48, 64])
    inception4 = inception_block(inception3, filters=[160, 112, 224, 24, 64, 64])
    inception5 = inception_block(inception4, filters=[128, 128, 256, 24, 64, 64])
    inception6 = inception_block(inception5, filters=[112, 144, 288, 32, 64, 64])
    inception7 = inception_block(inception6, filters=[256, 160, 320, 32, 128, 128])
    p4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(inception7)
    inception8 = inception_block(p4, filters=[256, 160, 320, 32, 128, 128])
    inception9 = inception_block(inception8, filters=[384, 192, 384, 48, 128, 128])
    avg_p = K.layers.GlobalAveragePooling2D()(inception9)
    drop = K.layers.Dropout(0.4)(avg_p)
    flat = K.layers.Flatten()(drop)
    dense = K.layers.Dense(units = 1000,activation = 'softmax')(flat)
    model = K.Model(inputs = [X_input], outputs = [dense])
    return model

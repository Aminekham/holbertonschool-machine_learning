#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def convolutional_GenDiscr():
    def generator():
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=(16,)))
        model.add(layers.Dense(2048))
        model.add(layers.Reshape((2, 2, 512)))
        
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(16, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(1, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('tanh'))
        
        return model

    def discriminator():
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=(16, 16, 1)))
        
        model.add(layers.Conv2D(32, (3, 3), padding='same'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Activation('relu'))
        
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Activation('relu'))
        
        model.add(layers.Conv2D(128, (3, 3), padding='same'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Activation('relu'))
        
        model.add(layers.Conv2D(256, (3, 3), padding='same'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Activation('relu'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        
        return model

    return generator(), discriminator()

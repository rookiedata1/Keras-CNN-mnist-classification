# -*- coding: utf-8 -*-
# @Time    : 2018/12
# @Author  : wengfutian
# @Email   : wengfutian@csu.edu.cn
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.utils import plot_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras import backend as K

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":  # for tensorflow
            inputShape = (depth, height, width)
        model.add(Conv2D(20, (5, 5), input_shape=inputShape, padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(40, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(80, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(80, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(40, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        # softmax classifier
        # model.add(Dense(1000, activation='softmax'))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        model.summary()
        # Save a PNG of the Model Build
        plot_model(model, to_file='../imgs/AlexNet.png')
        # return the constructed network architecture
        return model
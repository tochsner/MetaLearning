""" 
Downloads the FashionMNIST dataset and prepares it for the use with keras and tooc.
"""

import keras
from keras import utils
from keras.datasets import fashion_mnist
import numpy as np
import random

num_classes = 10
img_rows = 28
img_cols = 28

def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return ((x_train, y_train), (x_test, y_test))

"""
Formats the data for classification with Keras.
"""
def prepare_data_for_keras(data):
    (x_train, y_train), (x_test, y_test) = data
    
    #uses channels_last
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255   

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    shuffle_data((x_train, y_train))
    shuffle_data((x_test, y_test))

    return (x_train, y_train), (x_test, y_test)

"""
Formats the data for classification with my own implementation of neural nets.
"""
def prepare_data_for_tooc(data):
    (x_train, y_train), (x_test, y_test) = data

    #uses channels_last
    x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    shuffle_data((x_train, y_train))
    shuffle_data((x_test, y_test))

    return (x_train, y_train), (x_test, y_test)

"""
Shuffles x_data and y_data accordingly.
"""
def shuffle_data(data):
    x_data, y_data = data

    seed = random.randint(0, 1000)

    np.random.seed(seed)
    np.random.shuffle(x_data)
    np.random.seed(seed)
    np.random.shuffle(y_data)
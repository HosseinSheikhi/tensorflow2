"""
A simple classification using raw TF 2
"""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def load_dataset():
    #  fetch the data set directly
    data_set = tfds.image.HorsesOrHumans()
    image_shape = data_set.info.features['image'].shape
    num_classes = data_set.info.features['label'].num_classes

    #  download prepare and write to disk
    data_set.download_and_prepare()

    #  load data from disk as tf.data.Datasets
    data = data_set.as_dataset()
    train_data_set, test_data_set = data["train"], data["test"]

    X_train, Y_train, X_test, Y_test = [], [], [], []
    #  convert data set to numpy array
    for var in tfds.as_numpy(train_data_set):
        X_train.append(var['image'])
        Y_train.append(var['label'])

    for var in tfds.as_numpy(test_data_set):
        X_test.append(var['image'])
        Y_train.append(var['label'])

    return np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test)

def conv2D(input, filter, stride):
    out = tf.nn.conv2d(input, filter, strides=[1, stride, stride, 1])
    return tf.nn.relu(out)

def maxpool(input, pool_size, stride):
    return tf.nn.max_pool2d(input, ksize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1])

def dense(inputs, weights):
    return tf.nn.relu(tf.matmul(inputs, weights))




X_train, Y_train, X_test, Y_test = load_dataset()
fig = px.imshow(X_train[0])
fig.show()
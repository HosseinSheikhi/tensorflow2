import tensorflow as tf
from tensorflow.keras import backend, datasets
import numpy as np
import matplotlib.pyplot as plt

num_conv1_filters = 32
num_conv2_filters = 64
num_fc1_units = 1024
num_classes = 10

batch_size = 128
epoch = 1
learning_rate = 0.001
step_log = 50

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = np.array(x_train, dtype=np.float64), np.array(x_test, dtype=np.float64)
x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]
x_train, x_test = x_train / 255.0, x_test / 255.0

# convert to tf dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# create a pipeline for faster training
train_ds = train_ds.batch(batch_size).repeat(epoch).shuffle(10000)


def conv2d(input, filter, bias, stride=1):
    x = tf.nn.conv2d(input=input, filters=filter, strides=[1, stride, stride, 1], padding='SAME')
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)


def maxpool2d(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


random_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)

weights = {
    # First convolutional Layer, 5*5*1 kernels, 32 filters
    "weights_conv1": tf.Variable(random_initializer([5, 5, 1, num_conv1_filters], dtype=tf.float64)),
    # Second convolutional Layer, 3*3*32 kernels, 64 filters
    "weights_conv2": tf.Variable(random_initializer([3, 3, num_conv1_filters, num_conv2_filters], dtype=tf.float64)),
    # For the sake of simplicity conv2d padding is set to 'same', it means the output of conv2d layers
    # size is SAME as input. so just maxpool layers will reduce the size each on half the size so 7*7*64
    "weights_fc1": tf.Variable(random_initializer([4 * 4 * 64, num_fc1_units], dtype=tf.float64)),
    "weights_out": tf.Variable(random_initializer([num_fc1_units, num_classes], dtype=tf.float64))
}
biases = {
    "biases_conv1": tf.Variable(tf.zeros(num_conv1_filters, dtype=tf.float64)),
    "biases_conv2": tf.Variable(tf.zeros(num_conv2_filters, dtype=tf.float64)),
    "biases_fc1": tf.Variable(tf.zeros(num_fc1_units, dtype=tf.float64)),
    "biases_out": tf.Variable(tf.zeros(num_classes, dtype=tf.float64))
}


def convolutional_neural_net(input_images_in_batch):
    conv1_out = conv2d(input_images_in_batch, weights['weights_conv1'], biases['biases_conv1'])
    maxpool1_out = maxpool2d(conv1_out)

    conv2_out = conv2d(maxpool1_out, weights['weights_conv2'], biases['biases_conv2'])
    maxpool2_out = maxpool2d(conv2_out, k=4)
    flatten = tf.reshape(maxpool2_out, [-1, weights['weights_fc1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(flatten, weights["weights_fc1"]), biases["biases_fc1"])
    fc1 = tf.nn.relu(fc1)
    out = tf.add(tf.matmul(fc1, weights["weights_out"]), biases["biases_out"])
    return tf.nn.softmax(out)


@tf.function
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.clip_by_value(y_pred, backend.epsilon(), 1)
    return -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# ADAM optimizer.
optimizer = tf.optimizers.Adam(learning_rate)


@tf.function
def train_cnn(x, y):
    with tf.GradientTape() as g:
        prediction = convolutional_neural_net(x)
        loss = cross_entropy(prediction, y)

    trainable_variables = list(weights.values()) + list(biases.values())

    gradients = tf.gradients(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


step = 0
for (x_batch, y_batch) in train_ds:
    train_cnn(x_batch, y_batch)

    step += 1
    if step % step_log == 0:
        pred = convolutional_neural_net(x_batch)
        loss = cross_entropy(pred, y_batch)
        acc = accuracy(pred, y_batch)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))


pred = convolutional_neural_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))
n_images = 5
test_images = x_test[:n_images]
predictions = convolutional_neural_net(test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import Layer
import numpy as np
import matplotlib.pyplot as plt

feature_size = 28 * 28
batch_size = 256
epochs = 10

learning_rate = 0.001
num_hidden_1 = 128
num_hidden_2 = 256
num_classes = 10
display_step = 50

# load dataset - Flatten data set - cast to float32 - normalize
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = np.reshape(x_train, [-1, feature_size]), np.reshape(x_test, [-1, feature_size])
x_train, x_test = np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32)
x_train, x_test = x_train / 255.0, x_test / 255.0

# convert dataset to tf.dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(50000).repeat(epochs).batch(batch_size).prefetch(2)


class Dense(Layer):
    def __init__(self, units):
        super(Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


dense_layer_1 = Dense(num_hidden_1)
dense_layer_2 = Dense(num_hidden_2)
dense_layer_3 = Dense(num_classes)


def neural_net(x):
    layer_1_out = dense_layer_1(x)
    layer_1_out_activated = tf.nn.relu(layer_1_out)

    layer_2_out = dense_layer_2(layer_1_out_activated)
    layer_2_out_activated = tf.nn.relu(layer_2_out)

    out_layer = dense_layer_3(layer_2_out_activated)
    out_prob = tf.nn.softmax(out_layer)

    return out_prob


def cross_entropy(y_pred, y_true):
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-8, clip_value_max=1)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), 1))


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


optimizer = tf.optimizers.SGD(learning_rate)


def train_neural_network(x, y):
    with tf.GradientTape() as g:
        prediction = neural_net(x)
        loss = cross_entropy(prediction, y)

    # variables must train
    trainable_variables = dense_layer_1.trainable_variables + dense_layer_2.trainable_variables + dense_layer_3.trainable_variables

    gradient = g.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradient, trainable_variables))


step = 0
for (x_batch, y_batch) in train_ds:
    train_neural_network(x_batch, y_batch)
    step += 1

    if step % display_step == 0:
        pred = neural_net(x_batch)
        loss = cross_entropy(pred, y_batch)
        acc = accuracy(pred, y_batch)
        print("step: {}  loss: {}  accuracy: {}".format(step, loss, acc))

# Test model on validation set.
pred = neural_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

n_images = 5
test_images = x_test[:n_images]
predictions = neural_net(test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))

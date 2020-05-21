import tensorflow as tf
from tensorflow.keras import datasets, Model, layers
import numpy as np
import matplotlib.pyplot as plt

batch_size = 256
epochs = 100
learning_rate = 0.001
log_step = 50

num_hidden_1 = 128
num_hidden_2 = 256
num_classes = 10
feature_size = 28 * 28

"""
    Load the mnist dataset
    Flatten images
    cast to float32
    normalize images
    convert to tf.data.Dataset pipe line
"""
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = np.reshape(x_train, newshape=[-1, feature_size]), np.reshape(x_test, newshape=[-1, feature_size])
x_train, x_test = np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32)
x_train, x_test = x_train / 255.0, x_test / 255.0

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.batch(batch_size).repeat(epochs).shuffle(60000).prefetch(2)


# Create TF Model
class NeuralNet(Model):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = layers.Dense(num_hidden_1, activation=tf.nn.relu)
        self.fc2 = layers.Dense(num_hidden_2, activation=tf.nn.relu)
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(fc1_out)
        out = self.out(fc2_out)

        if not is_training:
            # tf cross entropy expect logits without softmax, so only apply softmax when not training
            out = tf.nn.softmax(out)

        return out


neural_net = NeuralNet()


def cross_entropy_loss(pred_y, true_y):
    true_y = tf.cast(true_y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_y, logits=pred_y)
    # Average Loss on batch
    return tf.reduce_mean(loss)


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


optimizer = tf.optimizers.SGD(learning_rate)


def train_neural_network(x, y):
    with tf.GradientTape() as g:
        prediction = neural_net(x, is_training=True)
        loss = cross_entropy_loss(prediction, y)

    trainable_variables = neural_net.trainable_variables

    gradients = g.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))


step = 0
for (x_batch, y_batch) in train_ds:
    train_neural_network(x_batch, y_batch)

    step += 1
    if step % log_step == 0:
        pred = neural_net(x_batch, is_training=True)
        loss = cross_entropy_loss(pred, y_batch)
        acc = accuracy(pred, y_batch)
        print("step: {}  loss: {}  acc: {} ".format(step, loss, acc))

prediction = neural_net(x_test, is_training=False)
print("Total Acc = {}".format(accuracy(prediction, y_test)))

n_images = 5
test_images = x_test[:n_images]
predictions = neural_net(test_images)

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i],[28,28]), cmap='gray')
    plt.show()
    print("Prediction: {}".format(np.argmax(predictions.numpy()[i])))

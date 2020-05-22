import tensorflow as tf
from tensorflow.keras import Model, layers, datasets
import numpy as np
import matplotlib.pyplot as plt

num_classes = 10
epoch = 2
batch_size = 128
conv_1_filters = 32
conv_2_filters = 64
fc_layer = 1024
learning_rate = 0.001
log_step = 50

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#  Add a channel dimension
x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.batch(batch_size).repeat(epoch).shuffle(10000)


class ConvolutionNeuralNet(Model):
    def __init__(self):
        super(ConvolutionNeuralNet, self).__init__()

        self.conv1 = layers.Conv2D(filters=conv_1_filters, input_shape=(None, 28, 28, 1), kernel_size=5, padding='same',
                                   activation='relu')
        self.maxpol1 = layers.MaxPool2D(pool_size=(4, 4), strides=4)
        self.conv2 = layers.Conv2D(filters=conv_2_filters, kernel_size=3, padding='same', activation='relu')
        self.maxpol2 = layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=fc_layer, activation='relu')
        self.out = layers.Dense(units=num_classes)  # activation is not defined on purpose

    def call(self, inputs, training=False, mask=None):
        x = tf.reshape(inputs, shape=[-1, 28, 28, 1])
        conv1_out = self.conv1(x)
        maxpol1_out = self.maxpol1(conv1_out)
        conv2_out = self.conv2(maxpol1_out)
        maxpol2_out = self.maxpol2(conv2_out)
        flatten_out = self.flatten(maxpol2_out)
        fc1_out = self.fc1(flatten_out)
        out = self.out(fc1_out)
        if not training:
            out = tf.nn.softmax(out)

        return out


conv_net = ConvolutionNeuralNet()

@tf.function
def cross_entropy_loss(pred_y, true_y):
    true_y = tf.cast(true_y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_y,
                                                          logits=pred_y)  # it doesn't net one hot encoding
    return tf.reduce_mean(loss)


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


optimizer = tf.keras.optimizers.Adam(learning_rate)

@tf.function
def train_neural_network(x, y):
    with tf.GradientTape() as g:
        prediction = conv_net(x, training=True)
        loss = cross_entropy_loss(prediction, y)

    trainable_variables = conv_net.trainable_variables

    gradients = g.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))


step = 0
for (x_batch, y_batch) in train_ds:
    train_neural_network(x_batch, y_batch)

    step += 1
    if step % log_step == 0:
        pred = conv_net(x_batch, training=True)
        loss = cross_entropy_loss(pred, y_batch)
        acc = accuracy(pred, y_batch)
        print("step: {}  loss: {}  acc: {} ".format(step, loss, acc))


print(conv_net.summary())
prediction = conv_net(x_test, training=False)
print("Total Acc = {}".format(accuracy(prediction, y_test)))

n_images = 5
test_images = x_test[:n_images]
predictions = conv_net(test_images)

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Prediction: {}".format(np.argmax(predictions.numpy()[i])))

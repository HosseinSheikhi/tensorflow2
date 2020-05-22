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
train_ds = train_ds.batch(batch_size).shuffle(10000)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))


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
        return out


conv_net = ConvolutionNeuralNet()

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_neural_network(x, y):
    with tf.GradientTape() as g:
        prediction = conv_net(x, training=True)
        loss = loss_function(y, prediction)

    trainable_variables = conv_net.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    train_loss(loss)
    train_acc(y, prediction)


@tf.function
def test_neural_network(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = conv_net(images, training=False)
    t_loss = loss_function(labels, predictions)

    test_loss(t_loss)
    test_acc(labels, predictions)


for repeat in range(epoch):
    # reset matrices at the start of each epoch
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    test_acc.reset_states()

    for (x_batch, y_batch) in train_ds:
        train_neural_network(x_batch, y_batch)

    for (x_test, y_test) in test_ds:
        test_neural_network(x_test, y_test)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(repeat + 1,
                          train_loss.result(),
                          train_acc.result() * 100,
                          test_loss.result(),
                          test_acc.result() * 100))

print(conv_net.summary())
n_images = 5
test_images = x_test[:n_images]
predictions = conv_net(test_images)

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Prediction: {}".format(np.argmax(predictions.numpy()[i])))

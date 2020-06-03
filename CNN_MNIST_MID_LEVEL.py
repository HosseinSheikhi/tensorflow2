import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import datasets, Model
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorboard

batch_size = 128
epoch = 1
num_cnn_filter_1 = 64
num_cnn_filter_2 = 32
fc_units_1 = 512
num_classes = 10
learning_rate = 0.001

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size).repeat(epoch)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(512)


class CnnLayer(Layer):
    def __init__(self, kernel_size, units, stride=1):
        super(CnnLayer, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.stride = [1, stride, stride, 1]

    def build(self, input_shape):
        temp = input_shape
        self.w = self.add_weight(shape=([self.kernel_size, self.kernel_size, input_shape[-1], self.units]),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        x = tf.nn.conv2d(input=inputs, filters=self.w, strides=self.stride, padding='SAME')
        x = tf.nn.bias_add(x, self.b)
        return tf.nn.relu(x)


class Dense(Layer):
    def __init__(self, units):
        super(Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class ConvolutionalNetwork(Model):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.cnn_layer_1 = CnnLayer(3, num_cnn_filter_1)
        self.cnn_layer_2 = CnnLayer(3, num_cnn_filter_2)
        self.fc_layer_1 = Dense(fc_units_1)
        self.fc_layer_2 = Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        x = self.cnn_layer_1(inputs)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        x = self.cnn_layer_2(x)
        x = tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        x = tf.reshape(x, shape=(-1, 4 * 4 * num_cnn_filter_2))
        x = self.fc_layer_1(x)
        x = tf.nn.relu(x)

        x = self.fc_layer_2(x)
        return x


conv_net = ConvolutionalNetwork()

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
@tf.function
def train_neural_network(x, y):
    with tf.GradientTape() as g:
        prediction = conv_net(x)
        loss = loss_function(y, prediction)

    trainable_variables = conv_net.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    train_loss(loss)
    train_acc(y, prediction)


@tf.function
def test_neural_network(images, labels):
    predictions = conv_net(images)
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
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_acc.result(), step=epoch)

    for (x_test, y_test) in test_ds:
        test_neural_network(x_test, y_test)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_acc.result(), step=epoch)

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

tensorboard


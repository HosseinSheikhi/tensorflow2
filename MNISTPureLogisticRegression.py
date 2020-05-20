import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt

num_classes = 10
num_features = 28 * 28

learning_rate = 0.001
epochs = 100
batch_size = 3
display_step = 50

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train, x_test = np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32)

# flatten images
x_train, x_test = np.reshape(x_train, [-1, num_features]), np.reshape(x_test, [-1, num_features])
x_train, x_test = x_train / 255.0, x_test / 255.0

# use TF.data.Dataset to get a fast pipeline
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat(epochs).shuffle(50000).batch(batch_size).prefetch(2)

W = tf.Variable(tf.ones([num_features, num_classes]), name='weights')
b = tf.Variable(tf.zeros([num_classes]), name='biases')


def logistic_regression(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)


# Cross entropy loss
def cross_entropy(y_pred, y_true):
    # encode labels to one-hot vector
    y_true = tf.one_hot(y_true, depth=num_classes)
    # clip prediction values to avoid log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-8, 1)
    # compute cross entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), 1))


# Accuracy matrix
def accuracy(y_pred, y_true):
    prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(prediction, tf.float32))


optimizer = tf.optimizers.SGD(learning_rate)


def run_optimization(x, y):
    with tf.GradientTape() as g:
        predictions = logistic_regression(x)
        loss = cross_entropy(predictions, y)

    # compute gradients
    gradients = g.gradient(loss, [W, b])

    optimizer.apply_gradients(zip(gradients, [W, b]))


# Run training for the given number of steps.
step = 0
for (batch_x, batch_y) in train_data:
    # print("---First Training Example---")
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    step += 1
    if step % display_step == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# Test model on validation set.
pred = logistic_regression(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# Predict 5 images from validation set.
n_images = 5
test_images = x_test[:n_images]
predictions = logistic_regression(test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

# parameters
learning_rate = 0.001
training_steps = 1000
display_step = 100

# Training Data.
X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
              7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
              2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

# Weight and Bias, Initialized Randomly
W = tf.Variable(initial_value=rng.randn(), trainable=True, name='weights')
b = tf.Variable(initial_value=rng.randn(), trainable=True, name='bias')


# Linear Regression (Wx+b)
def linear_regression(x):
    return W*x+b


# Mean Squared Error
def mean_square(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))


optimizer = tf.optimizers.SGD(learning_rate)


# Optimization process
def run_optimization():
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)

    # compute gradients
    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


for step in range(training_steps):
    run_optimization()

    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print(loss)

plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, np.array(W * X + b), label='Fitted Line')
plt.legend()
plt.show()

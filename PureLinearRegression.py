import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# parameters
learning_rate = 0.001  # alpha in gradient descent algorithm
epoch = 500  # How many time iterate on the whole data
log_counter = 100  # Log data every log_counter epoch

# Generate (x,y) data to predict
X = np.arange(-5, 5, 0.5) + 2
Y = X / 2 + 0.3 * np.random.standard_normal(size=(len(X),))  # Y = X/2 + some_noise

# Define and initialize Weight (or slope) and Bias (or intercept)
random_initializer = tf.random_normal_initializer(mean=0, stddev=0.1)
W = tf.Variable(initial_value=random_initializer([1], dtype=tf.float64), trainable=True)
b = tf.Variable(initial_value=random_initializer([1], dtype=tf.float64), trainable=True)


#  Linear regression is nothing just fit a line ( y_hat = W*x + b) by least possible error
def linear_regression(x):
    return W * x + b


# define mean square error to calculate loss function
@tf.function
def mean_square(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))  # make sure y_pred, y_true are in same data type


optimizer = tf.optimizers.SGD(learning_rate)


# fit a line by linear regression algorithm
# calculate how much the line is a good fit
# calculate gradients of objective (MSE) w.r.t weight and bias
# use SGD optimizer to apply the gradient descent algorithm on weight and bias
@tf.function
def run_optimization():
    with tf.GradientTape() as g:
        y_hat = linear_regression(X)
        loss = mean_square(y_hat, Y)

    # compute gradients
    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


for step in range(epoch):
    run_optimization()

    if (step+1) % log_counter == 0:
        prediction = linear_regression(X)
        loss = mean_square(prediction, Y)
        print("epoch: {}  loss:{}".format(step+1, loss.numpy()))


plt.scatter(X, Y, label='Original data')
plt.plot(X, prediction.numpy(), color='red', label='Fitted Line')
plt.legend()
plt.show()

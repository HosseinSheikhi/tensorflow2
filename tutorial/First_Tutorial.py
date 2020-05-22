import tensorflow as tf
import numpy as np

"""
Lets play by two type of widely use tensors
1- constants: as its names show you cannot change values in a constant
2- variables
"""

const_1 = tf.constant([1, 2, 3])
print(const_1)

const_2 = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float64)
print(const_2)

var_1 = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float64)
print(var_1)

"""
Having a tensor you also have a numpy array and vice versa
"""

first_arr = np.array([[1, 2, 3], [4, 5, 6]])
const_3 = tf.constant(first_arr)  # converting numpy array to a tensor
print(const_3)  # dtype=int32

second_arr = var_1.numpy()  # converting a tensor to a numpy array
print(second_arr)

"""
Some times we have to cast our tensors to other data types
"""
const_4 = tf.cast(const_3, tf.float16)
print(const_4)  # dtype=float16


"""
Time to get familiar by a few operators on tensors
"""

a = tf.Variable([[1, 2, 3], [4, 5, 6]])
b = tf.Variable([[5, 2], [7, 1], [0, 9]])
print(tf.matmul(a, b).numpy())  # MatMul =MatrixMultiplication
print(tf.multiply(a, 2).numpy())  # multiply = element wise multiplication


c = tf.Variable([[3], [-1]])
print(tf.multiply(a, c).numpy())  # I need your attention here. See how c propagates
print((a*c).numpy())  # tf.multiply(a, c) = a * c

print(tf.add(a, c).numpy())  # propagation





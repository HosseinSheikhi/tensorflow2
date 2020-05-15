"""
For the sake of simplicity tensorflow 2 takes advantage of EAGER EXECUTION
There is no longer the need to first statistically define a computational graph and then execute it.
By eager execution all the models can be dynamically defined and immediately executed.
----------------------------------------------------------------------------------------
AutoGraph:
TF2 natively supports imperative python code, including control flow such as if-while-print() and other native python
feature and CAN NATIVELY CONVERT IT TO TF GRAPH CODE

Auto Graph takes eager style Python code and automatically converts it to graph-generating code

How to use autograph? annotate your python code with the special detector tf.function
"""

import tensorflow as tf
import numpy as np
import time

tf.nn.relu(5)


def linear_layer(x):
    return 3 * x + 2


def simple_nn(x):
    return tf.nn.relu(linear_layer(x))


@tf.function
def modified_nn(x):
    return tf.nn.relu(linear_layer(x))


input_array = np.random.rand(50)

t1 = time.process_time()
without_ag = simple_nn(input_array)
t2 = time.process_time()
with_ag = modified_nn(input_array)
t3 = time.process_time()

print(t2 - t1)
print(t3 - t2)
print(without_ag)
print(with_ag)

"""
----------------------------------------------------------------------
in tensorflow 1.x we need to build a blue print of whatever NN we want
So first we have to define a computational graph and then execute it
----------------------------------------------------------------------
Computational Graph (or CG):
is a network of nodes and edges
All the data which would be used and all the computations to be performed are defined.
Node represents object (either var or operations)
----------------------------------------------------------------------
Placeholder:
A variable which we need to define to complete our computational graph and will be assigned later in execution
----------------------------------------------------------------------
Execution of CG:
is performed using the session object
When session is called the tensor objects which  were abstract (blue print) till now will come to life
Session is a place where actual calculations and transform of information from one layer to another take place
-----------------------------------------------------------------------
fid_dict:
is used to fed values to placeholders
"""

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
"""
ADD TWO VECTORS
"""
# define three nodes (two variable and one operation)
vec_1 = tf.constant([1, 2, 3, 4, 5])
vec_2 = tf.constant([-1, 4, 0, 3, -2])
vec_add = tf.add(vec_1, vec_2)

sess = tf.Session()
print(sess.run(vec_add))
sess.close()

"""
Multiply two matrix
"""
mat_1 = tf.random_normal([3, 2], mean=5, stddev=2, seed=5)
mat_2 = tf.constant([[7., 7., 7.], [1., 2., 3.]])
mat_mul = tf.matmul(mat_1, mat_2)

sess = tf.Session()
print(sess.run(mat_mul))
sess.close()

"""
placeholder example
"""
x = tf.placeholder('float')
y = tf.multiply(2., x)
data = tf.random_uniform([4, 5],3)
sess = tf.Session()
x_data = sess.run(data)
print(sess.run(y, feed_dict={x: x_data}))
sess.close()

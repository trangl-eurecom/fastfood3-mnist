import tensorflow as tf
import numpy as np
"""
Test the code:
x_tensors = tf.transpose(featuresCNN) #[Dfeatures, batch_size]
layer_times_Omega = self.fastfood_module.fast_food2(x_tensors, self.B[0], self.P[0], self.G[0], self.S[0]) #[Dfeatures, batch_size]
layer_times_Omega = tf.transpose(layer_times_Omega) #[batch_size, Dfeatures]
layer_times_Omega = tf.expand_dims(layer_times_Omega, 0)
layer_times_Omega = tf.tile(layer_times_Omega, [self.mc, 1, 1])

a1 = tf.constant([[1, 2, 3], [4, 5, 6]])
a2 = tf.expand_dims(a1, 0)
a3 = tf.tile(a2, [3, 1, 1])

with tf.Session() as session:
    b3 = session.run(a3)
    print(b3)
"""

"""
Test the code
x_tensors = tf.reshape(self.layer[i], [self.mc * batch_size, -1])
x_tensors = tf.transpose(x_tensors)
layer_times_Omega = self.fastfood_module.fast_food2(x_tensors, self.B[0], self.P[0], self.G[0], self.S[0]) #[-1, batch_size * mc]
layer_times_Omega = tf.transpose(layer_times_Omega) #[batch_size * mc, -1]
layer_times_Omega = tf.reshape(layer_times_Omega, [self.mc, batch_size, -1])

mc = 3
bs = 2;
D = 4;
a1 = tf.constant([[[1, 2, 3, 4], [4, 5, 6, 7]], [[7, 8, 9, 0], [0, 1, 2, 3]], [[3, 4, 5, 6], [6, 7, 8, 9]]])
a2 = tf.reshape(a1, [mc * bs, -1])
a3 = tf.transpose(a2)
a4 = tf.transpose(a3)
a5 = tf.reshape(a4, [mc, bs, -1])
with tf.Session() as session:
    b1, b5 = session.run([a1, a5])
    print(b1)
    print(b5)
"""
a1 = tf.random_normal([2, 1])
with tf.Session() as session:
    b1 = session.run(a1)
    print(b1)
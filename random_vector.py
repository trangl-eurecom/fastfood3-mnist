import tensorflow as tf
import numpy as np

# This function will yield the randomed binary scaling vector B in Fastfood
# Input:
#  + d: a scalar: size of vector B
# Output:
#  + B: [d, 1]: a randomed binary scaling vector whose diagonal element is sampled from {-1, +1}
def create_binary_scaling_vector(d):
    r_u = tf.random_uniform([1, d], minval=0, maxval=1.0, dtype=tf.float32)
    ones = tf.ones([1, d])
    means = tf.multiply(0.5, ones)
    B = tf.where(r_u > means, ones, tf.multiply(-1.0, ones))
    return tf.reshape(B, [d, 1])

# This function will yield the permutation vector in Fastfood
# Input:
#  + d: a scalar: size of vector pi
# Output:
#  + pi: [d, 1]: a permutation vector of [0, d-1]
def create_permutation(d):
    pi = tf.reshape(tf.range(d), [d, 1])
    pi = tf.random_shuffle(pi)
    return tf.cast(pi, tf.float32)

# This function will yield the gaussian scaling vector G in Fastfood
# Input:
#  + d: a scalar: size of vector G
# Output:
#  + G: [d, 1]: a gaussian vector
def create_gaussian_scaling_vector(d):
    return tf.random_normal([d, 1])

# This function will yield the s in Fastfood
# For RBF,  I assume that s should be followed chi distribution with d degree of freedom
# Input:
#  + d: a scalar: size of vector s
# Output:
#  + s: [d, 1]: a scaling vector
def create_chi_vector(d):
    s = tf.random_normal([d, d])
    s = tf.square(s)
    s = tf.reduce_sum(s, axis = 0)
    s = tf.sqrt(s)
    return tf.reshape(s, [d, 1])
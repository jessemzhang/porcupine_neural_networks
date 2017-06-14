# Architectures.
# Input will be a dictionary of required arguments (dimensions of certain tensors)
# Output will be a dictionary of access to graph tensors

import tensorflow as tf
import numpy as np

# Standard 3-layer mlp
def mlp(input_dict):
    p = input_dict['p']
    h = input_dict['h']
    mlp_in = tf.placeholder(tf.float32, shape=[None, p])
    weights1 = tf.get_variable('w1',initializer=tf.random_normal(shape=[p,h],stddev=1./np.sqrt(p)))
    mlp_hidden = tf.nn.relu(tf.matmul(mlp_in, weights1), name='mlp_hidden')
    weights2 = tf.get_variable('w2',initializer=tf.random_normal(shape=[h,1],stddev=1./np.sqrt(h)))
    mlp_out = tf.nn.relu(tf.matmul(mlp_hidden, weights2), name='mlp_output')
    return dict(x=mlp_in,yhat=mlp_out,hid=mlp_hidden,weights1=weights1,weights2=weights2)


# Simple mlp with one neuron and two inputs
def mlp_simple(input_dict):
    # 2-layer net. out = relu(<weights,mlp_in>)
    p = input_dict['p']
    mlp_in = tf.placeholder(tf.float32, shape=[None, p])
    weights = tf.get_variable('w',initializer=tf.random_normal(shape=[p,1],
                                                               stddev=1./np.sqrt(p)))
    mlp_out = tf.nn.relu(tf.matmul(mlp_in, weights), name='mlp_out')
    return dict(x=mlp_in,yhat=mlp_out,weights=weights)


# mlp with no relu at last layer
def mlp_noreluout(input_dict):
    p = input_dict['p']
    h = input_dict['h']
    mlp_in = tf.placeholder(tf.float32, shape=[None, p])
    weights1 = tf.get_variable('w1',initializer=tf.random_normal(shape=[p,h],
                                                                 stddev=1./np.sqrt(p)))
    mlp_hidden = tf.nn.relu(tf.matmul(mlp_in, weights1), name='mlp_hidden')
    weights2 = tf.get_variable('w2',initializer=tf.random_normal(shape=[h,1],
                                                                 stddev=1./np.sqrt(h)))
    mlp_out = tf.matmul(mlp_hidden, weights2)
    return dict(x=mlp_in,yhat=mlp_out,hid=mlp_hidden,weights1=weights1,weights2=weights2)


# mlp with no relu at last layer, and last layer weights not trainable
def mlp_noreluout_lastlayernottrainable(input_dict):
    p = input_dict['p']
    h = input_dict['h']
    mlp_in = tf.placeholder(tf.float32, shape=[None, p])
    weights1 = tf.get_variable('w1',initializer=tf.random_normal(shape=[p,h],
                                                                 stddev=1./np.sqrt(p)))
    mlp_hidden = tf.nn.relu(tf.matmul(mlp_in, weights1), name='mlp_hidden')
    weights2 = tf.get_variable('w2',initializer=tf.random_normal(shape=[h,1],
                                                                 stddev=1./np.sqrt(h)),trainable=False)
    mlp_out = tf.matmul(mlp_hidden, weights2)
    return dict(x=mlp_in,yhat=mlp_out,hid=mlp_hidden,weights1=weights1,weights2=weights2)


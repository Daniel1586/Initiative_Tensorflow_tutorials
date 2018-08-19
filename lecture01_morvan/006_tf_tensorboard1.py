#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_func=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_func is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_func(Wx_plus_b)
        return outputs


x_data = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, activation_func=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_func=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(ys - prediction), reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
writer = tf.summary.FileWriter('./log', graph=sess.graph)
init = tf.initialize_all_variables()

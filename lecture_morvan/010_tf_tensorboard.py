#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# 通过tf.set_random_seed设定种子数,后面定义的全部变量都可以跨会话生成相同的随机数
tf.set_random_seed(1)
np.random.seed(1)

xdata = np.linspace(-1, 1, 100)[:, np.newaxis]              # shape (100, 1)
noise = np.random.normal(0, 0.1, size=xdata.shape)
ydata = np.power(xdata, 2) + noise                          # shape (100, 1) + some noise

with tf.variable_scope('Inputs'):
    tf_x = tf.placeholder(tf.float32, xdata.shape, name='x')
    tf_y = tf.placeholder(tf.float32, ydata.shape, name='y')

with tf.variable_scope('Net'):
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu, name='hidden_layer')
    lo = tf.layers.dense(l1, 1, name='output_layer')

    # add to histogram summary
    tf.summary.histogram('hid_out', l1)
    tf.summary.histogram('pre_out', lo)

loss = tf.losses.mean_squared_error(tf_y, lo, scope='loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
tf.summary.scalar('loss', loss)         # add loss to scalar summary

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('log/', graph=sess.graph)     # write to file
merge_op = tf.summary.merge_all()                       # operation to merge all summary
for step in range(100):
    _, result = sess.run([train_op, merge_op], {tf_x: xdata, tf_y: ydata})
    writer.add_summary(result, step)

#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 通过tf.set_random_seed设定种子数,后面定义的全部变量都可以跨会话生成相同的随机数
tf.set_random_seed(1)
np.random.seed(1)

xdata = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=xdata.shape)
ydata = np.power(xdata, 2) + noise                      # shape (100, 1) + some noise

tf_x = tf.placeholder(tf.float32, xdata.shape)     # input x
tf_y = tf.placeholder(tf.float32, ydata.shape)     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer
lo = tf.layers.dense(l1, 1)                         # output layer

loss = tf.losses.mean_squared_error(tf_y, lo)       # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

plt.ion()   # something about plotting
for step in range(100):
    _, l, pred = sess.run([train_op, loss, lo], {tf_x: xdata, tf_y: ydata})
    if step % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(xdata, ydata)
        plt.plot(xdata, pred, 'r-', lw=5)
        plt.text(-0.25, 0.6, 'Loss=%.4f' % l, fontdict={'size': 18, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()

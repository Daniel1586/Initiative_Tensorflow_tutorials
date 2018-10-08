#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 通过tf.set_random_seed设定种子数,后面定义的全部变量都可以跨会话生成相同的随机数
tf.set_random_seed(1)
np.random.seed(1)

print('========== 1.Generating data...')
xdata = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=xdata.shape)
print(noise)
ydata = np.power(xdata, 2) + noise                      # shape (100, 1) + some noise


def save():
    tf_x = tf.placeholder(tf.float32, xdata.shape)      # input x
    tf_y = tf.placeholder(tf.float32, ydata.shape)      # input y
    hd_l = tf.layers.dense(tf_x, 10, tf.nn.relu)        # hidden layer
    ou_l = tf.layers.dense(hd_l, 1)                     # output layer
    loss = tf.losses.mean_squared_error(tf_y, ou_l)     # compute cost
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())         # initialize var in graph
    saver = tf.train.Saver()    # define a saver for saving and restoring

    for step in range(100):
        sess.run(train_op, {tf_x: xdata, tf_y: ydata})
    saver.save(sess, 'model/params', write_meta_graph=False)  # meta_graph is not recommended

    pred_o, pred_l = sess.run([ou_l, loss], {tf_x: xdata, tf_y: ydata})
    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(xdata, ydata)
    plt.plot(xdata, pred_o, 'r-', lw=5)
    plt.text(-0.8, 1.1, 'Save Loss=%.4f' % pred_l, fontdict={'size': 15, 'color': 'red'})


def reload():
    tf_x = tf.placeholder(tf.float32, xdata.shape)      # input x
    tf_y = tf.placeholder(tf.float32, ydata.shape)      # input y
    hd_l = tf.layers.dense(tf_x, 10, tf.nn.relu)        # hidden layer
    ou_l = tf.layers.dense(hd_l, 1)                     # output layer
    loss_ = tf.losses.mean_squared_error(tf_y, ou_l)    # compute cost

    sess = tf.Session()
    saver = tf.train.Saver()    # define a saver for saving and restoring
    saver.restore(sess, 'model/params')

    pred_o, pred_l = sess.run([ou_l, loss_], {tf_x: xdata, tf_y: ydata})
    plt.subplot(122)
    plt.scatter(xdata, ydata)
    plt.plot(xdata, pred_o, 'r-', lw=5)
    plt.text(-0.8, 1.1, 'Reload Loss=%.4f' % pred_l, fontdict={'size': 15, 'color': 'red'})
    plt.show()


print('========== 2.Building model and save model params...')
save()
tf.reset_default_graph()    # 用于清除默认图形堆栈并重置全局默认图形
print('========== 3.Loading model params ...')
reload()

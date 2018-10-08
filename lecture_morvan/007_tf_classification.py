#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 通过tf.set_random_seed设定种子数,后面定义的全部变量都可以跨会话生成相同的随机数
tf.set_random_seed(1)
np.random.seed(1)

n_data = np.ones((100, 2))
x0 = np.random.normal(2*n_data, 1)      # class0 x shape=(100, 2)
y0 = np.zeros(100)                      # class0 y shape=(100, 1)
x1 = np.random.normal(-2*n_data, 1)     # class1 x shape=(100, 2)
y1 = np.ones(100)                       # class1 y shape=(100, 1)
x = np.vstack((x0, x1))                 # shape (200, 2) + some noise 垂直(按行)把数组堆叠
y = np.hstack((y0, y1))                 # shape (200, ) 水平(按列)把数组堆叠

plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)      # input x
tf_y = tf.placeholder(tf.int32, y.shape)        # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)      # hidden layer
lo = tf.layers.dense(l1, 2)                     # output layer

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=lo)       # compute cost
accuracy = tf.metrics.accuracy(labels=tf.squeeze(tf_y), predictions=tf.argmax(lo, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

plt.ion()
for step in range(100):
    _, acc, pred = sess.run([train_op, accuracy, lo], {tf_x: x, tf_y: y})
    if step % 2 == 0:
        plt.cla()
        plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
        plt.text(-4.8, 2, 'Accuracy=%.4f' % acc, fontdict={'size': 18, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()

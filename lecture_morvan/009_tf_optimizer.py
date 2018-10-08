#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 通过tf.set_random_seed设定种子数,后面定义的全部变量都可以跨会话生成相同的随机数
tf.set_random_seed(1)
np.random.seed(1)

learn_rate = 0.01
batch_size = 64

xdata = np.linspace(-1, 1, 100)[:, np.newaxis]              # shape (100, 1)
noise = np.random.normal(0, 0.1, size=xdata.shape)
ydata = np.power(xdata, 2) + noise                          # shape (100, 1) + some noise

plt.scatter(xdata, ydata)
plt.show()


# default network
class Net:
    def __init__(self, opt, **kwargs):
        self.x = tf.placeholder(tf.float32, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        l1 = tf.layers.dense(self.x, 20, tf.nn.relu)
        out = tf.layers.dense(l1, 1)
        self.loss = tf.losses.mean_squared_error(self.y, out)
        self.train = opt(learn_rate, **kwargs).minimize(self.loss)


# different nets
net_SGD = Net(tf.train.GradientDescentOptimizer)
net_Momentum = Net(tf.train.MomentumOptimizer, momentum=0.9)
net_RMSprop = Net(tf.train.RMSPropOptimizer)
net_Adam = Net(tf.train.AdamOptimizer)
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
losses_ = [[], [], [], []]

# training
for step in range(300):          # for each training step
    index = np.random.randint(0, xdata.shape[0], batch_size)
    b_x = xdata[index]
    b_y = ydata[index]

    for net, l_ in zip(nets, losses_):
        _, l2 = sess.run([net.train, net.loss], {net.x: b_x, net.y: b_y})
        l_.append(l2)     # loss recoder

# plot loss history
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.3))
plt.show()

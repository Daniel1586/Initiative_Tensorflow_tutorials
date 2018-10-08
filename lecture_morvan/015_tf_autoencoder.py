#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.examples.tutorials.mnist import input_data

# 通过tf.set_random_seed设定种子数,后面定义的全部变量都可以跨会话生成相同的随机数
tf.set_random_seed(1)

print('========== 1.Setting Hyper Parameters...')
batch_size = 64
learn_rate = 0.002
n_test_img = 5

print('========== 2.Loading data...')
mnist = input_data.read_data_sets('./mnist', one_hot=False)     # use not one-hotted target data
test_x = mnist.test.images[:200]
test_y = mnist.test.labels[:200]

print('----- x shape: ', mnist.train.images.shape)      # (55000, 28 * 28)
print('----- y shape: ', mnist.train.labels.shape)      # (55000, 10)
plt.imshow(mnist.train.images[1].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[1]))
plt.show()

print('========== 3.Building Network...')
tf_x = tf.placeholder(tf.float32, [None, 28*28])        # value in the range of (0, 1)
en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
en2 = tf.layers.dense(en1, 12, tf.nn.tanh)
enc = tf.layers.dense(en2, 3)

de0 = tf.layers.dense(enc, 12, tf.nn.tanh)
de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
dec = tf.layers.dense(de2, 28*28, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=dec)
train = tf.train.AdamOptimizer(learn_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# initialize figure
f, a = plt.subplots(2, n_test_img, figsize=(5, 2))
plt.ion()

# original data (first row) for viewing
view_data = mnist.test.images[:n_test_img]
for i in range(n_test_img):
    a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for step in range(8000):
    b_x, b_y = mnist.train.next_batch(batch_size)
    _, encoded_, decoded_, loss_ = sess.run([train, enc, dec, loss], {tf_x: b_x})

    if step % 100 == 0:
        print('----- train loss: %.4f' % loss_)
        # plotting decoded image (second row)
        decoded_data = sess.run(dec, {tf_x: view_data})
        for i in range(n_test_img):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.draw()
        plt.pause(0.1)
plt.ioff()

# visualize in 3D plot
view_data = test_x[:200]
encoded_data = sess.run(enc, {tf_x: view_data})
fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2]
for x, y, z, s in zip(X, Y, Z, test_y):
    c = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()

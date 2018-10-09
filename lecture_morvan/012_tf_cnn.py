#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from tensorflow.examples.tutorials.mnist import input_data

# 通过tf.set_random_seed设定种子数,后面定义的全部变量都可以跨会话生成相同的随机数
tf.set_random_seed(1)
np.random.seed(1)

learn_rate = 0.001
batch_size = 64

print('========== 1.Loading data...')
mnist = input_data.read_data_sets('./mnist', one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

print('----- x shape: ', mnist.train.images.shape)      # (55000, 28 * 28)
print('----- y shape: ', mnist.train.labels.shape)      # (55000, 10)
plt.imshow(mnist.train.images[1].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[1]))
plt.show()

tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255
tf_y = tf.placeholder(tf.int32, [None, 10])            # input y
image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)

print('========== 2.Building Network...')
conv1 = tf.layers.conv2d(inputs=image, filters=16, kernel_size=5,
                         strides=1, padding='same', activation=tf.nn.relu)      # -> (28, 28, 16)
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)                  # -> (14, 14, 16)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)        # -> (14, 14, 32)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)                                    # -> (7, 7, 32)
flat = tf.reshape(pool2, [-1, 7*7*32])                                          # -> (7*7*32, )
output = tf.layers.dense(flat, 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)       # compute cost
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)


def plot_with_labels(low_dim, lab):
    plt.cla()
    x_, y_ = low_dim[:, 0], low_dim[:, 1]
    for x, y, s in zip(x_, y_, lab):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(x_.min()-5, x_.max()+5)
    plt.ylim(y_.min()-5, y_.max()+5)
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)


print('========== 3.Visualizing Network...')
plt.ion()   # 打开交互模式
for step in range(800):
    b_x, b_y = mnist.train.next_batch(batch_size)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 100 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

        # Visualization of trained flatten layer (T-SNE)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 300
        low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
        labels = np.argmax(test_y, axis=1)[:plot_only]
        plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_o = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_o, 1)
print('----- pred number: ', pred_y)
print('----- real number: ', np.argmax(test_y[:10], 1))

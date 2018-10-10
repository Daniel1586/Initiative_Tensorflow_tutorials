#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 通过tf.set_random_seed设定种子数,后面定义的全部变量都可以跨会话生成相同的随机数
tf.set_random_seed(1)
np.random.seed(1)

print('========== 1.Setting Hyper Parameters...')
input_size = 28         # rnn input size / image width
times_step = 28         # rnn time step / image height
batch_size = 64
learn_rate = 0.01

print('========== 2.Loading data...')
mnist = input_data.read_data_sets('mnist', one_hot=True)    # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

print('----- x shape: ', mnist.train.images.shape)      # (55000, 28 * 28)
print('----- y shape: ', mnist.train.labels.shape)      # (55000, 10)
plt.imshow(mnist.train.images[1].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[1]))
plt.show()

tf_x = tf.placeholder(tf.float32, [None, times_step * input_size])      # shape(batch, 784)
tf_y = tf.placeholder(tf.int32, [None, 10])                             # input y
image = tf.reshape(tf_x, [-1, times_step, input_size])                  # (batch, height, width)

print('========== 3.Building Network...')
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    image,                      # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
output = tf.layers.dense(outputs[:, -1, :], 10)              # output based on the last output step

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
for step in range(1200):
    b_x, b_y = mnist.train.next_batch(batch_size)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 100 == 0:
        accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print('----- pred number: ', pred_y)
print('----- real number: ', np.argmax(test_y[:10], 1))

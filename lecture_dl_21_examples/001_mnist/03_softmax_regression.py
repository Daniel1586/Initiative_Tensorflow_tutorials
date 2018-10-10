#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取MNIST数据,数据不存在时,会自动执行下载
mnist = input_data.read_data_sets("mnist/", one_hot=True)

# x/y_数据节点(占位符操作),w/b存储节点(模型参数)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print('start training...')

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

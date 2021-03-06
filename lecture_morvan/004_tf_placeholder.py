#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

# 定义数据节点
x1 = tf.placeholder(dtype=tf.float32, shape=None)
y1 = tf.placeholder(dtype=tf.float32, shape=None)
z1 = x1 + y1

x2 = tf.placeholder(dtype=tf.float32, shape=[2, 1])
y2 = tf.placeholder(dtype=tf.float32, shape=[1, 2])
z2 = tf.matmul(x2, y2)

with tf.Session() as sess:
    # run one operation
    z1_value = sess.run(z1, feed_dict={x1: 1, y1: 2})
    print('Run one op: ', z1_value)

    # run multiple operation
    z1_value, z2_value = sess.run([z1, z2],
                                  feed_dict={x1: 1, y1: 2, x2: [[2], [1]], y2: [[1, 3]]})
    print('Run multi op: ', z1_value)
    print('Run multi op: ', z2_value)

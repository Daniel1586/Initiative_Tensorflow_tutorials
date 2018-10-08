#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

matrix1 = tf.constant([[3, 4]])
matrix2 = tf.constant([[2],
                       [3]])
dot_op = tf.matmul(matrix1, matrix2)

# method 1: 使用close方法显示关闭会话
sess = tf.Session()
result = sess.run(dot_op)
print('Method 1: ', result)
sess.close()

# method 2: 使用with语句隐式关闭会话
with tf.Session() as sess:
    result2 = sess.run(dot_op)
    print('Method 2: ', result2)

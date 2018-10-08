#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# 原始数据生成
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# 定义存储节点
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# 定义计算节点
y = w*x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化存储节点
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b))

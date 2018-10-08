#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

var = tf.Variable(0, name='counter')
add_op = tf.add(var, 1)
update = tf.assign(var, add_op)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        sess.run(update)
        print(sess.run(var))

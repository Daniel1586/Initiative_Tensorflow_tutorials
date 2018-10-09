#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

print('========== 1.Generating data...')
npx = np.random.uniform(-1, 1, (1000, 1))                           # x data
npy = np.power(npx, 2) + np.random.normal(0, 0.1, size=npx.shape)   # y data
npx_train, npx_test = np.split(npx, [800])                          # training and test data
npy_train, npy_test = np.split(npy, [800])

tfx = tf.placeholder(npx_train.dtype, npx_train.shape)
tfy = tf.placeholder(npy_train.dtype, npy_train.shape)

print('========== 2.Creating dataset...')
dataset = tf.data.Dataset.from_tensor_slices((tfx, tfy))
dataset = dataset.shuffle(buffer_size=1000)         # choose data randomly from this buffer
dataset = dataset.batch(32)                         # batch size you will use
dataset = dataset.repeat(3)                         # repeat for 3 epochs
iterator = dataset.make_initializable_iterator()    # later we have to initialize this one

print('========== 3.Building Network...')
bx, by = iterator.get_next()                        # use batch to update
l1 = tf.layers.dense(bx, 10, tf.nn.relu)
lo = tf.layers.dense(l1, npy.shape[1])
loss = tf.losses.mean_squared_error(by, lo)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run([iterator.initializer, tf.global_variables_initializer()],
         feed_dict={tfx: npx_train, tfy: npy_train})

for step in range(201):
    try:
        _, train_loss = sess.run([train, loss])
        if step % 10 == 0:
            test_loss = sess.run(loss, {bx: npx_test, by: npy_test})
            print('step: %i/200' % step, '|train loss:', train_loss, '|test loss:', test_loss)
    except tf.errors.OutOfRangeError:     # if training takes more than 3 epochs, training will be stopped
        print('Finish the last epoch.')
        break

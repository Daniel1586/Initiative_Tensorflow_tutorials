#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print('========== 1.Setting Hyper Parameters...')
input_size = 1          # rnn input size
times_step = 10         # rnn time step
cells_size = 32
learn_rate = 0.02

print('========== 2.Generating data...')
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
xdata = np.sin(steps)
ydata = np.cos(steps)
plt.plot(steps, ydata, 'r-', label='target (cos)')
plt.plot(steps, xdata, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()

tf_x = tf.placeholder(tf.float32, [None, times_step, input_size])           # shape(batch, 5, 1)
tf_y = tf.placeholder(tf.float32, [None, times_step, input_size])           # input y

print('========== 3.Building Network...')
rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=cells_size)
init_s = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
outputs, final_s = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    tf_x,                       # input
    initial_state=init_s,       # the initial hidden state
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
outs2D = tf.reshape(outputs, [-1, cells_size])                      # reshape 3D output to 2D for fully connected layer
net_outs2D = tf.layers.dense(outs2D, input_size)
outs = tf.reshape(net_outs2D, [-1, times_step, input_size])         # reshape back to 3D
loss = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)  # compute cost
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())         # initialize var in graph
plt.figure(1, figsize=(16, 5))
plt.ion()
for step in range(60):
    start, end = step * np.pi, (step+1)*np.pi       # time range
    # use sin predicts cos
    steps = np.linspace(start, end, times_step)
    x = np.sin(steps)[np.newaxis, :, np.newaxis]    # shape (batch, time_step, input_size)
    y = np.cos(steps)[np.newaxis, :, np.newaxis]
    if 'final_s_' not in globals():                 # first state, no any hidden state
        feed_dict = {tf_x: x, tf_y: y}
    else:                                           # has hidden state, so pass it to rnn
        feed_dict = {tf_x: x, tf_y: y, init_s: final_s_}
    _, pred_, final_s_ = sess.run([train_op, outs, final_s], feed_dict)     # train

    # plotting
    plt.plot(steps, y.flatten(), 'r-', label='target')
    plt.plot(steps, pred_.flatten(), 'b-', label='predict')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.2)
plt.ioff()
plt.show()

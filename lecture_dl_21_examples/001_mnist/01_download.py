#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 读取MNIST数据,数据不存在时,会自动执行下载
mnist = input_data.read_data_sets("mnist/", one_hot=True)

print('----- train x shape: ', mnist.train.images.shape)            # (55000, 784)
print('----- train y shape: ', mnist.train.labels.shape)            # (55000, 10)
print('----- validation x shape: ', mnist.validation.images.shape)  # (5000, 784)
print('----- validation y shape: ', mnist.validation.labels.shape)  # (5000, 10)
print('----- test x shape: ', mnist.test.images.shape)              # (10000, 784)
print('----- test y shape: ', mnist.test.labels.shape)              # (10000, 10)

# 打印出第1幅图片的向量表示和label
print(mnist.train.images[1, :])
print(mnist.train.labels[1, :])
plt.imshow(mnist.train.images[1].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[1]))
plt.show()

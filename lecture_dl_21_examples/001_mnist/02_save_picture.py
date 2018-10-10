#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import scipy.misc
from tensorflow.examples.tutorials.mnist import input_data

# 读取MNIST数据,数据不存在时,会自动执行下载
mnist = input_data.read_data_sets("mnist/", one_hot=True)

# 图片保存在mnist/pic/文件夹下,文件夹不存在时,则自动创建
save_dir = 'mnist/pic/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 保存前20张图片
for i in range(20):
    # mnist.train.images[i, :]就表示第i张图片(序号从0开始)
    image_array = mnist.train.images[i, :]  # (1,784)
    image_array = image_array.reshape(28, 28)
    # 保存文件的格式为 mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
    filename = save_dir + 'mnist_train_%d.jpg' % i
    # 将image_array保存为图片
    # 先用scipy.misc.toimage转换为图像,再调用save直接保存
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

print('Please check: %s ' % save_dir)

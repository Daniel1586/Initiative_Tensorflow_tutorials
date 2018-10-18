#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import scipy.misc
import cifar10_input
import tensorflow as tf


def inputs_origin(data_dir):
    filenames = [os.path.join(data_dir, 'data_batch_%d' % i1) for i1 in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    filename_queue = tf.train.string_input_producer(filenames)
    # cifar10_input.read_cifar10是事先写好的从queue中读取文件的函数
    # 返回的结果read_input的属性uint8image就是图像的Tensor
    read_input = cifar10_input.read_cifar10(filename_queue)
    read_image = tf.cast(read_input.uint8image, tf.float32)

    return read_image


if __name__ == '__main__':
    with tf.Session() as sess:
        reshaped_image = inputs_origin('cifar10_pic')
        # 使用start_queue_runners之后,才会真正把tensor推入到文件名队列
        threads = tf.train.start_queue_runners(sess=sess)
        sess.run(tf.global_variables_initializer())
        # 创建文件夹cifar10_data/raw/
        if not os.path.exists('cifar10_pic/raw/'):
            os.makedirs('cifar10_pic/raw/')
        # 保存30张图片
        for i in range(50):
            # 每次sess.run(reshaped_image)，都会取出一张图片
            image_array = sess.run(reshaped_image)
            # 将图片保存
            scipy.misc.toimage(image_array).save('cifar10_pic/raw/%d.jpg' % i)

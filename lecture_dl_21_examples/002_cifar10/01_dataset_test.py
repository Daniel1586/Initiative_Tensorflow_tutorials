#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

if not os.path.exists('example_pic'):
    os.makedirs('example_pic/')

with tf.Session() as sess:
    filename = ['A.jpg', 'B.jpg', 'C.jpg']
    # string_input_producer会产生一个文件名队列,shuffle=False不打乱数据顺序,num_epochs训练周期
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=5)
    # reader从文件名队列中读数据,对应的方法是reader.read--输出文件名(key)和该文件内容(value)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    # tf.train.string_input_producer定义了一个epoch变量(num_epochs不为None),要对它进行初始化
    tf.local_variables_initializer().run()
    # 使用start_queue_runners之后,才会真正把tensor推入到文件名队列
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        image_data = sess.run(value)
        with open('example_pic/test_%d.jpg' % i, 'wb') as f:
            f.write(image_data)
# 程序最后会抛出一个OutOfRangeError,这是epoch跑完,队列关闭的标志

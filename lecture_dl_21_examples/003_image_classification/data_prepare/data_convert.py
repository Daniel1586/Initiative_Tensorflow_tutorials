#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import os
import logging
import argparse
from tfrecord import main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tensorflow-data-dir', default='pic/')      # 待转换格式的图像存放在pic文件夹
    parser.add_argument('--train-shards', default=2, type=int)              # 训练图像生成的tfrecord文件分成2份
    parser.add_argument('--valid-shards', default=2, type=int)              # 验证图像生成的tfrecord文件分成2份
    parser.add_argument('--num-threads', default=2, type=int)               # 线程数,必须整除train/valid-shards
    parser.add_argument('--dataset-name', default='satellite', type=str)    # 数据集名(数据来源于卫星航拍图)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.tensorflow_dir = args.tensorflow_data_dir
    args.train_directory = os.path.join(args.tensorflow_dir, 'train')
    args.valid_directory = os.path.join(args.tensorflow_dir, 'valid')
    args.output_directory = args.tensorflow_dir
    args.labels_file = os.path.join(args.tensorflow_dir, 'label.txt')
    if os.path.exists(args.labels_file) is False:
        logging.warning('Can\'t find label.txt. Now create it.')
        all_entries = os.listdir(args.train_directory)
        dirnames = []
        for entry in all_entries:
            if os.path.isdir(os.path.join(args.train_directory, entry)):
                dirnames.append(entry)
        with open(args.labels_file, 'w') as f:
            for dirname in dirnames:
                f.write(dirname + '\n')
    main(args)

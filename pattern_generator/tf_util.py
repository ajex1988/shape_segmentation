####################################
# util function for tensorflow
# Zhe Zhu created, 2020/03/23
####################################

import PatternGenerator as pg
import tensorflow as tf
import numpy as np
import os
import glob

import cv2

def imgs2tfrecord_comp(img_file_list, tfrecord_file):
    '''
    Convert the images to a tfrecord with compression
    img_file_list: image(not including mask) file list
    tfrecord_file: target tfrecord file to write
    Zhe Zhu, 2020/03/23
    '''
    suffix = '_mask.png' # maybe we change the mind in the future
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for img_path in img_file_list:
            img_file_name = os.path.basename(img_path)
            img_dir_name = os.path.dirname(img_path)
            mask_file_name = img_file_name[:-4] + '_mask.png'
            mask_path = os.path.join(img_dir_name,mask_file_name)
            with tf.gfile.FastGFile(img_path,'rb') as fid:
                img_data_raw = fid.read()
            with tf.gfile.FastGFile(mask_path,'rb') as fid:
                mask_data_raw = fid.read()
            mask = cv2.imread(mask_path)
            img_height = mask.shape[0]
            img_width = mask.shape[1]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'file_name':pg._bytes_feature(img_path),
                        'height': pg._int64_feature(img_height),
                        'width': pg._int64_feature(img_width),
                        'image': pg._bytes_feature(img_data_raw),
                        'mask': pg._bytes_feature(mask_data_raw)
                    }
                )
            )
            writer.write(example.SerializeToString())

def cvt_simulated_data(input_folder,img_name_fmt,img_num,tfrecord_file):
    '''
    Convert the images following the img_name_fmt to tfrecord_file
    Zhe Zhu, 2020/03/24
    '''
    img_file_list = []
    for i in range(img_num):
        img_file_name = img_name_fmt.format(i)
        img_path = os.path.join(input_folder,img_file_name)
        img_file_list.append(img_path)
    imgs2tfrecord_comp(img_file_list,tfrecord_file)

def task_1():
    '''
    Convert 0219_contour, 0224_contour to tfrecord
    Zhe Zhu,2020/03/23
    '''
    img_folder_list = ['/mnt/sdc/ShapeTexture/simulation_data/0219_contour/train',
                       '/mnt/sdc/ShapeTexture/simulation_data/0219_contour/val',
                       '/mnt/sdc/ShapeTexture/simulation_data/0224_contour/train',
                       '/mnt/sdc/ShapeTexture/simulation_data/0224_contour/val']
    img_name_fmt_list = ['train_{:05d}.png',
                         'val_{:05d}.png',
                         'train_{:05d}.png',
                         'val_{:05d}.png']
    img_num_list = [20000,500,20000,500]
    tfrecord_file_list = ['/mnt/sdc/ShapeTexture/simulation_data/0219_contour/train.tfrecord',
                          '/mnt/sdc/ShapeTexture/simulation_data/0219_contour/val.tfrecord',
                          '/mnt/sdc/ShapeTexture/simulation_data/0224_contour/train.tfrecord',
                          '/mnt/sdc/ShapeTexture/simulation_data/0224_contour/val.tfrecord']
    folder_num = len(img_folder_list)
    for i in range(folder_num):
        img_folder = img_folder_list[i]
        img_name_fmt = img_name_fmt_list[i]
        img_num = img_num_list[i]
        tfrecord_file = tfrecord_file_list[i]
        cvt_simulated_data(img_folder,img_name_fmt,img_num,tfrecord_file)

if __name__ == "__main__":
    task_1()


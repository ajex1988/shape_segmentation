import tensorflow as tf
import os
import glob
import numpy as np
from sets import Set
import cv2
import time
import re


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def dataset_decode(tfrecord):
    features = {
        'file_name': tf.FixedLenFeature([],tf.string,''),
        'height': tf.FixedLenFeature([],tf.int64,0),
        'width': tf.FixedLenFeature([],tf.int64,0),
        'depth': tf.FixedLenFeature([],tf.int64,0),
        'volume': tf.FixedLenFeature([],tf.string,''),
        'mask': tf.FixedLenFeature([],tf.string,'')
    }
    sample = tf.parse_single_example(tfrecord,features)
    return sample


def dataset_decode_(tfrecord):
    features = {
        'file_name': tf.FixedLenFeature([],tf.string,''),
        'height': tf.FixedLenFeature([],tf.int64,0),
        'width': tf.FixedLenFeature([],tf.int64,0),
        'depth': tf.FixedLenFeature([],tf.int64,0),
        'volume': tf.FixedLenFeature([],tf.string,''),
        'mask': tf.FixedLenFeature([],tf.string,'')
    }
    sample = tf.parse_single_example(tfrecord,features)
    sample['volume'] = tf.decode_raw(sample['volume'],tf.float64)
    sample['mask'] = tf.decode_raw(sample['mask'],tf.float64)
    return sample

def dataset_decode__(tfrecord):
    features = {
        'file_name': tf.FixedLenFeature([],tf.string,''),
        'height': tf.FixedLenFeature([],tf.int64,0),
        'width': tf.FixedLenFeature([],tf.int64,0),
        'image': tf.FixedLenFeature([],tf.string,''),
        'mask': tf.FixedLenFeature([],tf.string,'')
    }
    sample = tf.parse_single_example(tfrecord,features)
    return sample

def dataset_decode___(tfrecord):
    features = {
        'file_name': tf.FixedLenFeature([],tf.string,''),
        'height': tf.FixedLenFeature([],tf.int64,0),
        'width': tf.FixedLenFeature([],tf.int64,0),
        'image': tf.FixedLenFeature([],tf.string,''),
        'mask': tf.FixedLenFeature([],tf.string,'')
    }
    sample = tf.parse_single_example(tfrecord,features)
    sample['image'] = tf.decode_raw(sample['image'], tf.float64)
    sample['mask'] = tf.decode_raw(sample['mask'], tf.float64)
    return sample

def numpy2tfrecord(numpy_folder, tfrecord_file):

    nparray_list = glob.glob(numpy_folder+'/input*')
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for npy_file in nparray_list:

            file_name = os.path.basename(npy_file)
            print(file_name)
            volume = np.load(npy_file)

            mask_file = 'target'+file_name[5:]
            mask = np.load(os.path.join(numpy_folder,mask_file))

            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'file_name':_bytes_feature(file_name),
                        'height':_int64_feature(volume.shape[1]),
                        'width':_int64_feature(volume.shape[2]),
                        'depth':_int64_feature(volume.shape[0]),
                        'volume': _bytes_feature(volume.tostring()),
                        'mask':_bytes_feature(mask.tostring())
                    }
                )
            )
            writer.write(example.SerializeToString())

def process_ct_data(numpy_folder, tfrecord_file):
    nparray_list = glob.glob(numpy_folder + '/input*')
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for npy_file in nparray_list:
            file_name = os.path.basename(npy_file)
            print(file_name)
            volume = np.load(npy_file)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'file_name': _bytes_feature(file_name),
                        'height': _int64_feature(volume.shape[1]),
                        'width': _int64_feature(volume.shape[2]),
                        'depth': _int64_feature(volume.shape[0]),
                        'volume': _bytes_feature(volume.tostring()),
                    }
                )
            )
            writer.write(example.SerializeToString())

def separate_train_val(tfrecord_whole_file,train_tfrecord_file,val_tfrecord_file,val_file_list):
    dataset = tf.data.TFRecordDataset(tfrecord_whole_file)
    dataset = dataset.map(dataset_decode)
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()

    with tf.Session() as sess:
        with tf.python_io.TFRecordWriter(train_tfrecord_file) as train_writer:
            with tf.python_io.TFRecordWriter(val_tfrecord_file) as val_writer:
                sess.run(tf.global_variables_initializer())
                try:
                    while(True):
                        sample = sess.run(next_elem)
                        file_name = sample['file_name']
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    'file_name': _bytes_feature(sample['file_name']),
                                    'height': _int64_feature(sample['height']),
                                    'width': _int64_feature(sample['width']),
                                    'depth': _int64_feature(sample['depth']),
                                    'volume': _bytes_feature(sample['volume']),
                                    'mask': _bytes_feature(sample['mask'])
                                }
                            )
                        )
                        if file_name in val_file_list:
                            val_writer.write(example.SerializeToString())
                        else:
                            train_writer.write(example.SerializeToString())
                except Exception as e:
                    print(e)
def separate_train_val_by_rate(rate,dataset_folder,ridx,tfrecord_src,train_tfrecord,val_tfrecord):
    volume_file_list = glob.glob(dataset_folder+'/input*')
    patient_file_list = [file_name[:-ridx] for file_name in volume_file_list]
    unique_list = np.unique(patient_file_list)
    val_num = int(rate*len(unique_list))
    val_list = unique_list[-val_num:]

    val_sets = Set()
    val_file_list = [(glob.glob(val_file+'*')) for val_file in val_list]
    for files in val_file_list:
        for file in files:
            val_sets.add(os.path.basename(file))
    separate_train_val(tfrecord_src,train_tfrecord,val_tfrecord,val_sets)

def seperate_train_val_by_num(train_num,dataset_folder,pattern,tfrecord_src,train_tfrecord,val_tfrecord):
    volume_file_list = glob.glob(dataset_folder+'/input*')
    patient_file_list = [re.match(pattern,os.path.basename(file_name)).group(0) for file_name in volume_file_list]
    unique_list = np.unique(patient_file_list)
    val_num = len(unique_list) - train_num
    val_list = unique_list[-val_num:]
    val_sets = Set()
    val_file_list = [(glob.glob(val_file+'*')) for val_file in val_list]
    for files in val_file_list:
        for file in files:
            val_sets.add(os.path.basename(file))
    separate_train_val(tfrecord_src,train_tfrecord,val_tfrecord,val_sets)

def seperate_train_val_by_series_num(train_num,dataset_folder,tfrecord_src,train_tfrecord,val_tfrecord):
    volume_file_list = glob.glob(dataset_folder+'/input*')
    patient_file_list = [os.path.basename(file_name) for file_name in volume_file_list]
    val_num = len(patient_file_list) - train_num
    val_file_list = patient_file_list[-val_num:]
    val_sets = Set()
    for file in val_file_list:
        val_sets.add(os.path.basename(file))
    separate_train_val(tfrecord_src,train_tfrecord,val_tfrecord,val_sets)
# def separate_train_val_by_num_r(train_num,dataset_folder,ridx,tfrecord_src,train_tfrecord,val_tfrecord):
#     volume_file_list = glob.glob(dataset_folder+'/input*')
#     patient_file_list = [file_name[:-ridx] for file_name in volume_file_list]
#     unique_list = np.unique(patient_file_list)
#     val_num = len(unique_list) - train_num
#     val_list = unique_list[-val_num:]
#
#     val_sets = Set()
#     val_file_list = [(glob.glob(val_file+'*')) for val_file in val_list]
#     for files in val_file_list:
#         for file in files:
#             val_sets.add(os.path.basename(file))
#     separate_train_val(tfrecord_src,train_tfrecord,val_tfrecord,val_sets)

# def separate_train_val_by_num_l(train_num,dataset_folder,lidx,tfrecord_src,train_tfrecord,val_tfrecord):
#     volume_file_list = glob.glob(dataset_folder+'/input*')
#     patient_file_list = [file_name[:lidx] for file_name in volume_file_list]
#     unique_list = np.unique(patient_file_list)
#     val_num = len(unique_list) - train_num
#     val_list = unique_list[-val_num:]
#
#     val_sets = Set()
#     val_file_list = [(glob.glob(val_file+'*')) for val_file in val_list]
#     for files in val_file_list:
#         for file in files:
#             val_sets.add(os.path.basename(file))
#     separate_train_val(tfrecord_src,train_tfrecord,val_tfrecord,val_sets)

def print_tfrecord(tfrecord_file,output_folder):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(dataset_decode_)
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while(True):
                sample = sess.run(next_elem)
                file_name = sample['file_name']
                height = sample['height']
                width = sample['width']
                depth = sample['depth']
                volume = sample['volume']
                masks = sample['mask']
                volume = np.reshape(volume,(depth,height,width))
                masks = np.reshape(masks,(depth,height,width))
                seq_folder = file_name[:-4]
                if not os.path.exists(os.path.join(output_folder,seq_folder)):
                    os.mkdir(os.path.join(output_folder,seq_folder))
                for i in range(depth):
                    img_file_name = '{0}.png'.format(i)
                    mask_file_name = '{0}_mask.png'.format(i)
                    img = volume[i,:,:]
                    img = (img-np.amin(img)+1e-8)/(np.amax(img)-np.amin(img)+1e-8)*255.0
                    mask = masks[i,:,:]*255.0
                    cv2.imwrite(os.path.join(output_folder,seq_folder,img_file_name),img)
                    cv2.imwrite(os.path.join(output_folder,seq_folder,mask_file_name),mask)


        except Exception as e:
            print(e)

def convert2slice(tfrecord_3d,tfrecord_2d):
    dataset = tf.data.TFRecordDataset(tfrecord_3d)
    dataset = dataset.map(dataset_decode_)
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()

    with tf.Session() as sess:
        with tf.python_io.TFRecordWriter(tfrecord_2d) as writer:
            sess.run(tf.global_variables_initializer())
            try:
                while(True):
                    sample = sess.run(next_elem)
                    volume = sample['volume']
                    masks = sample['mask']
                    height = sample['height']
                    width = sample['width']
                    depth = sample['depth']
                    volume = np.reshape(volume, (depth, height, width))
                    masks = np.reshape(masks, (depth, height, width))
                    for i in range(depth):
                        image = volume[i,:,:]
                        mask = masks[i,:,:]
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    'file_name': _bytes_feature(sample['file_name']),
                                    'height': _int64_feature(sample['height']),
                                    'width': _int64_feature(sample['width']),
                                    'image': _bytes_feature(image.tostring()),
                                    'mask': _bytes_feature(mask.tostring())
                                }
                            )
                        )
                        writer.write(example.SerializeToString())


            except Exception as e:
                print(e)

def merge_dataset(input_tfrecord_list,output_tfrecord_file):
    with tf.Session() as sess:
        count = 0
        with tf.python_io.TFRecordWriter(output_tfrecord_file) as writer:
            sess.run(tf.global_variables_initializer())
            for input_tfrecord_file in input_tfrecord_list:
                dataset = tf.data.TFRecordDataset(input_tfrecord_file)
                dataset = dataset.map(dataset_decode__)
                iterator = dataset.make_one_shot_iterator()
                next_elem = iterator.get_next()
                try:
                    while(True):
                        sample = sess.run(next_elem)
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    'file_name': _bytes_feature(sample['file_name']),
                                    'height': _int64_feature(sample['height']),
                                    'width': _int64_feature(sample['width']),
                                    'image': _bytes_feature(sample['image']),
                                    'mask': _bytes_feature(sample['mask'])
                                }
                            )
                        )
                        count += 1
                        if count % 1000 == 0:
                            print(count)
                        writer.write(example.SerializeToString())
                except Exception as e:
                    print(e)

def write_gt(tfrecord_file,output_folder):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(dataset_decode___)
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()
    count = 0
    weight = 0.5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while(True):
                sample = sess.run(next_elem)
                image = sample['image']*255
                image = np.reshape(image,(sample['height'],sample['width'],1))
                mask = sample['mask']
                mask = np.reshape(mask,(sample['height'],sample['width']))
                color_gt = np.zeros((image.shape[0], image.shape[1], 3))
                ind = np.squeeze(mask > 0)
                color_gt[ind] = (0, 0, 255)

                masked_gt = weight * image + (1 - weight) * color_gt

                gt_file_path = os.path.join(output_folder, '{:04d}_gt.png'.format(count))
                count += 1

                cv2.imwrite(gt_file_path, masked_gt)


        except:
            pass

def re_check(folder,pattern):
    # check all the file names by regular expression
    passed = True
    file_name_list = [os.path.basename(name) for name in glob.glob(folder+'/*')]
    for file_name in file_name_list:
        matched = re.match(pattern,file_name)
        if matched == None:
            passed = False
            print('Invalid File Name: {0}'.format(file_name))
    return passed

def invert_dataset(src_tfrecord_file,tgt_tfrecord_file):
    with tf.Session() as sess:
        count = 0
        with tf.python_io.TFRecordWriter(tgt_tfrecord_file) as writer:
            sess.run(tf.global_variables_initializer())

            dataset = tf.data.TFRecordDataset(src_tfrecord_file)
            dataset = dataset.map(dataset_decode___)
            iterator = dataset.make_one_shot_iterator()
            next_elem = iterator.get_next()
            try:
                while(True):
                    sample = sess.run(next_elem)
                    img = sample['image']
                    img = 1.0-img
                    sample['image'] = img.tostring()
                    sample['mask'] = sample['mask'].tostring()
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'file_name': _bytes_feature(sample['file_name']),
                                'height': _int64_feature(sample['height']),
                                'width': _int64_feature(sample['width']),
                                'image': _bytes_feature(sample['image']),
                                'mask': _bytes_feature(sample['mask'])
                            }
                        )
                    )
                    count += 1
                    if count % 1000 == 0:
                        print(count)
                    writer.write(example.SerializeToString())
            except Exception as e:
                print(e)
def test_0():
    numpy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dynpre'
    nparray_list = glob.glob(numpy_folder + '/input*')
    for npy_file in nparray_list:
        file_name = os.path.basename(npy_file)
        print(file_name)
        volume = np.load(npy_file)

        mask_file = 'target' + file_name[5:]
        mask = np.load(os.path.join(numpy_folder, mask_file))
def test_1():
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dynpre'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_whole.tfrecord'
    numpy2tfrecord(npy_folder,tfrecord_file)
def test_2():
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/opposed'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_whole.tfrecord'
    numpy2tfrecord(npy_folder, tfrecord_file)

def test_3():
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/SSFSE'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_whole.tfrecord'
    numpy2tfrecord(npy_folder, tfrecord_file)

def test_4():
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/t1nfs'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_whole.tfrecord'
    numpy2tfrecord(npy_folder, tfrecord_file)

def test_5():
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/CT'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ct_whole.tfrecord'
    process_ct_data(npy_folder, tfrecord_file)

def test_6():
    rate = 0.2
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dynpre'
    ridx = 12
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_val.tfrecord'

    separate_train_val_by_rate(rate,dataset_folder,ridx,tfrecord_src,tfrecord_train,tfrecord_val)


def test_7():
    rate = 0.2
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/opposed'
    ridx = 12
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_val.tfrecord'


    separate_train_val_by_rate(rate,dataset_folder,ridx,tfrecord_src,tfrecord_train,tfrecord_val)

def test_8():
    rate = 0.2
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/SSFSE'
    ridx = 10
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_val.tfrecord'


    separate_train_val_by_rate(rate,dataset_folder,ridx,tfrecord_src,tfrecord_train,tfrecord_val)

def test_9():
    rate = 0.2
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/t1nfs'
    ridx = 10
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_val.tfrecord'


    separate_train_val_by_rate(rate,dataset_folder,ridx,tfrecord_src,tfrecord_train,tfrecord_val)

def test_10():
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_whole.tfrecord'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/img_mask/dynpre'
    print_tfrecord(tfrecord_file,output_folder)

def test_11():
    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_train_2d.tfrecord'
    convert2slice(tfrecord_3d,tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_val.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_val_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_val.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_val_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_val.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_val_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_val.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_val_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

def test_12():
    tfrecord_list = ['/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_train_2d.tfrecord',
                     '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_val_2d.tfrecord',
                     '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_train_2d.tfrecord',
                     '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_val_2d.tfrecord',
                     '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_train_2d.tfrecord',
                     '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_val_2d.tfrecord']
    output_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dataset_a_2d.tfrecord'
    merge_dataset(tfrecord_list,output_tfrecord_file)

    tfrecord_list = ['/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_train_2d.tfrecord',
                     '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_train_2d.tfrecord',
                     '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_train_2d.tfrecord',
                     '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_train_2d.tfrecord']
    output_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dataset_b_2d.tfrecord'
    merge_dataset(tfrecord_list, output_tfrecord_file)

def test_13():
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/CT'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ct_whole.tfrecord'
    numpy2tfrecord(npy_folder, tfrecord_file)

def test_14():
    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ct_whole.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ct_whole_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)


def test_15():
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_train_2d.tfrecord'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/output/model_b_dynpre'
    write_gt(tfrecord_file,output_folder)

def test_16():
    # rebuild the dataset
    test_1()
    print('dynpre finished')
    test_2()
    print('opposed finished')
    test_3()
    print('ssfse finished')
    test_4()
    print('t1nfs finished')

    # CT
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/CT'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ct_whole.tfrecord'
    numpy2tfrecord(npy_folder,tfrecord_file)
    print('ct finished')
    # Portal Venous
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/fifty_portal_venous_phases'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynportal_whole.tfrecord'
    numpy2tfrecord(npy_folder, tfrecord_file)
    print('dynportal finished')

def test_16_1():
    # re-do the ct due to the naming mistake: LRML0094 -> LRML_0094
    # CT
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/CT'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ct_whole.tfrecord'
    start = time.time()
    numpy2tfrecord(npy_folder, tfrecord_file)
    end = time.time()
    print('ct finished, cost {0}'.format(end-start))

def test_16_2():
    # check all the names to see if they have the same naming convention
    ct_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/CT'
    ct_re = '(input|target)_LRML_\d{4}_(ctpre|ctdelay|ctportal|ctarterial).npy'
    print('Check CT')
    result = re_check(folder=ct_folder,
                      pattern=ct_re)
    print(result)

    print('check dynportal')
    dynportal_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dynportal'
    dynportal_re = '(input|target)_LRML_\d{4}_dynportal\d{0,1}.npy'
    result = re_check(folder=dynportal_folder,
                      pattern=dynportal_re)
    print(result)

    print('check dynpre')
    dynpre_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dynpre'
    dynpre_re = '(input|target)_LRML_\d{4}_dynpre\d{0,1}.npy'
    result = re_check(folder=dynpre_folder,
                      pattern=dynpre_re)
    print(result)

    print('check opposed')
    opposed_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/opposed'
    opposed_re = '(input|target)_(NGM02|NSTx1)_\d{2,3}_\d{3}_(A|B|C|D)\d*_opposed.npy'
    result = re_check(folder=opposed_folder,
                      pattern=opposed_re)
    print(result)

    print('check ssfse')
    ssfse_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/SSFSE'
    ssfse_re = '(input|target)_(JKB01)_\d{3}_\d{2}_.*_SSFSE.npy'
    result = re_check(folder=ssfse_folder,
                      pattern=ssfse_re)
    print(result)

    print('check t1nfs')
    t1nfs_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/t1nfs'
    t1nfs_re = '(input|target)_(NGM01_\d{3}_\d{2}_(A|B)|NGM03_\d{5})_t1nfs.npy'
    result = re_check(folder=t1nfs_folder,
                      pattern=t1nfs_re)
    print(result)
def test_17():
    # construct dataset for network A, B, C, D
    ct_re = '(input|target)_LRML_\d{4}_(ctpre|ctdelay|ctportal|ctarterial).npy'
    dynportal_re = '(input|target)_LRML_\d{4}_dynportal\d{0,1}.npy'
    dynpre_re = '(input|target)_LRML_\d{4}_dynpre\d{0,1}.npy'
    opposed_re = '(input|target)_(NGM02|NSTx1)_\d{2,3}_\d{3}_(A|B|C|D)\d*_opposed.npy'
    ssfse_re = '(input|target)_(JKB01)_\d{3}_\d{2}_.*_SSFSE.npy'
    t1nfs_re = '(input|target)_(NGM01_\d{3}_\d{2}|NGM03_\d{5})'

    print('Constructing Dataset for Network A')
    print('Pick 99 t1nfs')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/t1nfs'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/t1nfs_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/t1nfs_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=99,
                              dataset_folder=dataset_folder,
                              tfrecord_src=tfrecord_src,
                              train_tfrecord=tfrecord_train,
                              val_tfrecord=tfrecord_val)
    end = time.time()
    print('t1nfs sperated into train and val, cost: {0}'.format(end-start))

    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/opposed'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/opposed_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/opposed_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=114,
                              dataset_folder=dataset_folder,
                              tfrecord_src=tfrecord_src,
                              train_tfrecord=tfrecord_train,
                              val_tfrecord=tfrecord_val)
    end = time.time()
    print('opposed sperated into train and val, cost: {0}'.format(end - start))

def test_18():
    print('Constructing Dataset for Network B')
    print('Pick 99 SSFSE')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/SSFSE'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/ssfse_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/ssfse_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=99,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('SSFSE sperated into train and val, cost: {0}'.format(end - start))

    print('Pick 75 t1nfs')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/t1nfs'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/t1nfs_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/t1nfs_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=75,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('t1nfs sperated into train and val, cost: {0}'.format(end - start))

    print('Pick 85 opposed')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/opposed'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/opposed_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/opposed_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=85,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('opposed sperated into train and val, cost: {0}'.format(end - start))

    print('Pick 86 dynpre')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dynpre'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/dynpre_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/dynpre_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=86,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('dynpre sperated into train and val, cost: {0}'.format(end - start))
def test_19():
    print('Constructing Dataset for Network C')
    print('Pick 113 SSFSE')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/SSFSE'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/ssfse_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/ssfse_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=113,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('SSFSE sperated into train and val, cost: {0}'.format(end - start))

    print('Pick 85 t1nfs')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/t1nfs'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/t1nfs_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/t1nfs_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=85,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('t1nfs sperated into train and val, cost: {0}'.format(end - start))

    print('Pick 97 opposed')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/opposed'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/opposed_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/opposed_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=97,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('opposed sperated into train and val, cost: {0}'.format(end - start))

    print('Pick 50 dynportal')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dynportal'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynportal_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/dynportal_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/dynportal_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=50,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('dynportal sperated into train and val, cost: {0}'.format(end - start))
def test_20():
    print('Constructing Dataset for Network D')
    print('Pick 80 SSFSE')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/SSFSE'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/ssfse_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/ssfse_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=80,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('SSFSE sperated into train and val, cost: {0}'.format(end - start))

    print('Pick 60 t1nfs')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/t1nfs'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t1nfs_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/t1nfs_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/t1nfs_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=60,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('t1nfs sperated into train and val, cost: {0}'.format(end - start))

    print('Pick 69 opposed')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/opposed'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/opposed_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/opposed_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/opposed_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=69,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('opposed sperated into train and val, cost: {0}'.format(end - start))

    print('Pick 86 dynpre')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dynpre'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/dynpre_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/dynpre_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=86,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('dynpre sperated into train and val, cost: {0}'.format(end - start))

    print('Pick 50 dynportal')
    dataset_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dynportal'
    tfrecord_src = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynportal_whole.tfrecord'
    tfrecord_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/dynportal_train.tfrecord'
    tfrecord_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/dynportal_val.tfrecord'
    start = time.time()
    seperate_train_val_by_series_num(train_num=50,
                                     dataset_folder=dataset_folder,
                                     tfrecord_src=tfrecord_src,
                                     train_tfrecord=tfrecord_train,
                                     val_tfrecord=tfrecord_val)
    end = time.time()
    print('dynportal sperated into train and val, cost: {0}'.format(end - start))

def test_21():
    # construct network a, b, c, d
    print('network a, 3d to 2d')
    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/ssfse_whole.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/ssfse_whole_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/t1nfs_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/t1nfs_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/opposed_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/opposed_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    print('network a, merging')
    tfrecord_list = ['/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/ssfse_whole_2d.tfrecord',
                     '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/t1nfs_train_2d.tfrecord',
                     '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_a/opposed_train_2d.tfrecord']
    output_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dataset_a.tfrecord'
    merge_dataset(tfrecord_list, output_tfrecord_file)

    print('network b, 3d to 2d')
    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/ssfse_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/ssfse_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/t1nfs_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/t1nfs_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/opposed_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/opposed_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/dynpre_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/dynpre_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    print('network b, merging')
    tfrecord_list = [
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/ssfse_train_2d.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/t1nfs_train_2d.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/opposed_train_2d.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_b/dynpre_train_2d.tfrecord']
    output_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dataset_b.tfrecord'
    merge_dataset(tfrecord_list, output_tfrecord_file)

    print('network c, 3d to 2d')
    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/ssfse_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/ssfse_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/t1nfs_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/t1nfs_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/opposed_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/opposed_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/dynportal_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/dynportal_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    print('network c, merging')
    tfrecord_list = [
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/ssfse_train_2d.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/t1nfs_train_2d.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/opposed_train_2d.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_c/dynportal_train_2d.tfrecord']
    output_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dataset_c.tfrecord'
    merge_dataset(tfrecord_list, output_tfrecord_file)

    print('network d, 3d to 2d')
    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/ssfse_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/ssfse_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/t1nfs_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/t1nfs_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/opposed_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/opposed_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/dynportal_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/dynportal_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    tfrecord_3d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/dynpre_train.tfrecord'
    tfrecord_2d = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/dynpre_train_2d.tfrecord'
    convert2slice(tfrecord_3d, tfrecord_2d)

    print('network d, merging')
    tfrecord_list = [
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/ssfse_train_2d.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/t1nfs_train_2d.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/opposed_train_2d.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/dynportal_train_2d.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/intermediate/network_d/dynpre_train_2d.tfrecord']
    output_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dataset_d.tfrecord'
    merge_dataset(tfrecord_list, output_tfrecord_file)

def test_22():
    # constructing maciej's fucking idea dataset step 1 separate train 1,2 test
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/ssfse_whole.tfrecord'
    # train_1 train_2 test
    train_1_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_1.tfrecord'
    train_2_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_2.tfrecord'
    test_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/test.tfrecord'

    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(dataset_decode_)
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()

    count = 0
    with tf.Session() as sess:
        with tf.python_io.TFRecordWriter(train_1_file) as writer_1:
            with tf.python_io.TFRecordWriter(train_2_file) as writer_2:
                with tf.python_io.TFRecordWriter(test_file) as writer_3:

                    sess.run(tf.global_variables_initializer())
                    try:
                        while (True):

                            sample = sess.run(next_elem)
                            volume = sample['volume']
                            masks = sample['mask']
                            height = sample['height']
                            width = sample['width']
                            depth = sample['depth']
                            volume = np.reshape(volume, (depth, height, width))
                            masks = np.reshape(masks, (depth, height, width))
                            for i in range(depth):
                                image = volume[i, :, :]
                                mask = masks[i, :, :]
                                example = tf.train.Example(
                                    features=tf.train.Features(
                                        feature={
                                            'file_name': _bytes_feature(sample['file_name']),
                                            'height': _int64_feature(sample['height']),
                                            'width': _int64_feature(sample['width']),
                                            'image': _bytes_feature(image.tostring()),
                                            'mask': _bytes_feature(mask.tostring())
                                        }
                                    )
                                )
                                if count < 60:
                                    writer_1.write(example.SerializeToString())
                                elif count>=60 and count < 120:
                                    writer_2.write(example.SerializeToString())
                                else:
                                    writer_3.write(example.SerializeToString())
                            count += 1
                    except Exception as e:
                        print(e)
def test_23():
    # follow up invert train1 train2
    src_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_1.tfrecord'
    tgt_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_1_inv.tfrecord'
    invert_dataset(src_file,tgt_file)

    src_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_2.tfrecord'
    tgt_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_2_inv.tfrecord'
    invert_dataset(src_file, tgt_file)

def test_24():
    tfrecord_list = ['/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_1.tfrecord',
                     '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_1_inv.tfrecord']
    output_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/exp3/train.tfrecord'
    merge_dataset(tfrecord_list, output_tfrecord_file)

    tfrecord_list = [
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_1.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_2_inv.tfrecord']
    output_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/exp4/train.tfrecord'
    merge_dataset(tfrecord_list, output_tfrecord_file)
def test_25():
    val_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/exp1/train.tfrecord'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/output/exp2_train'
    write_gt(val_tfrecord_file,output_folder)

def test_26():
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/exp2/train.tfrecord'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/output/exp2_train'
    write_gt(tfrecord_file,output_folder)

def test_27():
    """
    convert t2fse npy to tfrecord
    written by zhe zhu 02/Oct/2019
    """
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/t2fse_51/t2fse_51/output'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/t2nfs_whole_3d.tfrecord'
    numpy2tfrecord(npy_folder, tfrecord_file)

def test_28():
    """
    convert dynhbp npy to tfrecord
    written by zhe zhu 02/Oct/2019
    """
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dynhbp_33/dynhbp_33/output'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynhbp_whole_3d.tfrecord'
    numpy2tfrecord(npy_folder, tfrecord_file)

def test_29():
    """
    convert dyntransitional npy to tfrecord
    written by zhe zhu 02/Oct/2019
    :return:
    """
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dyntransitional_10/output'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dyntransitional_whole_3d.tfrecord'
    numpy2tfrecord(npy_folder, tfrecord_file)

def test_30():
    """
    convert portal_venous npy to tfrecord
    written by zhe zhu 02/Oct/2019
    :return:
    """
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/fifty_portal_venous_phases'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/portalvenous_whole_3d.tfrecord'
    numpy2tfrecord(npy_folder, tfrecord_file)

def test_31():
    """
    convert ssfsefs npy to tfrecord
    written by zhe zhu 02/Oct/2019
    :return:
    """
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/SSFSEfs_41/SSFSEfs_41/output'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/SSFSEfs_whole_3d.tfrecord'
    numpy2tfrecord(npy_folder, tfrecord_file)

def test_32():
    """
    convert dynarterial npy to tfrecord
    written by zhe zhu 03/Oct/2019
    :return:
    """
    npy_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/preprocessed_data/dynarterial_52/output'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynarterial_whole_3d.tfrecord'
    numpy2tfrecord(npy_folder, tfrecord_file)

def test_33():
    """
    invert the test set
    written by zhe zhu 07/Oct/2019
    :return:
    """
    src_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/test.tfrecord'
    tgt_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/test_inv.tfrecord'
    invert_dataset(src_file, tgt_file)
def test_34():
    """
    construct datasets for exp5 and exp6
    written by zhe zhu 09/Oct/2019
    :return:
    """
    tfrecord_list = [
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_1.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_2.tfrecord']
    output_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/exp5/train.tfrecord'
    merge_dataset(tfrecord_list, output_tfrecord_file)

    tfrecord_list = [
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_1_inv.tfrecord',
        '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/train_2_inv.tfrecord']
    output_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/exp6/train.tfrecord'
    merge_dataset(tfrecord_list, output_tfrecord_file)
if __name__ == '__main__':
    #test_0()
    #test_1()
    #test_2()
    #test_3()
    #test_4()
    #test_5()
    #test_7()
    #test_8()
    #test_9()
    #test_10()
    #test_11()
    #test_12()
    #test_13()
    #test_14()
    #test_15()
    #test_16()
    #test_16_1()
    #test_16_2()
    #test_17()
    #test_18()
    #test_19()
    #test_20()
    #test_21()
    #test_22()
    #test_23()
    #test_24()
    #test_25()
    #test_26()
    #test_27()
    #test_28()
    #test_29()
    #test_30()
    #test_31()
    #test_32()
    #test_33()
    test_34()


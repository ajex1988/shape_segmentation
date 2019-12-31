import tensorflow as tf
from unet_utils import Recorder
import config
import os
import cv2
import numpy as np
import sys

tf.logging.set_verbosity(tf.logging.INFO)

def conv_conv_pool(input_,
                   n_filters,
                   training,
                   flags,
                   name,
                   pool=True,
                   activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(flags),
                name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upconv_concat(inputA, input_B, n_filter, flags, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D(inputA, n_filter, flags, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))


def upconv_2D(tensor, n_filter, flags, name):
    """Up Convolution `tensor` by 2 times

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations

    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(flags),
        name="upsample_{}".format(name))


def make_unet(X, training, flags=None):
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    net = X / 127.5 - 1
    conv1, pool1 = conv_conv_pool(net, [8, 8], training, flags, name=1)
    #tf.summary.histograms('conv1',conv1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, flags, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, flags, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, flags, name=4)
    conv5 = conv_conv_pool(
        pool4, [128, 128], training, flags, name=5, pool=False)

    up6 = upconv_concat(conv5, conv4, 64, flags, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, flags, name=6, pool=False)

    up7 = upconv_concat(conv6, conv3, 32, flags, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, flags, name=7, pool=False)

    up8 = upconv_concat(conv7, conv2, 16, flags, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, flags, name=8, pool=False)

    up9 = upconv_concat(conv8, conv1, 8, flags, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, flags, name=9, pool=False)

    return tf.layers.conv2d(
        conv9,
        1, (1, 1),
        name='final',
        activation=tf.nn.sigmoid,
        padding='same')


def Dice(logit,y):
    y_pred = tf.math.greater(logit,0.5)
    y_pred = tf.to_float(y_pred)
    y_pred_flat = tf.layers.flatten(y_pred)
    y_true_flat = tf.layers.flatten(y)

    intersection = 2*tf.reduce_sum(y_pred_flat * y_true_flat, axis=1) + 1e-7

    denominator = tf.reduce_sum(y_pred_flat, axis=1)+tf.reduce_sum(y_true_flat, axis=1) + 1e-7

    dice = tf.reduce_mean(intersection/denominator)
       
    return dice

def IoU(logit, y):
    y_pred = tf.math.greater(logit, 0.5)
    y_pred = tf.to_float(y_pred)
    y_pred_flat = tf.layers.flatten(y_pred)
    y_true_flat = tf.layers.flatten(y)

    intersection = tf.reduce_sum(y_pred_flat * y_true_flat, axis=1) + 1e-7

    denominator = tf.reduce_sum(tf.to_float(tf.cast(y_pred_flat,dtype=tf.bool)|tf.cast(y_true_flat,dtype=tf.bool))) + 1e-7

    iou = tf.reduce_mean(intersection / denominator)

    return iou


def unet(features,labels,mode,params):
    layer9_up = make_unet(features,training= mode==tf.estimator.ModeKeys.TRAIN,flags=0.1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'input':features,
            'mask_pred':layer9_up
        }
        return tf.estimator.EstimatorSpec(mode,predictions=predictions)

    tf.summary.image('PREDICTED_MASK',layer9_up)
    tf.summary.image('INPUT_IMAGE', features)
    tf.summary.image('GT_MASK',labels)

    #print 'Features shape %s, logits shape %s, labels shape %s'%(features.shape,layer9_up.shape,labels.shape)
    labels_bg = 1-labels
    combined_mask = tf.concat(axis=3,values=[labels,labels_bg])
    flat_mask = tf.reshape(combined_mask,(-1,2))
    layer9_up_bg = 1-layer9_up
    combined_logits = tf.concat(axis=3,values=[layer9_up,layer9_up_bg])
    flat_logits = tf.reshape(combined_logits,(-1,2))
    loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=flat_mask,logits=flat_logits))
    loss_dice = 1-Dice(logit=layer9_up,y=labels)
    loss_iou = 1-IoU(logit=layer9_up,y=labels)

    tf.summary.scalar('Cross Entropy Loss: ', loss_ce)
    tf.summary.scalar('Dice Loss: ',loss_dice)
    tf.summary.scalar('IOU Loss: ',loss_iou)
    #mean_iou, update_op_iou = tf.metrics.mean_iou(labels=labels,predictions=layer9_up, num_classes=1, name='acc_op')
    #tf.summary.scalar('MEAN_IOU', mean_iou)



    if mode == tf.estimator.ModeKeys.EVAL:
        predictions = {
            'gt': labels,
            'input': features,
            'mask_pred': layer9_up
        }
        return tf.estimator.EstimatorSpec(mode,loss=loss_ce,predictions=predictions)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

    global_step = tf.train.get_global_step()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_ce, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss_ce, train_op=train_op)


def train_val():
    configuration = tf.estimator.RunConfig(
        model_dir=config.model_dir,
        keep_checkpoint_max=config.keep_checkpoint_max,
        save_checkpoints_secs=config.save_checkpoints_secs,
        log_step_count_steps=config.log_step_count_steps)  # set the frequency of logging steps for loss function

    rec = Recorder()
    classifier = tf.estimator.Estimator(model_fn=unet, params={'learning_rate': 0.001}, config=configuration)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: rec.imgs_input_fn(config.tfrecord_file_train,resize_height=config.resize_height,resize_width=config.resize_width,shuffle=False,repeat_count=-1),
                                        max_steps=config.max_steps,
                                        )
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: rec.imgs_input_fn(config.tfrecord_file_val,resize_height=config.resize_height,resize_width=config.resize_width,shuffle=False,repeat_count=1))
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

def evaluate_and_save(model_dir,val_tfrecord_file,output_folder):
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        )  # set the frequency of logging steps for loss function

    rec = Recorder()
    classifier = tf.estimator.Estimator(model_fn=unet,config=configuration)
    predictions = classifier.evaluate(input_fn=lambda: rec.imgs_input_fn(val_tfrecord_file,resize_height=config.resize_height,resize_width=config.resize_width,shuffle=False,repeat_count=1))
    dice_acc = 0.0
    slice_num = 0.0
    count = 0
    weight = 0.5
    for prediction in predictions:
        input_img = prediction['input']
        mask_pred = prediction['mask_pred']
        labels = prediction['gt']
        mask_pred[mask_pred>0.05] = 255
        mask_pred[mask_pred<=0.05] = 0

        color_pred = np.zeros((input_img.shape[0],input_img.shape[1],3))
        ind = np.squeeze(mask_pred>0)
        color_pred[ind] = (0,0,255)

        # calculate dice
        mask_flat = masked_pred.flatten()/255.0
        label_flat = labels.flatten()
        intersection = np.multiply(mask_flat,label_flat)
        denominator = np.sum(mask_flat)+np.sum(label_flat)
        dice_coef = 2*intersection/denominator
        print('dice: {0}'.format(dice_coef))

        dice_acc += dice_coef
        slice_num += 1


        masked_pred = weight*input_img+(1-weight)*color_pred


        pred_file_path = os.path.join(output_folder,'{:04d}_pred.png'.format(count))
        count += 1

        cv2.imwrite(pred_file_path,masked_pred)
    print('Totoal Slice: {0}'.format(slice_num))
    print('Dice Average: {0}'.format(dice_acc/slice_num))

def eval_and_save_v2(model_dir,val_tfrecord_file,output_folder,batch_size=32):
    """
    eval the model, save the mean dice, number of slice and prediction results
    written by zhe zhu, 08/Oct/2019
    :return:
    """
    rec = Recorder()
    features, labels = rec.imgs_input_fn(val_tfrecord_file,resize_height=config.resize_height,resize_width=config.resize_width,shuffle=False,repeat_count=1)
    predictions = unet(features=features,
                       labels=labels,
                       mode=tf.estimator.ModeKeys.EVAL,
                       params={}).predictions
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver.restore(sess,ckpt.model_checkpoint_path)

        pred_list = []
        gt_list = []

        while True:
            try:
                preds,gts = sess.run([predictions,labels])
                pred_list.append(preds)
                gt_list.append(gts)
            except tf.errors.OutOfRangeError:
                break
    # Save the results
    slice_num = (len(pred_list)-1)*batch_size+len(pred_list[-1]['gt'])
    dice_list = []
    slice_idx = 0
    for preds in pred_list:
        mask_pred = preds['mask_pred']
        mask_pred[mask_pred > 0.5] = 1.0
        mask_pred[mask_pred <= 0.5] = 0
        gt = preds['gt']
        gt[gt>0.5] = 1.0
        for i in range(gt.shape[0]):
            mask_flat = mask_pred[i,:,:,:].flatten()
            label_flat = gt[i,:,:,:].flatten()
            intersection = np.sum(np.multiply(mask_flat, label_flat))
            denominator = np.sum(mask_flat) + np.sum(label_flat)
            dice_coef = (2 * intersection+1e-8) / (denominator+1e-8)
            dice_list.append(dice_coef)

            # save the images
            slice_ori = preds['input'][i,:,:,:]
            slice_ori_filename = '{:04d}_ori.png'.format(slice_idx)
            slice_ori_filepath = os.path.join(output_folder,slice_ori_filename)
            cv2.imwrite(slice_ori_filepath,slice_ori)

            pred_slice = slice_ori #np.concatenate((slice_ori,slice_ori,slice_ori),axis=2)
            gt_slice = np.copy(pred_slice)
            mask8ui_pred = mask_pred[i,:,:,:].astype(np.uint8)
            pred_cnts, hierarchy = cv2.findContours(mask8ui_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in pred_cnts:
                cv2.drawContours(pred_slice, [c], -1, (0, 255, 0), 2)
            pred_filename = '{:04d}_pred.png'.format(slice_idx)
            pred_filepath = os.path.join(output_folder,pred_filename)
            cv2.imwrite(pred_filepath,pred_slice)

            mask8ui_gt = gt[i,:,:,:].astype(np.uint8)
            gt_cnts, hierarchy = cv2.findContours(mask8ui_gt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for c in gt_cnts:
                cv2.drawContours(gt_slice,[c],-1,(255,0,0), 2)
            gt_filename = '{:04d}_gt.png'.format(slice_idx)
            gt_filepath = os.path.join(output_folder,gt_filename)
            cv2.imwrite(gt_filepath,gt_slice)
            slice_idx += 1
    print("Eval Finished!")
    print(np.sum(dice_list)/slice_num)

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    rec = Recorder()
    train_outpath='/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/tfrecord/dynpre_train_2d.tfrecord'

    configuration = tf.estimator.RunConfig(
                                   model_dir = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/models/dynpre',
                                   keep_checkpoint_max=3,
                                   save_checkpoints_secs = 100,
                                   log_step_count_steps=10)  # set the frequency of logging steps for loss function

    classifier = tf.estimator.Estimator(model_fn = unet, params = {'learning_rate' : 0.001}, config=configuration)

    classifier.train(input_fn = lambda:rec.imgs_input_fn(train_outpath,resize_height=128,resize_width=128,shuffle=False,repeat_count=-1), steps=500)
    #print classifier


def test():
    features = tf.placeholder(tf.float32,[None,128,128,3])
    labels = tf.placeholder(tf.float32,[None,128,128,1])
    mode = tf.estimator.ModeKeys.TRAIN
    params = {'learning_rate':0.001}

    model = unet(features,labels,mode,params)


def test_1():
    """
    Save the results of exp1, on test_inv
    written by zhe zhu, on 08/Oct/2019

    """
    model_dir = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/models/maciej/exp1/ce'
    val_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/intermediate/test_inv.tfrecord'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/output/exp1_ce_inv'
    eval_and_save_v2(model_dir,val_tfrecord_file,output_folder)

def test_2(model_dir,val_tfrecord_file,output_folder):
    """
    Use command line input for evaluation
    written by zhe zhu 09/Oct/2019
    :return:
    """
    eval_and_save_v2(model_dir, val_tfrecord_file, output_folder)

def test_3():
    '''Eva the results of exp1218 and exp1219
    Zhe Zhu, 20191231'''
    #eval_and_save_v2('/mnt/sdc/ShapeTexture/models/1218', '/mnt/sdc/ShapeTexture/simulation_data/1218/val.tfrecord', '/mnt/sdc/ShapeTexture/models/1218_eva')
    eval_and_save_v2('/mnt/sdc/ShapeTexture/models/1219', '/mnt/sdc/ShapeTexture/simulation_data/1219/val.tfrecord', '/mnt/sdc/ShapeTexture/models/1219_eva')
if __name__=='__main__':
    #model_dir = sys.argv[1]
    #val_tfrecord_file = sys.argv[2]
    #output_folder = sys.argv[3]
    #test_2(model_dir=model_dir,val_tfrecord_file=val_tfrecord_file,output_folder=output_folder)
    #main()
    #test()
    #os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    #os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_device_id
    #train_val()
    #model_dir = '/home/zzhu/Data/Liver/Segmentation/maciej_exp2'
    #val_tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/exp2/train.tfrecord'
    #output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Segmentation/maciej/output/exp2_train'
    #evaluate_and_save(model_dir,val_tfrecord_file,output_folder)
    test_3()





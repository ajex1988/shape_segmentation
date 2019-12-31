import tensorflow as tf
import sys
#from PIL import Image
import numpy as np
import cv2
import glob

class Recorder:


 def wrap_bytes(self,value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


 def convert(self,image_dir,mask_dir,out_path,size=None):
    '''Args:
    image_paths : List of file-paths for the images
    labels : class-labels for images(vector of size len(image_paths)X1)
    out_path : Destination of TFRecords output file
    size : expected images size
    '''
    
    image_paths = glob.glob(image_dir+'/*.*')
    mask_paths = glob.glob(mask_dir+'/*.*')
    num_images = len(image_paths)

    
    with tf.python_io.TFRecordWriter(out_path) as writer :

       for i,(path,mask) in enumerate(zip(image_paths,mask_paths)):

           sys.stdout.write('\rProcessed :: {} outof ::{}'.format(i+1,num_images))
           sys.stdout.flush()

           #img = Image.open(path)
           img = cv2.imread(path)
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           #img = np.array(img.resize((128,128)))
           image_bytes = img.tostring()


           msk = cv2.imread(mask,0)
           #msk = (msk > 200).astype(np.unit8) * 255.0
           msk_bytes = msk.tostring()
           # dictionary to store image data and image label



           '''image_file = tf.read_file(path)
           mask_file = tf.read_file(mask)

           image = tf.image.decode_jpeg(image_file, channels=3)
           image.set_shape([128, 128, 3])
           image = tf.cast(image, tf.float32)

           msk = tf.image.decode_jpeg(mask_file, channels=1)
           msk.set_shape([128, 128, 1])
           msk = tf.cast(mask, tf.float32)
           msk = mask / (tf.reduce_max(mask) + 1e-7)'''




           data = \
             {
               'image':self.wrap_bytes(image_bytes),
               'mask':self.wrap_bytes(msk_bytes)
             }

           # Wrap the data as Tensorflow Feature.
           feature = tf.train.Features(feature=data)

           # Wrap again as a Tensorflow Example.
           example = tf.train.Example(features=feature)

           # Serialize the data
           serialized = example.SerializeToString()

           # Write the serialized 
           writer.write(serialized)
    print '\nWritten images and mask into {}'.format(out_path)


 def imgs_input_fn(self,filenames,resize_height,resize_width,shuffle=False,repeat_count=-1,batch_size=32):

    def _parse_function(serialized,resize_height=512,resize_width=512):
       features = {
            'file_name': tf.FixedLenFeature([],tf.string,''),
            'height': tf.FixedLenFeature([],tf.int64,0),
            'width': tf.FixedLenFeature([],tf.int64,0),
            'image': tf.FixedLenFeature([],tf.string,''),
            'mask': tf.FixedLenFeature([],tf.string,'')
        }

       parsed_example = tf.parse_single_example(serialized=serialized, features=features)
       height = parsed_example['height']
       width = parsed_example['width']
       image_shape = tf.stack([height,width,3])
       mask_shape = tf.stack([height,width,1])

       image_raw = parsed_example['image']
       mask_raw = parsed_example['mask']

       # decode the raw bytes so it becomes a tensor with type

       image = tf.decode_raw(image_raw,tf.float64)
       image = tf.cast(image,tf.float32)
       image = tf.reshape(image, image_shape)
       image = tf.div(tf.subtract(image,
                                tf.reduce_min(image) + 1e-8),
                    tf.subtract(tf.reduce_max(image),
                                tf.reduce_min(image)) + 1e-8) * 255.0
       image = tf.image.resize_images(image,(resize_height,resize_width))
   
       mask = tf.decode_raw(mask_raw,tf.float64)
       mask = tf.cast(mask, tf.float32)
       mask = tf.reshape(mask, mask_shape)
       mask = mask/(tf.reduce_max(mask)+1e-7)
       mask = tf.image.resize_images(mask,(resize_height,resize_width))

       #d={'input':image,'mask': mask}
       d = image,mask
       return d
    
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialised data to TFRecords files.
    # returns Tensorflow tensors for the image and labels.
    dataset = dataset.map(_parse_function)

    if shuffle:
        dataset = dataset.shuffle(buffer_size = 256)
    
    dataset = dataset.repeat(repeat_count) # Repeat the dataset this time
    batch_dataset = dataset.batch(batch_size)    # Batch Size
    iterator = batch_dataset.make_one_shot_iterator()   # Make an iterator
    batch_features,batch_labels = iterator.get_next()  # Tensors to get next batch of image and their labels
    
    return batch_features, batch_labels

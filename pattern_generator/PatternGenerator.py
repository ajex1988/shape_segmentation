import cv2
import math
import numpy as np
import os
import tensorflow as tf
import glob
import shutil
from random import shuffle
import random

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def dataset_decode_2d(serialized, resize_height=512, resize_width=512):
    features = {
        'file_name': tf.FixedLenFeature([], tf.string, ''),
        'height': tf.FixedLenFeature([], tf.int64, 0),
        'width': tf.FixedLenFeature([], tf.int64, 0),
        'image': tf.FixedLenFeature([], tf.string, ''),
        'mask': tf.FixedLenFeature([], tf.string, '')
    }

    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    height = parsed_example['height']
    width = parsed_example['width']
    image_shape = tf.stack([height, width, 3])
    mask_shape = tf.stack([height, width, 1])

    image_raw = parsed_example['image']
    mask_raw = parsed_example['mask']

    # decode the raw bytes so it becomes a tensor with type

    image = tf.decode_raw(image_raw, tf.float64)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, image_shape)
    image = tf.div(tf.subtract(image,
                        tf.reduce_min(image) + 1e-8),
                       tf.subtract(tf.reduce_max(image),
                                   tf.reduce_min(image)) + 1e-8) * 255.0
    image = tf.image.resize_images(image, (resize_height, resize_width))

    mask = tf.decode_raw(mask_raw, tf.float64)
    mask = tf.cast(mask, tf.float32)
    mask = tf.reshape(mask, mask_shape)
    mask = mask / (tf.reduce_max(mask) + 1e-7)
    mask = tf.image.resize_images(mask, (resize_height, resize_width))


    d = image, mask
    return d
class BasicShape:
    def __init__(self):
        self.bbox = {}
        self.bbox['x'] = -0.5
        self.bbox['y'] = 0.5
        self.bbox['width'] = 1.0
        self.bbox['height'] = 1.0

    def update_bbox(self):
        x_min = np.amin(self.x_array)
        x_max = np.amax(self.x_array)
        y_min = np.amin(self.y_array)
        y_max = np.amax(self.y_array)

        self.bbox['x'] = x_min
        self.bbox['y'] = y_min
        self.bbox['width'] = x_max - x_min
        self.bbox['height'] = y_max - y_min

    def check_range(self,threshold=1000):
        for i in range(len(self.x_array)):
            if abs(self.x_array[i]) > threshold:
                print('ERROR')

    def rotate(self,radian):
        rot_mat = np.array([[math.cos(radian), -math.sin(radian)], [math.sin(radian), math.cos(radian)]])

        for i in range(self.num_pts):
            po = np.array([[self.x_array[i]], [self.y_array[i]]])
            pot = np.matmul(rot_mat, po)
            self.x_array[i] = pot[0, 0]
            self.y_array[i] = pot[1, 0]
        self.check_range()
        self.update_bbox()

    def scale(self,rate):
        if rate > 0.0:
            for i in range(self.num_pts):
                self.x_array[i] *= rate
                self.y_array[i] *= rate
                if abs(self.x_array[i]) > 1000 or abs(self.y_array[i])>1000:
                    print('BUG')
            self.check_range()
            self.update_bbox()
        else:
            raise ValueError('Scaling factor should be larger than 0')

    def translate(self,offset_x,offset_y):
        for i in range(self.num_pts):
            self.x_array[i] += offset_x
            self.y_array[i] += offset_y
        self.update_bbox()

class Circle(BasicShape):
    def __init__(self,center_x=0.0, center_y=0.0, radius=0.5):
        BasicShape.__init__(self)
        self.center = {}
        self.center['x'] = center_x
        self.center['y'] = center_y
        self.radius = radius

    def update_bbox(self):
        self.bbox['x'] = self.center['x'] - self.radius
        self.bbox['y'] = self.center['y'] - self.radius
        self.bbox['width'] = self.radius*2
        self.bbox['height'] = self.radius*2

    def rotate(self,rad):
        pass

    def translate(self,offset_x,offset_y):
        self.center['x'] += offset_x
        self.center['y'] += offset_y
        self.update_bbox()

    def scale(self, rate):
        self.radius = self.radius*rate
        self.update_bbox()

class Rectangle(BasicShape):
    def __init__(self,x_array=[-0.5,0.5,0.5,-0.5],y_array=[0.5,0.5,-0.5,-0.5]):
        BasicShape.__init__(self)
        self.x_array = x_array
        self.y_array = y_array
        self.num_pts = 4

class Triangle(BasicShape):
    def __init__(self,x_array=[0.0,-0.57735,0.57735],y_array=[0.5,-0.5,-0.5]):
        BasicShape.__init__(self)
        self.x_array = x_array
        self.y_array = y_array
        self.num_pts = 3

class Star(BasicShape):
    def __init__(self):
        pass


class Pattern:
    """Generate example patterns"""
    def __init__(self,shape_type='circle',texture=np.zeros((512,512,3))):
        self.shape_type = shape_type
        self.texture = texture
        self.generate_standard_shape()

    # generate a zero-centered shape
    def generate_standard_shape(self):
        if self.shape_type == 'circle':
            self.shape = Circle(center_x=0.0, center_y=0.0, radius=0.5)
        elif self.shape_type == 'rect':
            self.shape = Rectangle(x_array=[-0.5,0.5,0.5,-0.5],y_array=[0.5,0.5,-0.5,-0.5])
        elif self.shape_type == 'triangle':
            self.shape = Triangle(x_array=[0.0,-0.57735,0.57735],y_array=[0.5,-0.5,-0.5])
        elif self.shape_type == 'star':
            self.shape = Star()
        else:
            raise ValueError('Invalid shape type')

    # shape transformation
    def scale(self,scale_rate):
        self.shape.scale(scale_rate)

    def rotate(self,rot_radian):
        self.shape.rotate(rot_radian)

    def translate(self,trans_x,trans_y):
        self.shape.translate(offset_x=trans_x,offset_y=trans_y)

    def transform_shape(self,scale_rate,rot_radian,trans_x,trans_y):
        self.shape.scale(scale_rate)
        self.shape.rotate(rot_radian)
        self.shape.translate(trans_x,trans_y)

    # texture transformation
    def add_noise(self, noise_typ, param):
        if len(self.texture.shape) == 2:
            row, col = self.texture.shape
            ch = 1
        else:
            row, col, ch = self.texture.shape
        if noise_typ == "gauss":
            mean = param['mean']
            var = param['var']
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col)
            self.texture = self.texture + gauss
        elif noise_typ == "s&p":
            #row, col, ch = self.texture.shape
            s_vs_p = param['s_vs_p']
            amount = param['amount']
            out = np.copy(self.texture)
            # Salt mode
            num_salt = np.ceil(amount * self.texture.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in self.texture.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * self.texture.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in self.texture.shape]
            out[coords] = 0
            self.texture = out

        elif noise_typ == "poisson":
            vals = len(np.unique(self.texture))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(self.texture * vals) / float(vals)
            self.texture = noisy

        elif noise_typ == "speckle":
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = self.texture + self.texture * gauss
            self.texture = noisy

    def get_bbox(self):
        return self.shape.bbox

class ContextPatternTRR(Pattern):
    # implement the basic context shape combination suggested by maciej. A triangle between two rects
    def __init__(self,triangle, triangle_t, rect_l, rect_l_t, rect_r,rect_r_t):
        self.triangle = triangle
        self.rect_l = rect_l
        self.rect_r = rect_r

        self.triangle_t = triangle_t
        self.rect_l_t = rect_l_t
        self.rect_r_t = rect_r_t

        self.shape_type = 'trr'

    # the following transform should be performed in order, scale, rotate, translate
    def scale(self,scale_rate):
        self.scale_rate = scale_rate

        self.triangle.scale(scale_rate)
        self.rect_l.scale(30)
        self.rect_r.scale(30)

        # then move rect_l and rect_r to their position
        self.rect_l.translate(-(0.75 * self.scale_rate+30), 0.0)
        self.rect_r.translate(0.75 * (self.scale_rate+30), 0.0)

    def rotate(self,rot_radian):
        # only rotate the triangle
        self.triangle.rotate(rot_radian)

    def translate(self,trans_x,trans_y):
        # move them as a whole
        self.triangle.translate(trans_x,trans_y)
        self.rect_l.translate(trans_x,trans_y)
        self.rect_r.translate(trans_x,trans_y)

    def get_bbox(self):
        bbox = {}
        bbox['x'] = min([self.triangle.bbox['x'], self.rect_l.bbox['x'], self.rect_r.bbox['x']])
        bbox['y'] = min([self.triangle.bbox['y'], self.rect_l.bbox['y'], self.rect_r.bbox['y']])
        x_max = max([self.triangle.bbox['x']+self.triangle.bbox['width'],
                     self.rect_l.bbox['x']+self.rect_l.bbox['width'],
                     self.rect_r.bbox['x']+self.rect_r.bbox['width']])
        y_max = max([self.triangle.bbox['y']+self.triangle.bbox['height'],
                     self.rect_l.bbox['y']+self.rect_l.bbox['height'],
                     self.rect_r.bbox['y']+self.rect_r.bbox['height']])
        bbox['height'] = y_max - bbox['y']
        bbox['width'] = x_max - bbox['x']
        return bbox

class Layer:
    """Layer for rendering. Similar as in the PS"""
    def __init__(self,height, width, pattern, is_gray=True):
        self.height = height
        self.width = width
        self.pattern = pattern

        if is_gray:
            self.rasterized = np.zeros(shape=(self.height, self.width), dtype=np.float) * 255.0
        else:
            self.rasterized = np.zeros(shape=(self.height,self.width,3),dtype=np.float)*255.0
        self.mask = np.zeros(shape=(self.height,self.width),dtype=np.float)

    @classmethod
    def union_mask(cls,mask_tgt, mask_src):
        for i in range(mask_src.shape[0]):
            for j in range(mask_src.shape[1]):
                if mask_src[i,j] != 0:
                    mask_tgt[i,j] = mask_src[i,j]
        return mask_tgt

    def mask_union(self,mask_tgt, mask_src):
        # union mask_tgt, mask_src -> mask_tgt
        for i in range(mask_src.shape[0]):
            for j in range(mask_src.shape[1]):
                if mask_src[i,j] != 0:
                    mask_tgt[i,j] = mask_src[i,j]
        return mask_tgt

    def add_pattern(self,pattern_img,pattern_mask):
        for i in range(self.height): # use loop, can be improved using masked copy. TODO
            for j in range(self.width):
                if pattern_mask[i,j]:
                    self.rasterized[i,j] = pattern_img[i,j]
        self.mask = self.mask_union(self.mask,pattern_mask)
    def render(self):

        pattern_mask = np.zeros(shape=(self.height,self.width),dtype=np.float)
        if self.pattern.shape_type == 'circle':
            cv2.circle(img=pattern_mask,center=(int(self.pattern.shape.center['x']),int(self.pattern.shape.center['y'])),radius=int(self.pattern.shape.radius),
                        color=(1,1,1),thickness=-1)
        else:
            pts = np.vstack((self.pattern.shape.x_array,self.pattern.shape.y_array)).T
            cv2.fillPoly(img=pattern_mask,pts=[pts.astype(np.int)],
                             color=(1,1,1))
        # debug
        #print(self.pattern.shape_type)
        #cv2.waitKey(0)
        #cv2.imshow('test',pattern_mask)

        pattern_img = self.pattern.texture
        self.add_pattern(pattern_img,pattern_mask)

class ContextLayerTRR(Layer):
    # used for context pattern rect-triangle-rect
    def __init__(self,height, width, pattern):
        self.height = height
        self.width = width
        self.pattern = pattern

        self.rasterized = np.zeros(shape=(self.height, self.width, 3), dtype=np.float) * 255.0
        self.triangle_mask = np.zeros(shape=(self.height, self.width), dtype=np.float)
        self.rect_l_mask = np.zeros(shape=(self.height, self.width), dtype=np.float)
        self.rect_r_mask = np.zeros(shape=(self.height, self.width), dtype=np.float)

        self.mask = np.zeros(shape=(self.height, self.width), dtype=np.float)

    def render(self):
        pts = np.vstack((self.pattern.triangle.x_array, self.pattern.triangle.y_array)).T
        cv2.fillPoly(img=self.triangle_mask, pts=[pts.astype(np.int)],
                     color=(1, 1, 1)) # triangle
        pts = np.vstack((self.pattern.rect_l.x_array, self.pattern.rect_l.y_array)).T
        cv2.fillPoly(img=self.rect_l_mask, pts=[pts.astype(np.int)],
                     color=(1, 1, 1))
        pts = np.vstack((self.pattern.rect_r.x_array, self.pattern.rect_r.y_array)).T
        cv2.fillPoly(img=self.rect_r_mask, pts=[pts.astype(np.int)],
                     color=(1, 1, 1))

        for i in range(self.height): # Triangle
            for j in range(self.width):
                if self.triangle_mask[i,j] or self.rect_l_mask[i,j] or self.rect_r_mask[i,j]:
                    self.rasterized[i,j] = self.pattern.triangle_t[i,j]
        for i in range(self.height): # Rectangle left
            for j in range(self.width):
                if self.rect_l_mask[i,j]:
                    self.rasterized[i,j] = self.pattern.rect_l_t[i,j]
        for i in range(self.height): # Rectangle right
            for j in range(self.width):
                if self.rect_r_mask[i,j]:
                    self.rasterized[i,j] = self.pattern.rect_r_t[i,j]
        self.mask = Layer.union_mask(self.mask,self.triangle_mask)
        self.mask = Layer.union_mask(self.mask,self.rect_l_mask)
        self.mask = Layer.union_mask(self.mask,self.rect_r_mask)



class ImageGenerator:
    """Generate an example image containing several patterns"""
    shape_list = ['circle',
                  'rect',
                  'triangle']  # supported basic shapes
    def __init__(self,texture_folder, texture_img_type,img_height, img_width, layer_num_max, scale_min, scale_max, offset_min, offset_max, overlap_rate=0.0):
        self.img_width = img_width
        self.img_height = img_height
        self.layer_num_max = layer_num_max
        self.layers = []
        self.textures = []

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset_min = offset_min
        self.offset_max = offset_max

        self.min_dim = 20

        self.overlap_rate = overlap_rate

        # load texture
        for i in range(len(self.shape_list)):
            texture_img_name = self.shape_list[i]+'.'+texture_img_type
            texture_file = os.path.join(texture_folder,texture_img_name)
            texture = cv2.imread(texture_file)
            self.textures.append(texture)

    def add_layer(self,layer):
        self.layers.append(layer)

    def generate_layout(self):
        # core function
        layer_num = np.random.randint(low=1,high=self.layer_num_max)
        #layer_num = 2 # debug

        h_min = 0
        h_max = self.img_height
        w_min = 0
        w_max = self.img_width

        for i in range(layer_num):
            shape_type_idx = np.random.randint(len(self.shape_list))
            #shape_type_idx = 1 # debug
            shape_type = self.shape_list[shape_type_idx]
            texture = self.textures[shape_type_idx]
            pattern = Pattern(shape_type=shape_type,
                              texture=texture,
                              )
            layer = Layer(height=self.img_height,
                          width=self.img_width,
                          pattern=pattern)
            h_scale_allowd = (h_max-h_min)/2
            w_scale_allowd = (w_max-w_min)/2
            scale_allowed = min(h_scale_allowd,w_scale_allowd)
            if self.scale_min<scale_allowed<self.scale_max:
                scale_rate = np.random.uniform(low=self.scale_min,high=scale_allowed)
            elif scale_allowed > self.scale_max:
                scale_rate = np.random.uniform(low=self.scale_min,high=self.scale_max)
            else:
                scale_rate = np.random.uniform(low=1.0,high=scale_allowed)
            rot_radian = np.random.uniform(low=0.0,high=2*math.pi)

            layer.pattern.scale(scale_rate)
            layer.pattern.rotate(rot_radian)

            offset_x_low = w_min + 1 - layer.pattern.shape.bbox['x']  # 2 is used for safe truncating
            offset_x_high = w_max - (layer.pattern.shape.bbox['x']+layer.pattern.shape.bbox['width']) - 1
            offset_y_low = h_min + 1 - layer.pattern.shape.bbox['y']
            offset_y_high = h_max - (layer.pattern.shape.bbox['y']+layer.pattern.shape.bbox['height']) - 1

            #debug
            if offset_x_low >= offset_x_high:
                print(offset_x_low,offset_x_high)
            if offset_y_low >= offset_y_high:
                print(offset_y_low,offset_y_high)
            offset_x = np.random.randint(low=offset_x_low,high=offset_x_high)
            offset_y = np.random.randint(low=offset_y_low,high=offset_y_high)

            layer.pattern.shape.translate(offset_x,offset_y)

            to_top = layer.pattern.shape.bbox['y'] + layer.pattern.shape.bbox['height'] - h_min
            to_bottom = h_max - layer.pattern.shape.bbox['y']
            to_left = layer.pattern.shape.bbox['x'] + layer.pattern.shape.bbox['width'] - w_min
            to_right = w_max - layer.pattern.shape.bbox['x']

            min_dis = min([to_top, to_bottom, to_left, to_right])

            if min_dis == to_top:
                h_min = layer.pattern.shape.bbox['y'] + layer.pattern.shape.bbox['height']
            elif min_dis == to_bottom:
                h_max = layer.pattern.shape.bbox['y']
            elif min_dis == to_left:
                w_min = layer.pattern.shape.bbox['x'] + layer.pattern.shape.bbox['width']
            elif min_dis == to_right:
                w_max = layer.pattern.shape.bbox['x']

            layer.render()
            self.add_layer(layer)

            if h_max - h_min < self.min_dim or w_max - w_min < self.min_dim:
                break

    def render(self):
        img = np.ones((self.img_height,self.img_width,3))*255.0
        for layer in self.layers:
            for i in range(self.img_height):
                for j in range(self.img_width):
                    if layer.mask[i,j]:
                        img[i,j] = layer.rasterized[i,j]
        return img

class ImageGenerator_v2:
    '''ImageGenerator version 2
    updates from previous version: use bbox variable from pattern instance in stead of shape instance. Thus, 'context pattern' can be supported
     Written by Zhe Zhu, 11/20/2019'''
    shape_list = ['circle',
                  'rect',
                  'triangle',
                  'trr']  # supported basic shapes & 'context' shape
    name2idx = {'circle':0,
                'rect':1,
                'triangle':2} # for convenience
    texture_list = ['circle',
                  'rect',
                  'triangle'] # texture list only store basic shape textures

    def __init__(self, texture_folder, texture_img_type, img_height, img_width, layer_num_max, scale_min, scale_max,
                 offset_min, offset_max, overlap_rate=0.0):
        self.img_width = img_width
        self.img_height = img_height
        self.layer_num_max = layer_num_max
        self.layers = []
        self.textures = []

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset_min = offset_min
        self.offset_max = offset_max

        self.min_dim = 20

        self.overlap_rate = overlap_rate

        # load texture
        for i in range(len(self.texture_list)):
            texture_img_name = self.texture_list[i] + '.' + texture_img_type
            texture_file = os.path.join(texture_folder, texture_img_name)
            texture = cv2.imread(texture_file)
            self.textures.append(texture)

    def add_layer(self,layer):
        self.layers.append(layer)

    def generate_layout(self):
        # core function
        layer_num = np.random.randint(low=1,high=self.layer_num_max)
        #layer_num = 2 # debug

        h_min = 0
        h_max = self.img_height
        w_min = 0
        w_max = self.img_width

        for i in range(layer_num):
            shape_type_idx = np.random.randint(len(self.shape_list))
            shape_type = self.shape_list[shape_type_idx]
            if shape_type == 'trr':
                triangle = Triangle(x_array=[0.0,-0.57735,0.57735],y_array=[0.5,-0.5,-0.5])
                rect_l = Rectangle(x_array=[-0.5,0.5,0.5,-0.5],y_array=[0.5,0.5,-0.5,-0.5])
                rect_r = Rectangle(x_array=[-0.5,0.5,0.5,-0.5],y_array=[0.5,0.5,-0.5,-0.5])
                pattern = ContextPatternTRR(triangle=triangle,
                                            triangle_t=self.textures[self.name2idx['triangle']],
                                            rect_l=rect_l,
                                            rect_l_t=self.textures[self.name2idx['rect']],
                                            rect_r=rect_r,
                                            rect_r_t=self.textures[self.name2idx['rect']])
                layer = ContextLayerTRR(height=self.img_height,
                                        width=self.img_width,
                                        pattern=pattern)
            else:
                texture = self.textures[shape_type_idx]
                pattern = Pattern(shape_type=shape_type,
                                texture=texture,
                                )
                layer = Layer(height=self.img_height,
                            width=self.img_width,
                            pattern=pattern)
            h_scale_allowd = (h_max-h_min)/2
            w_scale_allowd = (w_max-w_min)/2
            scale_allowed = min(h_scale_allowd,w_scale_allowd)
            if self.scale_min<scale_allowed<self.scale_max:
                scale_rate = np.random.uniform(low=self.scale_min,high=scale_allowed)
            elif scale_allowed > self.scale_max:
                scale_rate = np.random.uniform(low=self.scale_min,high=self.scale_max)
            else:
                scale_rate = np.random.uniform(low=1.0,high=scale_allowed)
            rot_radian = np.random.uniform(low=0.0,high=2*math.pi)

            layer.pattern.scale(scale_rate)
            layer.pattern.rotate(rot_radian)

            offset_x_low = w_min + 1 - layer.pattern.get_bbox()['x']  # 2 is used for safe truncating
            offset_x_high = w_max - (layer.pattern.get_bbox()['x']+layer.pattern.get_bbox()['width']) - 1
            offset_y_low = h_min + 1 - layer.pattern.get_bbox()['y']
            offset_y_high = h_max - (layer.pattern.get_bbox()['y']+layer.pattern.get_bbox()['height']) - 1

            #debug
            if offset_x_low >= offset_x_high:
                print('Invalid! offset_x low {0}, high {1} '.format(offset_x_low,offset_x_high))
                continue
            if offset_y_low >= offset_y_high:
                print('Invalid! offset_y low {0}, high {1} '.format(offset_y_low,offset_y_high))
                continue
            print(offset_x_low,offset_x_high,offset_y_low,offset_y_high)
            offset_x = np.random.randint(low=offset_x_low,high=offset_x_high)
            offset_y = np.random.randint(low=offset_y_low,high=offset_y_high)

            layer.pattern.translate(offset_x,offset_y)

            to_top = layer.pattern.get_bbox()['y'] + layer.pattern.get_bbox()['height'] - h_min
            to_bottom = h_max - layer.pattern.get_bbox()['y']
            to_left = layer.pattern.get_bbox()['x'] + layer.pattern.get_bbox()['width'] - w_min
            to_right = w_max - layer.pattern.get_bbox()['x']

            min_dis = min([to_top, to_bottom, to_left, to_right])

            if min_dis == to_top:
                h_min = layer.pattern.get_bbox()['y'] + layer.pattern.get_bbox()['height']
            elif min_dis == to_bottom:
                h_max = layer.pattern.get_bbox()['y']
            elif min_dis == to_left:
                w_min = layer.pattern.get_bbox()['x'] + layer.pattern.get_bbox()['width']
            elif min_dis == to_right:
                w_max = layer.pattern.get_bbox()['x']

            layer.render()
            self.add_layer(layer)

            if h_max - h_min < self.min_dim or w_max - w_min < self.min_dim:
                break

    def render(self):
        img = np.ones((self.img_height,self.img_width,3))*255.0
        for layer in self.layers:
            for i in range(self.img_height):
                for j in range(self.img_width):
                    if layer.mask[i,j]:
                        img[i,j] = layer.rasterized[i,j]
        return img

class ImageGenerator_v3:
    '''
    Generate two patterns: context triangle and pure triangle
    Zhe Zhu 2019/12/09
    '''
    shape_list = ['triangle',
                  'trr']  # supported basic shapes & 'context' shape
    name2idx = {'rect': 0,
                'triangle': 1}  # for convenience
    texture_list = ['rect',
                    'triangle']  # texture list only store basic shape textures

    def __init__(self, texture_folder, texture_img_type, img_height, img_width, layer_num_max, scale_min, scale_max,
                 offset_min, offset_max, overlap_rate=0.0):
        self.img_width = img_width
        self.img_height = img_height
        self.layer_num_max = layer_num_max
        self.layers = []
        self.textures = []

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset_min = offset_min
        self.offset_max = offset_max

        self.min_dim = 20

        self.overlap_rate = overlap_rate

        # load texture
        for i in range(len(self.texture_list)):
            texture_img_name = self.texture_list[i] + '.' + texture_img_type
            texture_file = os.path.join(texture_folder, texture_img_name)
            texture = cv2.imread(texture_file)
            self.textures.append(texture)

    def add_layer(self, layer):
        self.layers.append(layer)

    def generate_layout(self):
        # core function
        layer_num = np.random.randint(low=1, high=self.layer_num_max)
        # layer_num = 2 # debug

        h_min = 0
        h_max = self.img_height
        w_min = 0
        w_max = self.img_width

        for i in range(layer_num):
            shape_type_idx = np.random.randint(len(self.shape_list))
            shape_type = self.shape_list[shape_type_idx]
            if shape_type == 'trr':
                triangle = Triangle(x_array=[0.0, -0.57735, 0.57735], y_array=[0.5, -0.5, -0.5])
                rect_l = Rectangle(x_array=[-0.5, 0.5, 0.5, -0.5], y_array=[0.5, 0.5, -0.5, -0.5])
                rect_r = Rectangle(x_array=[-0.5, 0.5, 0.5, -0.5], y_array=[0.5, 0.5, -0.5, -0.5])
                pattern = ContextPatternTRR(triangle=triangle,
                                            triangle_t=self.textures[self.name2idx['triangle']],
                                            rect_l=rect_l,
                                            rect_l_t=self.textures[self.name2idx['rect']],
                                            rect_r=rect_r,
                                            rect_r_t=self.textures[self.name2idx['rect']])
                layer = ContextLayerTRR(height=self.img_height,
                                        width=self.img_width,
                                        pattern=pattern)
            else:
                texture = self.textures[self.name2idx['triangle']]
                pattern = Pattern(shape_type=shape_type,
                                  texture=texture,
                                  )
                layer = Layer(height=self.img_height,
                              width=self.img_width,
                              pattern=pattern)
            h_scale_allowd = (h_max - h_min) / 2
            w_scale_allowd = (w_max - w_min) / 2
            scale_allowed = min(h_scale_allowd, w_scale_allowd)
            if self.scale_min < scale_allowed < self.scale_max:
                scale_rate = np.random.uniform(low=self.scale_min, high=scale_allowed)
            elif scale_allowed > self.scale_max:
                scale_rate = np.random.uniform(low=self.scale_min, high=self.scale_max)
            else:
                scale_rate = np.random.uniform(low=1.0, high=scale_allowed)
            rot_radian = np.random.uniform(low=0.0, high=2 * math.pi)

            layer.pattern.scale(scale_rate)
            layer.pattern.rotate(rot_radian)

            offset_x_low = w_min + 1 - layer.pattern.get_bbox()['x']  # 2 is used for safe truncating
            offset_x_high = w_max - (layer.pattern.get_bbox()['x'] + layer.pattern.get_bbox()['width']) - 1
            offset_y_low = h_min + 1 - layer.pattern.get_bbox()['y']
            offset_y_high = h_max - (layer.pattern.get_bbox()['y'] + layer.pattern.get_bbox()['height']) - 1

            # debug
            if offset_x_low >= offset_x_high:
                print('Invalid! offset_x low {0}, high {1} '.format(offset_x_low, offset_x_high))
                continue
            if offset_y_low >= offset_y_high:
                print('Invalid! offset_y low {0}, high {1} '.format(offset_y_low, offset_y_high))
                continue
            print(offset_x_low, offset_x_high, offset_y_low, offset_y_high)
            offset_x = np.random.randint(low=offset_x_low, high=offset_x_high)
            offset_y = np.random.randint(low=offset_y_low, high=offset_y_high)

            layer.pattern.translate(offset_x, offset_y)

            to_top = layer.pattern.get_bbox()['y'] + layer.pattern.get_bbox()['height'] - h_min
            to_bottom = h_max - layer.pattern.get_bbox()['y']
            to_left = layer.pattern.get_bbox()['x'] + layer.pattern.get_bbox()['width'] - w_min
            to_right = w_max - layer.pattern.get_bbox()['x']

            min_dis = min([to_top, to_bottom, to_left, to_right])

            if min_dis == to_top:
                h_min = layer.pattern.get_bbox()['y'] + layer.pattern.get_bbox()['height']
            elif min_dis == to_bottom:
                h_max = layer.pattern.get_bbox()['y']
            elif min_dis == to_left:
                w_min = layer.pattern.get_bbox()['x'] + layer.pattern.get_bbox()['width']
            elif min_dis == to_right:
                w_max = layer.pattern.get_bbox()['x']

            layer.render()
            self.add_layer(layer)

            if h_max - h_min < self.min_dim or w_max - w_min < self.min_dim:
                break

    def render(self):
        img = np.ones((self.img_height, self.img_width, 3)) * 255.0
        for layer in self.layers:
            for i in range(self.img_height):
                for j in range(self.img_width):
                    if layer.mask[i, j]:
                        img[i, j] = layer.rasterized[i, j]
        return img

class ImageGenerator_v4:
    """
    Generate an example image containing several patterns
    use shape list as parameter so that shapes can be specified
    Zhe Zhu 2019/12/09
    """
    def __init__(self,texture_folder, texture_img_type,img_height, img_width, layer_num_max, scale_min, scale_max, offset_min, offset_max, shape_list, overlap_rate=0.0):
        self.img_width = img_width
        self.img_height = img_height
        self.layer_num_max = layer_num_max
        self.layers = []
        self.textures = []

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset_min = offset_min
        self.offset_max = offset_max

        self.min_dim = 20
        self.shape_list = shape_list
        self.overlap_rate = overlap_rate

        # load texture
        for i in range(len(self.shape_list)):
            texture_img_name = self.shape_list[i]+'.'+texture_img_type
            texture_file = os.path.join(texture_folder,texture_img_name)
            texture = cv2.imread(texture_file)
            self.textures.append(texture)

    def add_layer(self,layer):
        self.layers.append(layer)

    def generate_layout(self):
        # core function
        layer_num = np.random.randint(low=1,high=self.layer_num_max)
        #layer_num = 2 # debug

        h_min = 0
        h_max = self.img_height
        w_min = 0
        w_max = self.img_width

        for i in range(layer_num):
            shape_type_idx = np.random.randint(len(self.shape_list))
            #shape_type_idx = 1 # debug
            shape_type = self.shape_list[shape_type_idx]
            texture = self.textures[shape_type_idx]
            pattern = Pattern(shape_type=shape_type,
                              texture=texture,
                              )
            layer = Layer(height=self.img_height,
                          width=self.img_width,
                          pattern=pattern)
            h_scale_allowd = (h_max-h_min)/2
            w_scale_allowd = (w_max-w_min)/2
            scale_allowed = min(h_scale_allowd,w_scale_allowd)
            if self.scale_min<scale_allowed<self.scale_max:
                scale_rate = np.random.uniform(low=self.scale_min,high=scale_allowed)
            elif scale_allowed > self.scale_max:
                scale_rate = np.random.uniform(low=self.scale_min,high=self.scale_max)
            else:
                scale_rate = np.random.uniform(low=1.0,high=scale_allowed)
            rot_radian = np.random.uniform(low=0.0,high=2*math.pi)

            layer.pattern.scale(scale_rate)
            layer.pattern.rotate(rot_radian)

            offset_x_low = w_min + 1 - layer.pattern.shape.bbox['x']  # 2 is used for safe truncating
            offset_x_high = w_max - (layer.pattern.shape.bbox['x']+layer.pattern.shape.bbox['width']) - 1
            offset_y_low = h_min + 1 - layer.pattern.shape.bbox['y']
            offset_y_high = h_max - (layer.pattern.shape.bbox['y']+layer.pattern.shape.bbox['height']) - 1

            #debug
            if offset_x_low >= offset_x_high:
                print(offset_x_low,offset_x_high)
            if offset_y_low >= offset_y_high:
                print(offset_y_low,offset_y_high)
            offset_x = np.random.randint(low=offset_x_low,high=offset_x_high)
            offset_y = np.random.randint(low=offset_y_low,high=offset_y_high)

            layer.pattern.shape.translate(offset_x,offset_y)

            to_top = layer.pattern.shape.bbox['y'] + layer.pattern.shape.bbox['height'] - h_min
            to_bottom = h_max - layer.pattern.shape.bbox['y']
            to_left = layer.pattern.shape.bbox['x'] + layer.pattern.shape.bbox['width'] - w_min
            to_right = w_max - layer.pattern.shape.bbox['x']

            min_dis = min([to_top, to_bottom, to_left, to_right])

            if min_dis == to_top:
                h_min = layer.pattern.shape.bbox['y'] + layer.pattern.shape.bbox['height']
            elif min_dis == to_bottom:
                h_max = layer.pattern.shape.bbox['y']
            elif min_dis == to_left:
                w_min = layer.pattern.shape.bbox['x'] + layer.pattern.shape.bbox['width']
            elif min_dis == to_right:
                w_max = layer.pattern.shape.bbox['x']

            layer.render()
            self.add_layer(layer)

            if h_max - h_min < self.min_dim or w_max - w_min < self.min_dim:
                break

    def render(self):
        img = np.ones((self.img_height,self.img_width,3))*255.0
        for layer in self.layers:
            for i in range(self.img_height):
                for j in range(self.img_width):
                    if layer.mask[i,j]:
                        img[i,j] = layer.rasterized[i,j]
        return img

class ImageGenerator_v5:
    """
    Update: the pattern contains noise
    The background is gray-level
    Zhe Zhu 2020/01/10
    Update: Add a parameter iscolor. if 0 load gray-level image
    """
    def __init__(self,texture_folder, texture_img_type,img_height, img_width, layer_num_max, scale_min, scale_max,
                 offset_min, offset_max, shape_list, noise_type, noise_param, iscolor, overlap_rate=0.0):
        self.img_width = img_width
        self.img_height = img_height
        self.layer_num_max = layer_num_max
        self.layers = []
        self.textures = []
        self.iscolor = iscolor

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.offset_min = offset_min
        self.offset_max = offset_max

        self.min_dim = 20
        self.shape_list = shape_list
        self.overlap_rate = overlap_rate

        self.noise_type = noise_type
        self.noise_param = noise_param

        # load texture
        for i in range(len(self.shape_list)):
            texture_img_name = self.shape_list[i]+'.'+texture_img_type
            texture_file = os.path.join(texture_folder,texture_img_name)
            texture = cv2.imread(texture_file,iscolor)
            self.textures.append(texture)

    def add_layer(self,layer):
        self.layers.append(layer)

    def generate_layout(self):
        # core function
        layer_num = np.random.randint(low=1,high=self.layer_num_max)
        #layer_num = 2 # debug

        h_min = 0
        h_max = self.img_height
        w_min = 0
        w_max = self.img_width

        for i in range(layer_num):
            shape_type_idx = np.random.randint(len(self.shape_list))
            #shape_type_idx = 1 # debug
            shape_type = self.shape_list[shape_type_idx]
            texture = self.textures[shape_type_idx]
            pattern = Pattern(shape_type=shape_type,
                              texture=texture,
                              )
            if self.noise_type == "gauss" or self.noise_type == 's&p' or self.noise_type == 'poisson' or self.noise_type == 'speckle':
                pattern.add_noise(self.noise_type,self.noise_param)
            layer = Layer(height=self.img_height,
                          width=self.img_width,
                          pattern=pattern,
                          is_gray= not self.iscolor)
            h_scale_allowd = (h_max-h_min)/2
            w_scale_allowd = (w_max-w_min)/2
            scale_allowed = min(h_scale_allowd,w_scale_allowd)
            if self.scale_min<scale_allowed<self.scale_max:
                scale_rate = np.random.uniform(low=self.scale_min,high=scale_allowed)
            elif scale_allowed > self.scale_max:
                scale_rate = np.random.uniform(low=self.scale_min,high=self.scale_max)
            else:
                scale_rate = np.random.uniform(low=1.0,high=scale_allowed)
            rot_radian = np.random.uniform(low=0.0,high=2*math.pi)

            layer.pattern.scale(scale_rate)
            layer.pattern.rotate(rot_radian)

            offset_x_low = w_min + 1 - layer.pattern.shape.bbox['x']  # 2 is used for safe truncating
            offset_x_high = w_max - (layer.pattern.shape.bbox['x']+layer.pattern.shape.bbox['width']) - 1
            offset_y_low = h_min + 1 - layer.pattern.shape.bbox['y']
            offset_y_high = h_max - (layer.pattern.shape.bbox['y']+layer.pattern.shape.bbox['height']) - 1

            #debug
            if offset_x_low >= offset_x_high:
                print(offset_x_low,offset_x_high)
            if offset_y_low >= offset_y_high:
                print(offset_y_low,offset_y_high)
            offset_x = np.random.randint(low=offset_x_low,high=offset_x_high)
            offset_y = np.random.randint(low=offset_y_low,high=offset_y_high)

            layer.pattern.shape.translate(offset_x,offset_y)

            to_top = layer.pattern.shape.bbox['y'] + layer.pattern.shape.bbox['height'] - h_min
            to_bottom = h_max - layer.pattern.shape.bbox['y']
            to_left = layer.pattern.shape.bbox['x'] + layer.pattern.shape.bbox['width'] - w_min
            to_right = w_max - layer.pattern.shape.bbox['x']

            min_dis = min([to_top, to_bottom, to_left, to_right])

            if min_dis == to_top:
                h_min = layer.pattern.shape.bbox['y'] + layer.pattern.shape.bbox['height']
            elif min_dis == to_bottom:
                h_max = layer.pattern.shape.bbox['y']
            elif min_dis == to_left:
                w_min = layer.pattern.shape.bbox['x'] + layer.pattern.shape.bbox['width']
            elif min_dis == to_right:
                w_max = layer.pattern.shape.bbox['x']

            layer.render()
            self.add_layer(layer)

            if h_max - h_min < self.min_dim or w_max - w_min < self.min_dim:
                break

    def render(self, gray_level):
        img = np.ones((self.img_height,self.img_width,3))
        img[:,:] = gray_level
        for layer in self.layers:
            for i in range(self.img_height):
                for j in range(self.img_width):
                    if layer.mask[i,j]:
                        img[i,j] = layer.rasterized[i,j]
        return img
    def render_randbg(self, gray_level):
        '''Used to set each pixel a random value
        Zhe Zhu, 2020/02/08'''
        img = np.ones((self.img_height, self.img_width, 3))
        img = gray_level
        for layer in self.layers:
            for i in range(self.img_height):
                for j in range(self.img_width):
                    if layer.mask[i, j]:
                        img[i, j] = layer.rasterized[i, j]
        return img

def img2tfrecord(img_folder,img_num,mode,tfrecord_file):
    #
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for i in range(img_num):
            img_file_name = mode + '_{:04d}.png'.format(i)
            mask_file_name = mode + '_{:04d}_mask.png'.format(i)
            img_path = os.path.join(img_folder, img_file_name)
            mask_path = os.path.join(img_folder, mask_file_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(float)
            mask = mask.astype(float)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'file_name': _bytes_feature(img_file_name),
                        'height': _int64_feature(img.shape[0]),
                        'width': _int64_feature(img.shape[1]),
                        'image': _bytes_feature(img.tostring()),
                        'mask': _bytes_feature(mask.tostring())
                    }
                )
            )
            writer.write(example.SerializeToString())

def img2tfrecord_(img_folder,img_num,mode,tfrecord_file):
    #
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for i in range(img_num):
            img_file_name = mode + '_{:05d}.png'.format(i)
            mask_file_name = mode + '_{:05d}_mask.png'.format(i)
            img_path = os.path.join(img_folder, img_file_name)
            mask_path = os.path.join(img_folder, mask_file_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(float)
            mask = mask.astype(float)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'file_name': _bytes_feature(img_file_name),
                        'height': _int64_feature(img.shape[0]),
                        'width': _int64_feature(img.shape[1]),
                        'image': _bytes_feature(img.tostring()),
                        'mask': _bytes_feature(mask.tostring())
                    }
                )
            )
            writer.write(example.SerializeToString())
def imgs2tfrecord(img_file_list, tfrecord_file):
    '''
    Convert the images (stored in the image_file_list as absolute path) to tfrecord file
    Zhe Zhu, 2020/02/03
    '''
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for img_path in img_file_list:
            img_file_name = os.path.basename(img_path)
            img_dir_name = os.path.dirname(img_path)
            mask_file_name = img_file_name[:-4] + '_mask.png'
            mask_path = os.path.join(img_dir_name,mask_file_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(float)
            mask = mask.astype(float)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'file_name': _bytes_feature(img_path),
                        'height': _int64_feature(img.shape[0]),
                        'width': _int64_feature(img.shape[1]),
                        'image': _bytes_feature(img.tostring()),
                        'mask': _bytes_feature(mask.tostring())
                    }
                )
            )
            writer.write(example.SerializeToString())

def merge_shuffle_folders2tfrecord(folder_list, tfrecord_file):
    '''
    Merge multiple folders, shuffle the images, and convert to tfrecord
    Zhe Zhu, 2020/02/03
    '''
    img_file_list = []
    for folder in folder_list:
        imgs_list = glob.glob(folder+'/*.png')
        for img_file in imgs_list:
            if img_file[-9:-4] != '_mask':
                img_file_list.append(img_file)
    shuffle(img_file_list)
    imgs2tfrecord(img_file_list,tfrecord_file)

def make_texture_folders(img_folder,texture_num,output_folder):
    '''
    Construct texture_num texture folders. This function can be used to construct the texture folders
    to alter the texture of the target shape.
    Zhe Zhu, 2020/02/04
    '''
    tex_img_list = glob.glob(img_folder+'/*')
    if len(tex_img_list) < texture_num:
        raise Exception('Not enough texture images')

    texture_height = 512
    texture_width = 512
    rect_img_path = '/mnt/sdc/ShapeTexture/simulation_data/0120/texture/rect.jpg'
    for i in range(texture_num):
        target_idx = '{:05d}'.format(i)
        target_folder = os.path.join(output_folder,target_idx)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        texture_folder = os.path.join(target_folder,'texture')
        if not os.path.exists(texture_folder):
            os.makedirs(texture_folder)
        tex_img_path = tex_img_list[i]
        tex_img = cv2.imread(tex_img_path)
        if tex_img.shape[0] < texture_height or tex_img.shape[1] < texture_width:
            tex_img = cv2.resize(tex_img,(texture_height,texture_width))
        cv2.imwrite(os.path.join(texture_folder,'triangle.jpg'),tex_img)
        shutil.copyfile(rect_img_path,os.path.join(texture_folder,'rect.jpg'))

def make_texture_folders_v2(img_folder,texture_num,output_folder):
    '''
    Construct texture_num texture folders. This function can be used to construct the texture folders
    to alter the texture of the target shape. Triangle, rectangle and background have different texture
    Zhe Zhu, 2020/02/19
    2020/02/21 bug fix: bg_img should always set to 512*512
    '''
    tex_img_list = glob.glob(img_folder+'/*')
    if len(tex_img_list) < texture_num*3:
        raise Exception('Not enough texture images')

    texture_height = 512
    texture_width = 512
    for i in range(texture_num):
        target_idx = '{:05d}'.format(i)
        target_folder = os.path.join(output_folder,target_idx)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        texture_folder = os.path.join(target_folder,'texture')
        if not os.path.exists(texture_folder):
            os.makedirs(texture_folder)
        triangle_tex_path = tex_img_list[i*3]
        rectangle_tex_path = tex_img_list[i*3+1]
        bg_tex_path = tex_img_list[i*3+2]
        triangle_img = cv2.imread(triangle_tex_path)
        rectangle_img = cv2.imread(rectangle_tex_path)
        bg_img = cv2.imread(bg_tex_path)

        if triangle_img.shape[0] < texture_height or triangle_img.shape[1] < texture_width:
            triangle_img = cv2.resize(triangle_img,(texture_height,texture_width))
        if rectangle_img.shape[0] < texture_height or rectangle_img.shape[1] < texture_width:
            rectangle_img = cv2.resize(rectangle_img,(texture_height,texture_width))

        bg_img = cv2.resize(bg_img,(texture_height,texture_width))
        cv2.imwrite(os.path.join(texture_folder,'triangle.jpg'),triangle_img)
        cv2.imwrite(os.path.join(texture_folder, 'rect.jpg'), rectangle_img)
        cv2.imwrite(os.path.join(texture_folder, 'bg.jpg'), bg_img)

def make_texture_folders_v3(texture_num,output_folder):
    '''
    Construct texture_num texture folders. This function can be used to construct the texture folders
    to alter the texture of the target shape. Triangle, rectangle and background have different texture
    For experiment that use uniform random texture
    Zhe Zhu, 2020/02/24
    '''
    texture_height = 512
    texture_width = 512
    for i in range(texture_num):
        target_idx = '{:05d}'.format(i)
        target_folder = os.path.join(output_folder,target_idx)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        texture_folder = os.path.join(target_folder,'texture')
        if not os.path.exists(texture_folder):
            os.makedirs(texture_folder)

        triangle_img = np.zeros((texture_height,texture_width,3))
        rectangle_img = np.zeros((texture_height,texture_width,3))
        bg_img = np.zeros((texture_height,texture_width,3))

        t_r = random.randint(0,255)
        t_g = random.randint(0,255)
        t_b = random.randint(0,255)
        triangle_img[:,:] = (t_r,t_g,t_b)

        r_r = random.randint(0,255)
        r_g = random.randint(0,255)
        r_b = random.randint(0,255)
        rectangle_img[:,:] = (r_r,r_g,r_b)

        b_r = random.randint(0,255)
        b_g = random.randint(0,255)
        b_b = random.randint(0,255)
        bg_img[:,:] = (b_r,b_g,b_b)

        cv2.imwrite(os.path.join(texture_folder,'triangle.jpg'),triangle_img)
        cv2.imwrite(os.path.join(texture_folder, 'rect.jpg'), rectangle_img)
        cv2.imwrite(os.path.join(texture_folder, 'bg.jpg'), bg_img)

def tfrecord2imgs(tfrecord_file,output_folder):
    '''
    Save the images in tfrecord to a folder
    Debug purposes
    Zhe Zhu. 2020/02/21
    '''
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(dataset_decode_2d)
    iterator = dataset.make_one_shot_iterator()

    next_elem = iterator.get_next()
    i = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while True:
                img,  mask = sess.run(next_elem)
                img_file_path = os.path.join(output_folder, '{:05d}.png'.format(i))
                i += 1
                cv2.imwrite(img_file_path, img)
        except Exception as e:
            print(e)

def test_1():
    # test samples. 11/11/2019
    texture_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/basic_shapes/texture'
    img_height = 256
    img_width = 256
    layer_num_max = 4
    scale_min = 20
    scale_max = 80
    offset_min = 20
    offset_max = 80
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    img_generator = ImageGenerator(texture_folder=texture_folder,
                                   texture_img_type=texture_img_type,
                                   img_height=img_height,
                                   img_width=img_width,
                                   layer_num_max=layer_num_max,
                                   scale_min=scale_min,
                                   scale_max=scale_max,
                                   offset_min=offset_min,
                                   offset_max=offset_max,
                                   overlap_rate=overlap_rate)
    img_generator.generate_layout()
    img = img_generator.render()

    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/basic_shapes/samples'
    output_img = 'test.png'
    output_img_path = os.path.join(output_folder,output_img)
    cv2.imwrite(output_img_path,img)

def test_2():
    # create a dataset for triangle segmentation
    # written by zhe zhu 11/19 2019
    train_sample_num = 2000
    val_sample_num = 100
    train_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1119_samples/train'
    val_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1119_samples/val'

    texture_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/basic_shapes/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 4
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'


    for i in range(train_sample_num):
        img_generator = ImageGenerator(texture_folder=texture_folder,
                                       texture_img_type=texture_img_type,
                                       img_height=img_height,
                                       img_width=img_width,
                                       layer_num_max=layer_num_max,
                                       scale_min=scale_min,
                                       scale_max=scale_max,
                                       offset_min=offset_min,
                                       offset_max=offset_max,
                                       overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height,img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask,layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder,img_file_name)
        mask_path = os.path.join(train_output_folder,mask_file_name)
        cv2.imwrite(img_path,img)
        cv2.imwrite(mask_path,mask)
    for i in range(val_sample_num):
        img_generator = ImageGenerator(texture_folder=texture_folder,
                                       texture_img_type=texture_img_type,
                                       img_height=img_height,
                                       img_width=img_width,
                                       layer_num_max=layer_num_max,
                                       scale_min=scale_min,
                                       scale_max=scale_max,
                                       offset_min=offset_min,
                                       offset_max=offset_max,
                                       overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height,img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask,layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder,img_file_name)
        mask_path = os.path.join(val_output_folder,mask_file_name)
        cv2.imwrite(img_path,img)
        cv2.imwrite(mask_path,mask)
def test_3():
    # build tfrecord
    img_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1119_samples/train'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1119_tfrecord/train.tfrecord'
    img2tfrecord(img_folder,2000,'train',tfrecord_file)

    img_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1119_samples/val'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1119_tfrecord/val.tfrecord'
    img2tfrecord(img_folder, 100, 'val', tfrecord_file)

def test_4():
    # generate some samples to show to maciej
    train_sample_num = 400

    train_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1120_samples/train'

    texture_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/basic_shapes/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 6
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'

    for i in range(train_sample_num):
        img_generator = ImageGenerator(texture_folder=texture_folder,
                                       texture_img_type=texture_img_type,
                                       img_height=img_height,
                                       img_width=img_width,
                                       layer_num_max=layer_num_max,
                                       scale_min=scale_min,
                                       scale_max=scale_max,
                                       offset_min=offset_min,
                                       offset_max=offset_max,
                                       overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height,img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask,layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder,img_file_name)
        mask_path = os.path.join(train_output_folder,mask_file_name)
        cv2.imwrite(img_path,img)
        cv2.imwrite(mask_path,mask)

def test_5():
    '''
    Generate sample containing context samples.
    Written by Zhe Zhu 11/21/2019
    '''
    train_sample_num = 2000
    val_sample_num = 100
    train_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1121_samples/train'
    val_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1121_samples/val'

    texture_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/basic_shapes/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 4
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'

    for i in range(train_sample_num):
        img_generator = ImageGenerator_v2(texture_folder=texture_folder,
                                       texture_img_type=texture_img_type,
                                       img_height=img_height,
                                       img_width=img_width,
                                       layer_num_max=layer_num_max,
                                       scale_min=scale_min,
                                       scale_max=scale_max,
                                       offset_min=offset_min,
                                       offset_max=offset_max,
                                       overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height,img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask,layer.mask)
            elif layer.pattern.shape_type == 'trr':
                mask = layer.mask_union(mask,layer.triangle_mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder,img_file_name)
        mask_path = os.path.join(train_output_folder,mask_file_name)
        cv2.imwrite(img_path,img)
        cv2.imwrite(mask_path,mask)

    for i in range(val_sample_num):
        img_generator = ImageGenerator_v2(texture_folder=texture_folder,
                                       texture_img_type=texture_img_type,
                                       img_height=img_height,
                                       img_width=img_width,
                                       layer_num_max=layer_num_max,
                                       scale_min=scale_min,
                                       scale_max=scale_max,
                                       offset_min=offset_min,
                                       offset_max=offset_max,
                                       overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height,img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask,layer.mask)
            elif layer.pattern.shape_type == 'trr':
                mask = layer.mask_union(mask,layer.triangle_mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder,img_file_name)
        mask_path = os.path.join(val_output_folder,mask_file_name)
        cv2.imwrite(img_path,img)
        cv2.imwrite(mask_path,mask)
def test_6():
    '''
    Regenerate the val data, due to the fault.
    Zhe Zhu, 11/26/2019
    :return:
    '''
    val_sample_num = 100
    val_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1121_samples/val'

    texture_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/basic_shapes/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 4
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    for i in range(val_sample_num):
        img_generator = ImageGenerator_v2(texture_folder=texture_folder,
                                       texture_img_type=texture_img_type,
                                       img_height=img_height,
                                       img_width=img_width,
                                       layer_num_max=layer_num_max,
                                       scale_min=scale_min,
                                       scale_max=scale_max,
                                       offset_min=offset_min,
                                       offset_max=offset_max,
                                       overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height,img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask,layer.mask)
            elif layer.pattern.shape_type == 'trr':
                mask = layer.mask_union(mask,layer.triangle_mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder,img_file_name)
        mask_path = os.path.join(val_output_folder,mask_file_name)
        cv2.imwrite(img_path,img)
        cv2.imwrite(mask_path,mask)

def test_7():
    '''
    Construct the 'context' dataset tfrecord
    Zhe Zhu 11/26/2019
    :return:
    '''
    # build tfrecord
    img_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1121_samples/train'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1121_tfrecord/train.tfrecord'
    img2tfrecord(img_folder, 2000, 'train', tfrecord_file)

    img_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1121_samples/val'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1121_tfrecord/val.tfrecord'
    img2tfrecord(img_folder, 100, 'val', tfrecord_file)

def test_8():
    '''
    construct context dataset
    updated: only generate two shape: context triangle and purely triangle
    Zhe Zhu 2019/12/09
    '''
    train_sample_num = 2000
    val_sample_num = 100
    train_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/train'
    val_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/val'

    texture_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/basic_shapes/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'

    for i in range(train_sample_num):
        img_generator = ImageGenerator_v3(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        img_generator = ImageGenerator_v3(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_9():
    '''
    construct the context dataset
    Zhe Zhu 2019/12/09
    '''
    img_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/train'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_tfrecord/train.tfrecord'
    img2tfrecord(img_folder, 2000, 'train', tfrecord_file)

    img_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/val'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_tfrecord/val.tfrecord'
    img2tfrecord(img_folder, 100, 'val', tfrecord_file)

def test_10():
    '''
    construct the dataset for exp1
    Zhe Zhu 2019/12/09
    '''
    train_sample_num = 2000
    val_sample_num = 100
    train_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/exp1/train'
    val_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/exp1/val'

    texture_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/exp1/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 4
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect','triangle']

    for i in range(train_sample_num):
        img_generator = ImageGenerator_v4(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        img_generator = ImageGenerator_v4(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_11():
    img_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/exp1/train'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_tfrecord/exp1/train.tfrecord'
    img2tfrecord(img_folder, 2000, 'train', tfrecord_file)

    img_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/exp1/val'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_tfrecord/exp1/val.tfrecord'
    img2tfrecord(img_folder, 100, 'val', tfrecord_file)

def test_12():
    # exp2 Zhe Zhu 2019/12/09
    train_sample_num = 2000
    val_sample_num = 100
    train_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/exp2/train'
    val_output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/exp2/val'

    texture_folder_train = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/exp2/texture'
    texture_folder_val = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/exp2/texture_switch'
    img_height = 512
    img_width = 512
    layer_num_max = 4
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    for i in range(train_sample_num):
        img_generator = ImageGenerator_v4(texture_folder=texture_folder_train,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        img_generator = ImageGenerator_v4(texture_folder=texture_folder_val,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_13():
    img_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/exp2/train'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_tfrecord/exp2/train.tfrecord'
    img2tfrecord(img_folder, 2000, 'train', tfrecord_file)

    img_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_samples/exp2/val'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Understanding_CNN/1209_tfrecord/exp2/val.tfrecord'
    img2tfrecord(img_folder, 100, 'val', tfrecord_file)

def test_14():
    '''Correct the bug, triangle as target
    Zhe Zhu, 2019/12/18
    '''
    train_sample_num = 2000
    val_sample_num = 100
    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/1218/train'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/1218/val'

    texture_folder = '/mnt/sdc/ShapeTexture/simulation_data/1218/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'

    for i in range(train_sample_num):
        img_generator = ImageGenerator_v3(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        img_generator = ImageGenerator_v3(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_15():
    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/1218/train'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/1218/train.tfrecord'
    img2tfrecord(img_folder, 2000, 'train', tfrecord_file)

    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/1218/val'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/1218/val.tfrecord'
    img2tfrecord(img_folder, 100, 'val', tfrecord_file)

def test_16():
    '''
    Context triangle as positive
    Zhe Zhu, 2019/12/18
    :return:
    '''
    train_sample_num = 2000
    val_sample_num = 100
    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/1219/train'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/1219/val'

    texture_folder = '/mnt/sdc/ShapeTexture/simulation_data/1219/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'

    for i in range(train_sample_num):
        img_generator = ImageGenerator_v3(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'trr':
                mask = layer.mask_union(mask, layer.triangle_mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        img_generator = ImageGenerator_v3(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render()
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'trr':
                mask = layer.mask_union(mask, layer.triangle_mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_17():
    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/1219/train'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/1219/train.tfrecord'
    img2tfrecord(img_folder, 2000, 'train', tfrecord_file)

    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/1219/val'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/1219/val.tfrecord'
    img2tfrecord(img_folder, 100, 'val', tfrecord_file)

def test_18():
    # noise pattern & gray-level background
    # Zhe Zhu, 2020/01/10
    train_sample_num = 2000
    val_sample_num = 100
    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0110/train'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0110/val'

    texture_folder = '/mnt/sdc/ShapeTexture/simulation_data/0110/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type = 'gauss'
    noise_param = {'mean':0,
                   'var':0.2*255}
    bg_gray_level = 100

    for i in range(train_sample_num):
        img_generator = ImageGenerator_v5(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type,
                                          noise_param=noise_param,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        img_generator = ImageGenerator_v5(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type,
                                          noise_param=noise_param,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)


def test_19():
    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0110/train'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0110/train.tfrecord'
    img2tfrecord(img_folder, 2000, 'train', tfrecord_file)

    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0110/val'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0110/val.tfrecord'
    img2tfrecord(img_folder, 100, 'val', tfrecord_file)

def test_20():
    # noise pattern & gray-level background
    # train & test have different noise degrees
    # Zhe Zhu, 2020/01/14
    train_sample_num = 2000
    val_sample_num = 100
    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0114/train'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0114/val'

    texture_folder = '/mnt/sdc/ShapeTexture/simulation_data/0114/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type = 'gauss'
    noise_param_train = {'mean':0,
                   'var':0.2*255}
    noise_param_val = {'mean': 0,
                         'var': 0.4 * 255}
    bg_gray_level = 100

    for i in range(train_sample_num):
        img_generator = ImageGenerator_v5(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type,
                                          noise_param=noise_param_train,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        img_generator = ImageGenerator_v5(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type,
                                          noise_param=noise_param_val,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)
def test_21():
    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0114/train'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0114/train.tfrecord'
    img2tfrecord(img_folder, 2000, 'train', tfrecord_file)

    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0114/val'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0114/val.tfrecord'
    img2tfrecord(img_folder, 100, 'val', tfrecord_file)

def test_22():
    # noise pattern & gray-level background
    # train & test have different types of noise
    # Zhe Zhu, 2020/01/15
    train_sample_num = 2000
    val_sample_num = 100
    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0115/train'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0115/val'

    texture_folder = '/mnt/sdc/ShapeTexture/simulation_data/0115/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type_train = 'gauss'
    noise_param_train = {'mean':0,
                   'var':0.2*255}
    noise_type_val = 's&p'
    noise_param_val = {'s_vs_p': 0.5,
                         'amount': 0.004}

    bg_gray_level = 100

    for i in range(train_sample_num):
        img_generator = ImageGenerator_v5(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_train,
                                          noise_param=noise_param_train,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        img_generator = ImageGenerator_v5(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_val,
                                          noise_param=noise_param_val,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_23():
    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0115/train'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0115/train.tfrecord'
    img2tfrecord(img_folder, 2000, 'train', tfrecord_file)

    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0115/val'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0115/val.tfrecord'
    img2tfrecord(img_folder, 100, 'val', tfrecord_file)

def test_24():
    # Paper quality pattern texture
    # Segment triangle while ignore rectangle
    # Zhe Zhu, 2020/01/20
    train_sample_num = 2000
    val_sample_num = 100
    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0120/train'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0120/val'

    texture_folder = '/mnt/sdc/ShapeTexture/simulation_data/0120/texture'
    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type_train = 'None'
    noise_param_train = {'mean':0,
                   'var':0.2*255}
    noise_type_val = 'None'
    noise_param_val = {'s_vs_p': 0.5,
                         'amount': 0.004}

    bg_gray_level = (0,172,223) # RGB
    bg_gray_level_bgr = (223,172,0)
    iscolor = 1

    for i in range(train_sample_num):
        img_generator = ImageGenerator_v5(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_train,
                                          noise_param=noise_param_train,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level_bgr)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        img_generator = ImageGenerator_v5(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_val,
                                          noise_param=noise_param_val,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level_bgr)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_25():
    '''
    Experiment 0203: training 50% T texture 50% R texture, same in test
    Then, apply to new dataset, where triangles have totally different texture
    Zhe Zhu, 2020/02/03
    '''
    train_sample_num = 1000
    val_sample_num = 50
    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0203/train_2'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0203/val_2'

    texture_folder = '/mnt/sdc/ShapeTexture/simulation_data/0203/texture_2'
    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type_train = 'None'
    noise_param_train = {'mean': 0,
                         'var': 0.2 * 255}
    noise_type_val = 'None'
    noise_param_val = {'s_vs_p': 0.5,
                       'amount': 0.004}

    bg_gray_level = (0, 172, 223)  # RGB
    bg_gray_level_bgr = (223, 172, 0)
    iscolor = 1

    for i in range(train_sample_num):
        img_generator = ImageGenerator_v5(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_train,
                                          noise_param=noise_param_train,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level_bgr)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:04d}.png'.format(i)
        mask_file_name = 'train_{:04d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        img_generator = ImageGenerator_v5(texture_folder=texture_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_val,
                                          noise_param=noise_param_val,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level_bgr)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:04d}.png'.format(i)
        mask_file_name = 'val_{:04d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_26():
    '''Merge train_1 train_2 and val_1 val_2
    Zhe Zhu 2020/02/03'''
    train_folder_list = ['/mnt/sdc/ShapeTexture/simulation_data/0203/train_1',
                         '/mnt/sdc/ShapeTexture/simulation_data/0203/train_2']
    train_tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0203/train.tfrecord'
    val_folder_list = ['/mnt/sdc/ShapeTexture/simulation_data/0203/val_1',
                       '/mnt/sdc/ShapeTexture/simulation_data/0203/val_2']
    val_tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0203/val.tfrecord'
    merge_shuffle_folders2tfrecord(train_folder_list,train_tfrecord_file)
    merge_shuffle_folders2tfrecord(val_folder_list,val_tfrecord_file)

def test_27():
    '''Mask texture folders
    Zhe Zhu, 2020/02/04'''
    src_img_folder = '/mnt/sdc/dataset/train2014' # coco dataset
    tex_img_num = 20500
    tgt_tex_folder = '/mnt/sdc/ShapeTexture/simulation_data/coco_texture'
    make_texture_folders(src_img_folder,tex_img_num,tgt_tex_folder)

def test_28():
    '''
    coco texture for training and test
    Zhe Zhu, 2020/02/04
    '''
    train_sample_num = 20000
    val_sample_num = 500

    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0204/train'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0204/val'

    textures_folder = '/mnt/sdc/ShapeTexture/simulation_data/coco_texture'

    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type_train = 'None'
    noise_param_train = {'mean': 0,
                         'var': 0.2 * 255}
    noise_type_val = 'None'
    noise_param_val = {'s_vs_p': 0.5,
                       'amount': 0.004}

    bg_gray_level = (0, 172, 223)  # RGB
    bg_gray_level_bgr = (223, 172, 0)
    iscolor = 1

    for i in range(train_sample_num):
        tex_folder = os.path.join(textures_folder,'{:05d}'.format(i),'texture')
        img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_train,
                                          noise_param=noise_param_train,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level_bgr)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:05d}.png'.format(i)
        mask_file_name = 'train_{:05d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        tex_folder = os.path.join(textures_folder,'{:05d}'.format(i+train_sample_num),'texture')
        img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_val,
                                          noise_param=noise_param_val,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level_bgr)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:05d}.png'.format(i)
        mask_file_name = 'val_{:05d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_29():
    '''build the dataset where triangles have coco textures
    Zhe Zhu, 2020/02/04
    '''
    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0204/train'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0204/train.tfrecord'
    img2tfrecord_(img_folder, 20000, 'train', tfrecord_file)

    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0204/val'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0204/val.tfrecord'
    img2tfrecord_(img_folder, 500, 'val', tfrecord_file)

def test_30():
    '''Build a test set that have different bg color for each image
    Zhe Zhu, 2020/02/06'''
    train_sample_num = 20000
    val_sample_num = 500

    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0206/val'

    textures_folder = '/mnt/sdc/ShapeTexture/simulation_data/coco_texture'

    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type_val = 'None'
    noise_param_val = {'s_vs_p': 0.5,
                       'amount': 0.004}

    iscolor = 1

    for i in range(val_sample_num):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        bg_gray_level_bgr = (b,g,r)
        tex_folder = os.path.join(textures_folder,'{:05d}'.format(i+train_sample_num),'texture')
        img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_val,
                                          noise_param=noise_param_val,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level_bgr)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:05d}.png'.format(i)
        mask_file_name = 'val_{:05d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_31():
    '''construct the tfrecord for images with different bg colorr
    Used to test if the model learns the bg
     Zhe Zhu, 2020/02/07'''
    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0206/val'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0206/val.tfrecord'
    img2tfrecord_(img_folder, 500, 'val', tfrecord_file)

def test_32():
    '''Construct a dataset where each image has different bg color
    Zhe Zhu, 2020/02/07'''
    train_sample_num = 20000
    val_sample_num = 500

    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0207/train'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0207/val'

    textures_folder = '/mnt/sdc/ShapeTexture/simulation_data/coco_texture'

    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type_train = 'None'
    noise_param_train = {'mean': 0,
                         'var': 0.2 * 255}
    noise_type_val = 'None'
    noise_param_val = {'s_vs_p': 0.5,
                       'amount': 0.004}

    iscolor = 1

    for i in range(train_sample_num):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        bg_gray_level_bgr = (b, g, r)
        tex_folder = os.path.join(textures_folder, '{:05d}'.format(i), 'texture')
        img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_train,
                                          noise_param=noise_param_train,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level_bgr)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:05d}.png'.format(i)
        mask_file_name = 'train_{:05d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        bg_gray_level_bgr = (b, g, r)
        tex_folder = os.path.join(textures_folder, '{:05d}'.format(i + train_sample_num), 'texture')
        img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_val,
                                          noise_param=noise_param_val,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render(gray_level=bg_gray_level_bgr)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:05d}.png'.format(i)
        mask_file_name = 'val_{:05d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_33():
    '''build the dataset where triangles have coco textures
    and each image has different bg color
    Zhe Zhu, 2020/02/07
    '''
    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0207/train'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0207/train.tfrecord'
    img2tfrecord_(img_folder, 20000, 'train', tfrecord_file)

    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0207/val'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0207/val.tfrecord'
    img2tfrecord_(img_folder, 500, 'val', tfrecord_file)

def test_34():
    '''Construct a dataset where each pixel in the image has different bg color
    Zhe Zhu, 2020/02/08'''
    train_sample_num = 20000
    val_sample_num = 500

    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0208/train'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0208/val'

    textures_folder = '/mnt/sdc/ShapeTexture/simulation_data/coco_texture'

    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type_train = 'None'
    noise_param_train = {'mean': 0,
                         'var': 0.2 * 255}
    noise_type_val = 'None'
    noise_param_val = {'s_vs_p': 0.5,
                       'amount': 0.004}

    iscolor = 1

    for i in range(train_sample_num):
        bg_gray_level_bgr = np.random.rand(img_height,img_width,3)*255
        tex_folder = os.path.join(textures_folder, '{:05d}'.format(i), 'texture')
        img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_train,
                                          noise_param=noise_param_train,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render_randbg(gray_level=bg_gray_level_bgr)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:05d}.png'.format(i)
        mask_file_name = 'train_{:05d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        bg_gray_level_bgr = np.random.rand(img_height,img_width,3)*255
        tex_folder = os.path.join(textures_folder, '{:05d}'.format(i + train_sample_num), 'texture')
        img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_val,
                                          noise_param=noise_param_val,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        img = img_generator.render_randbg(gray_level=bg_gray_level_bgr)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:05d}.png'.format(i)
        mask_file_name = 'val_{:05d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_35():
    '''build the dataset where triangles have coco textures
    and each pixel in each image has different bg color
    Zhe Zhu, 2020/02/08
    '''
    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0208/train'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0208/train.tfrecord'
    img2tfrecord_(img_folder, 20000, 'train', tfrecord_file)

    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0208/val'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0208/val.tfrecord'
    img2tfrecord_(img_folder, 500, 'val', tfrecord_file)

def test_36():
    '''
    Experiment
    train test, triangle positive, rectangle and bg negative, random texture from coco
    Zhe Zhu, 2020/02/19
    '''
    src_img_folder = '/mnt/sdc/dataset/train2014'  # coco dataset
    tex_img_num = 20500
    tgt_tex_folder = '/mnt/sdc/ShapeTexture/simulation_data/coco_texture_0219'
    make_texture_folders_v2(src_img_folder, tex_img_num, tgt_tex_folder)

def test_37():
    '''Construct a dataset where triangle, rectangle and bg are all different coco-texture
    Zhe Zhu, 2020/02/19'''
    train_sample_num = 20000
    val_sample_num = 500

    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0219/train'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0219/val'

    textures_folder = '/mnt/sdc/ShapeTexture/simulation_data/coco_texture_0219'

    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type_train = 'None'
    noise_param_train = {'mean': 0,
                         'var': 0.2 * 255}
    noise_type_val = 'None'
    noise_param_val = {'s_vs_p': 0.5,
                       'amount': 0.004}

    iscolor = 1

    for i in range(train_sample_num):
        tex_folder = os.path.join(textures_folder, '{:05d}'.format(i), 'texture')
        img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_train,
                                          noise_param=noise_param_train,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        bg_img_path = os.path.join(tex_folder,'bg.jpg')
        bg_img = cv2.imread(bg_img_path)
        img = img_generator.render_randbg(gray_level=bg_img)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:05d}.png'.format(i)
        mask_file_name = 'train_{:05d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        bg_gray_level_bgr = np.random.rand(img_height,img_width,3)*255
        tex_folder = os.path.join(textures_folder, '{:05d}'.format(i + train_sample_num), 'texture')
        img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_val,
                                          noise_param=noise_param_val,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        bg_img_path = os.path.join(tex_folder, 'bg.jpg')
        bg_img = cv2.imread(bg_img_path)
        img = img_generator.render_randbg(gray_level=bg_img)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:05d}.png'.format(i)
        mask_file_name = 'val_{:05d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_38():
    '''build the dataset where triangle, rectangle and bg have coco textures
    Zhe Zhu, 2020/02/19
    '''
    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0219/train'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0219/train.tfrecord'
    img2tfrecord_(img_folder, 20000, 'train', tfrecord_file)

    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0219/val'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0219/val.tfrecord'
    img2tfrecord_(img_folder, 500, 'val', tfrecord_file)

def test_39():
    '''
    Check the problem in 0219/val.tfrecord
    Zhe Zhu, 2020/02/21
    '''
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0219/val.tfrecord'
    output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0221_debug'
    tfrecord2imgs(tfrecord_file,output_folder)

def test_40():
    '''
    uniform random texture experiment
    :return:
    '''
    tex_img_num = 20500
    tgt_tex_folder = '/mnt/sdc/ShapeTexture/simulation_data/uniform_texture_0224'
    make_texture_folders_v3(tex_img_num, tgt_tex_folder)

def test_41():
    '''
    Uniform random texture samples
    Zhe Zhu, 2020/02/24
    '''
    train_sample_num = 20000
    val_sample_num = 500

    train_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0224/train'
    val_output_folder = '/mnt/sdc/ShapeTexture/simulation_data/0224/val'

    textures_folder = '/mnt/sdc/ShapeTexture/simulation_data/uniform_texture_0224'

    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type_train = 'None'
    noise_param_train = {'mean': 0,
                         'var': 0.2 * 255}
    noise_type_val = 'None'
    noise_param_val = {'s_vs_p': 0.5,
                       'amount': 0.004}

    iscolor = 1

    for i in range(train_sample_num):
        tex_folder = os.path.join(textures_folder, '{:05d}'.format(i), 'texture')
        img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_train,
                                          noise_param=noise_param_train,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        bg_img_path = os.path.join(tex_folder, 'bg.jpg')
        bg_img = cv2.imread(bg_img_path)
        img = img_generator.render_randbg(gray_level=bg_img)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'train_{:05d}.png'.format(i)
        mask_file_name = 'train_{:05d}_mask.png'.format(i)
        img_path = os.path.join(train_output_folder, img_file_name)
        mask_path = os.path.join(train_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    for i in range(val_sample_num):
        bg_gray_level_bgr = np.random.rand(img_height, img_width, 3) * 255
        tex_folder = os.path.join(textures_folder, '{:05d}'.format(i + train_sample_num), 'texture')
        img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                          texture_img_type=texture_img_type,
                                          img_height=img_height,
                                          img_width=img_width,
                                          layer_num_max=layer_num_max,
                                          scale_min=scale_min,
                                          scale_max=scale_max,
                                          offset_min=offset_min,
                                          offset_max=offset_max,
                                          shape_list=shape_list,
                                          noise_type=noise_type_val,
                                          noise_param=noise_param_val,
                                          iscolor=iscolor,
                                          overlap_rate=overlap_rate)
        img_generator.generate_layout()
        bg_img_path = os.path.join(tex_folder, 'bg.jpg')
        bg_img = cv2.imread(bg_img_path)
        img = img_generator.render_randbg(gray_level=bg_img)
        mask = np.zeros((img_height, img_width))
        for layer in img_generator.layers:
            if layer.pattern.shape_type == 'triangle':
                mask = layer.mask_union(mask, layer.mask)
        mask *= 255.0
        img_file_name = 'val_{:05d}.png'.format(i)
        mask_file_name = 'val_{:05d}_mask.png'.format(i)
        img_path = os.path.join(val_output_folder, img_file_name)
        mask_path = os.path.join(val_output_folder, mask_file_name)
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

def test_42():
    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0224/train'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0224/train.tfrecord'
    img2tfrecord_(img_folder, 20000, 'train', tfrecord_file)

    img_folder = '/mnt/sdc/ShapeTexture/simulation_data/0224/val'
    tfrecord_file = '/mnt/sdc/ShapeTexture/simulation_data/0224/val.tfrecord'
    img2tfrecord_(img_folder, 500, 'val', tfrecord_file)

def test_43():
    '''
    Start from simplest case
    Create three simple domains
    Zhe Zhu 2020/04/17
    '''
    sample_num = 5000
    tex_folder_list = ['/mnt/sdc/ShapeTexture/simulation_data/0417_da/domain_1/texture',
                       '/mnt/sdc/ShapeTexture/simulation_data/0417_da/domain_2/texture',
                       '/mnt/sdc/ShapeTexture/simulation_data/0417_da/domain_3/texture']
    output_folder_list = ['/mnt/sdc/ShapeTexture/simulation_data/0417_da/domain_1/images',
                          '/mnt/sdc/ShapeTexture/simulation_data/0417_da/domain_2/images',
                          '/mnt/sdc/ShapeTexture/simulation_data/0417_da/domain_3/images']

    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type = 'None'
    noise_param = {'mean': 0,
                         'var': 0.2 * 255}
    iscolor = 1

    for i in range(len(tex_folder_list)):
        tex_folder = tex_folder_list[i]
        output_folder = output_folder_list[i]
        for j in range(sample_num):
            img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                              texture_img_type=texture_img_type,
                                              img_height=img_height,
                                              img_width=img_width,
                                              layer_num_max=layer_num_max,
                                              scale_min=scale_min,
                                              scale_max=scale_max,
                                              offset_min=offset_min,
                                              offset_max=offset_max,
                                              shape_list=shape_list,
                                              noise_type=noise_type,
                                              noise_param=noise_param,
                                              iscolor=iscolor,
                                              overlap_rate=overlap_rate)
            img_generator.generate_layout()
            bg_img_path = os.path.join(tex_folder, 'bg.jpg')
            bg_img = cv2.imread(bg_img_path)
            img = img_generator.render_randbg(gray_level=bg_img)
            mask = np.zeros((img_height, img_width))
            for layer in img_generator.layers:
                if layer.pattern.shape_type == 'triangle':
                    mask = layer.mask_union(mask, layer.mask)
            mask *= 255.0
            img_file_name = 'train_{:05d}.png'.format(j)
            mask_file_name = 'train_{:05d}_mask.png'.format(j)
            img_path = os.path.join(output_folder, img_file_name)
            mask_path = os.path.join(output_folder, mask_file_name)
            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)

def test_44():
    '''
    Following maciej's stupid idea, use different bg color
    Zhe Zhu, 2020/04/28
    '''
    sample_num = 5000
    tex_folder_list = ['/mnt/sdc/ShapeTexture/simulation_data/0428_da/domain_1/texture',
                       '/mnt/sdc/ShapeTexture/simulation_data/0428_da/domain_2/texture',
                       '/mnt/sdc/ShapeTexture/simulation_data/0428_da/domain_3/texture']
    output_folder_list = ['/mnt/sdc/ShapeTexture/simulation_data/0428_da/domain_1/images',
                          '/mnt/sdc/ShapeTexture/simulation_data/0428_da/domain_2/images',
                          '/mnt/sdc/ShapeTexture/simulation_data/0428_da/domain_3/images']

    img_height = 512
    img_width = 512
    layer_num_max = 3
    scale_min = 20
    scale_max = 160
    offset_min = 20
    offset_max = 160
    overlap_rate = 0.0
    texture_img_type = 'jpg'
    shape_list = ['rect', 'triangle']

    noise_type = 'None'
    noise_param = {'mean': 0,
                   'var': 0.2 * 255}
    iscolor = 1

    for i in range(len(tex_folder_list)):
        tex_folder = tex_folder_list[i]
        bgtex_idx = random.randint(0,19999)
        bg_tex_folder = os.path.join('/mnt/sdc/ShapeTexture/simulation_data/uniform_texture_0224', '{:05d}'.format(bgtex_idx),'texture')
        output_folder = output_folder_list[i]
        for j in range(sample_num):
            img_generator = ImageGenerator_v5(texture_folder=tex_folder,
                                              texture_img_type=texture_img_type,
                                              img_height=img_height,
                                              img_width=img_width,
                                              layer_num_max=layer_num_max,
                                              scale_min=scale_min,
                                              scale_max=scale_max,
                                              offset_min=offset_min,
                                              offset_max=offset_max,
                                              shape_list=shape_list,
                                              noise_type=noise_type,
                                              noise_param=noise_param,
                                              iscolor=iscolor,
                                              overlap_rate=overlap_rate)
            img_generator.generate_layout()
            bg_img_path = os.path.join(bg_tex_folder, 'bg.jpg')
            bg_img = cv2.imread(bg_img_path)
            img = img_generator.render_randbg(gray_level=bg_img)
            mask = np.zeros((img_height, img_width))
            for layer in img_generator.layers:
                if layer.pattern.shape_type == 'triangle':
                    mask = layer.mask_union(mask, layer.mask)
            mask *= 255.0
            img_file_name = 'train_{:05d}.png'.format(j)
            mask_file_name = 'train_{:05d}_mask.png'.format(j)
            img_path = os.path.join(output_folder, img_file_name)
            mask_path = os.path.join(output_folder, mask_file_name)
            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)

def main():
    #test_1()
    #test_2()
    #test_3()
    #test_4()
    #test_5()
    #test_6()
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
    #test_34()
    #test_35()
    #test_36()
    #test_37()
    #test_38()
    #test_39()
    #test_40()
    #test_41()
    #test_42()
    #test_43()
    test_44()
if __name__ == "__main__":
    main()
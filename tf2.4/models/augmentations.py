from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import scipy.ndimage
import time
import math
import os
import cv2
from skimage.measure import regionprops
from shutil import copyfile
import tensorflow as tf
from tensorflow.python.ops import variables
import tensorflow_addons as tfa


'''
Prostate Cancer Detection in bpMRI
Script:         Model Definition
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         18/05/2021

'''


# Tensor-Based Morphological Augmentations Compatible with TensorFlow Datasets
def augment_tensors(features, targets, prob, translate_factor=0.20, rotation_degree=20, axial_hflip=True, zoom_factor=1.25, gauss_noise_stddev=False, cascaded=False, soft_labels=True):
    
    # Master Probability of Augmentations
    if (np.random.uniform(low=0,high=1)>(1-prob)):     
        
        # Extract Original Images/Labels
        if (cascaded!=False):
            input_image    = tf.identity(features["image"])
            target_label_1 = tf.identity(targets["stage_1_detection"])
            target_label_2 = tf.identity(targets["stage_2_detection"])
        else:  
            input_image    = tf.identity(features["image"])
            target_label_1 = tf.identity(targets["detection"])
    
        # Translation Probability, Offset Values and Augmentation    
        if (translate_factor!=0.00): 
            trans_prob     = tf.constant(np.random.uniform(low=0, high=1))>(1-prob)                                                     # Translation Probability
            pad_top        = tf.constant(np.random.randint(low=0, high=input_image.get_shape()[1]*translate_factor), dtype=tf.int32)  # Translation Offset
            pad_bottom     = tf.constant(np.random.randint(low=0, high=input_image.get_shape()[1]*translate_factor), dtype=tf.int32)  # Translation Offset
            pad_right      = tf.constant(np.random.randint(low=0, high=input_image.get_shape()[2]*translate_factor), dtype=tf.int32)  # Translation Offset
            pad_left       = tf.constant(np.random.randint(low=0, high=input_image.get_shape()[2]*translate_factor), dtype=tf.int32)  # Translation Offset
            input_image    = tf.cond(trans_prob, lambda: translate_4D_tensor(input_image, pad_top=pad_top,     pad_bottom=pad_bottom,                    
                                                                       pad_right=pad_right, pad_left=pad_left), lambda: input_image)  

        # Horizontal Flipping Along Axial Plane Probability and Augmentation
        if (axial_hflip==True):
            flip_prob      = tf.constant(np.random.uniform(low=0, high=1))>(1-prob)                                                  # Horizontal Flipping Probability
            input_image    = tf.cond(flip_prob, lambda: axial_4D_hflip(input_image), lambda: input_image)                           
        
        # Rotation Probability, Offset Value and Augmentation
        if (rotation_degree!=0):
            rot_prob       = tf.constant(np.random.uniform(low=0, high=1))>(1-prob)                                                  # Rotation Probability
            angle          = tf.constant(np.random.uniform(low=-rotation_degree, high=rotation_degree))                              # Rotation Angle
            input_image    = tf.cond(rot_prob, lambda: rotate_4D_tensor(input_image,  angle=angle), lambda: input_image)                
               
        # Scaling Probability and Augmentation
        if (zoom_factor!=0.00):
            zoom_prob      = tf.constant(np.random.uniform(low=0, high=1))>(1-prob)                                                  # Scaling Probability
            scale          = np.random.randint(low=input_image.get_shape()[1], high=input_image.get_shape()[1]*zoom_factor)      # Scaling Factor
            input_image    = tf.cond(zoom_prob, lambda: zoom_4D_tensor(input_image, scale=scale), lambda: input_image)                    
  
        # Additive Gaussian Noise Probability and Augmentation
        if (gauss_noise_stddev!=False):
            gauss_prob     = tf.constant(np.random.uniform(low=0, high=1))>(1-prob)                                                  # Gaussian Noise Probability
            stddev         = tf.constant(np.random.uniform(low=0, high=gauss_noise_stddev))                                          # Standard Deviation of Noise
            input_image    = tf.cond(gauss_prob, lambda: gaussian_noise(input_image, stddev=stddev), lambda: input_image)                    

        # Label Augmentations
        if (translate_factor!=0.00):
            target_label_1   = tf.cond(trans_prob, lambda: translate_4D_tensor(target_label_1, binary=(not soft_labels), 
               pad_top=pad_top, pad_bottom=pad_bottom, pad_right=pad_right, pad_left=pad_left), lambda: target_label_1)
            if (cascaded!=False):
                target_label_2   = tf.cond(trans_prob, lambda: translate_4D_tensor(target_label_2, binary=(not soft_labels), 
                   pad_top=pad_top, pad_bottom=pad_bottom, pad_right=pad_right, pad_left=pad_left), lambda: target_label_2)

        if (axial_hflip==True):      target_label_1 = tf.cond(flip_prob, lambda: axial_4D_hflip(target_label_1,   binary=(not soft_labels)),               lambda: target_label_1)                
        if (rotation_degree!=0):     target_label_1 = tf.cond(rot_prob,  lambda: rotate_4D_tensor(target_label_1, binary=(not soft_labels),  angle=angle), lambda: target_label_1)        
        if (zoom_factor!=0.00):      target_label_1 = tf.cond(zoom_prob, lambda: zoom_4D_tensor(target_label_1,   binary=(not soft_labels),  scale=scale), lambda: target_label_1)
        if (cascaded!=False):         
            if (axial_hflip==True):  target_label_2 = tf.cond(flip_prob, lambda: axial_4D_hflip(target_label_2,   binary=(not soft_labels)),               lambda: target_label_2)                
            if (rotation_degree!=0): target_label_2 = tf.cond(rot_prob,  lambda: rotate_4D_tensor(target_label_2, binary=(not soft_labels),  angle=angle), lambda: target_label_2)        
            if (zoom_factor!=0.00):  target_label_2 = tf.cond(zoom_prob, lambda: zoom_4D_tensor(target_label_2,   binary=(not soft_labels),  scale=scale), lambda: target_label_2)          

        # Export Augmentated Images/Labels
        if (cascaded!=False):
            features["image"]            = tf.identity(input_image)  
            targets["stage_1_detection"] = tf.identity(target_label_1)  
            targets["stage_2_detection"] = tf.identity(target_label_2)  
        else:
            features["image"]            = tf.identity(input_image)  
            targets["detection"]         = tf.identity(target_label_1)  

    return features, targets


# Scaling Augmentation w/ 4D Tensors
def zoom_4D_tensor(input_tensor, binary=False, scale=1.00):
    output_list = []
        
    for i in range(input_tensor.get_shape()[0]):
        scaled_img  = tf.image.resize(input_tensor[i], tf.constant([scale, scale]))
        cropped_img = tf.image.crop_to_bounding_box(scaled_img, 
                                                    scale-input_tensor.get_shape()[1], 
                                                    scale-input_tensor.get_shape()[2], 
                                                    input_tensor.get_shape()[1], 
                                                    input_tensor.get_shape()[2])
        output_list.append(cropped_img)
    output = tf.stack(output_list)
    
    if (binary==True):   return tf.cast(output[:,:,:,0], dtype=tf.int32)
    else:                return tf.cast(output,          dtype=tf.float32)


# Horizontal Flip Augmentation w/ 4D Tensors
def axial_4D_hflip(input_tensor, binary=False):
    output_list = []
        
    for i in range(input_tensor.get_shape()[0]):
        output_list.append(tf.image.flip_left_right(input_tensor[i]))
    output = tf.stack(output_list)
    
    if (binary==True):   return tf.cast(output[:,:,:,0], dtype=tf.int32)
    else:                return tf.cast(output,          dtype=tf.float32)


# Translation Augmentation w/ 4D Tensors
def translate_4D_tensor(input_tensor, binary=False, pad_mode='SYMMETRIC',
                        pad_top=0, pad_bottom=0, pad_right=0, pad_left=0):
        
    # Translation + Padding
    x      = pad_to_bounding_box(input_tensor, pad_top, pad_left, 
                                 input_tensor.get_shape()[1] + pad_bottom + pad_top, 
                                 input_tensor.get_shape()[2] + pad_right  + pad_left,
                                 pad_mode=pad_mode)
    
    # Cropping to Original Shape
    output = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right, 
                                           input_tensor.get_shape()[1], 
                                           input_tensor.get_shape()[2])
    
    if (binary==True):   return tf.cast(output[:,:,:,0], dtype=tf.int32)
    else:                return tf.cast(output,          dtype=tf.float32)


# Rotation Augmentation w/ 4D Tensors
def rotate_4D_tensor(input_tensor, binary=False, pad_mode='SYMMETRIC', angle=0):   
    
    # Translation Offset Values
    diagonal    = ((input_tensor.get_shape()[1])**2+(input_tensor.get_shape()[2])**2)**0.5
    pad         = np.ceil((diagonal-min((input_tensor.get_shape()[1]),(input_tensor.get_shape()[2])))/2).astype(np.int32)
        
    # Translation + Padding
    x           = pad_to_bounding_box(input_tensor, pad, pad, 
                                      input_tensor.get_shape()[1] + (2*pad), 
                                      input_tensor.get_shape()[2] + (2*pad),
                                      pad_mode=pad_mode)
    
    # Rotation + Cropping to Original Shape
    x           = tfa.image.rotate(x,angle*math.pi/180, interpolation='BILINEAR')
    ctr_frac    = input_tensor.get_shape()[1]/x.get_shape()[1]
    output      = tf.image.central_crop(x,ctr_frac)

    if (binary==True):   return tf.cast(output[:,:,:,0], dtype=tf.int32)
    else:                return tf.cast(output,          dtype=tf.float32)


# Additive Gaussian Noise
def gaussian_noise(input_tensor, stddev):
    noise = tf.random.normal(shape=input_tensor.get_shape(), mean=0.0, stddev=stddev, dtype=tf.float32) 
    return tf.cast(input_tensor+noise, dtype=tf.float32)


# Additive Gaussian Noise
def gaussian_noise(input_tensor, stddev):
    noise = tf.random.normal(shape=input_tensor.get_shape(), mean=0.0, stddev=stddev, dtype=tf.float32) 
    return tf.cast(input_tensor+noise, dtype=tf.float32)


# Modified Native 'tf.image.pad_to_bounding_box' Function
def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width, pad_mode='CONSTANT'):

    image = tf.convert_to_tensor(image, name='image')

    is_batch = True
    image_shape = image.get_shape()
    if image_shape.ndims == 3:
      is_batch = False
      image = tf.expand_dims(image, 0)
    elif image_shape.ndims is None:
      is_batch = False
      image = tf.expand_dims(image, 0)
      image.set_shape([None] * 4)
    elif image_shape.ndims != 4:
      raise ValueError('\'image\' must have either 3 or 4 dimensions.')

    batch, height, width, depth = _ImageDimensions(image, rank=4)

    after_padding_width  = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height

    paddings = tf.reshape(
        tf.stack([
            0, 0, offset_height, after_padding_height, offset_width,
            after_padding_width, 0, 0
        ]), [4, 2])
    padded = tf.pad(image, paddings, mode=pad_mode)

    padded_shape = [
        None if isinstance(i,(tf.Tensor, variables.Variable)) else i
        for i in [batch, target_height, target_width, depth]
    ]
    padded.set_shape(padded_shape)

    if not is_batch:
      padded = tf.squeeze(padded, axis=[0])

    return padded


# Extract Image Dimensions [Native to TF]
def _ImageDimensions(image, rank):
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape  = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(image), rank)
    return [ s if s is not None else d for s, d in zip(static_shape, dynamic_shape) ]

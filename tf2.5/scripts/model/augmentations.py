from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import SimpleITK as sitk
import os
import sys
import numpy as np
import pandas as pd
import scipy.ndimage
import time
import math
import os
import cv2
from skimage.measure import regionprops
from skimage.transform import resize
from shutil import copyfile
import tensorflow as tf
from tensorflow.python.ops import variables
import tensorflow_addons as tfa

'''
Prostate Cancer Detection or Zonal Segmentation in MRI
Script:         Train-Time Augmentations
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Task 1: Benign(0), Malignant(1)
                Task 2: Whole-Gland(0), Transitional Zone(1),
                        Peripheral Zone (2)
Update:         07/10/2021

'''


# Tensor-Based Morphological Augmentations
def augment_tensors(features, targets, augmentation_params, train_obj='lesion', debug_on=False):

    # Extract Augmentation Hyperparameters
    prob                = augmentation_params[0]
    tx_prob             = augmentation_params[1] 
    translate_factor    = augmentation_params[2]
    rotation_degree     = augmentation_params[3]
    axial_hflip         = augmentation_params[4]
    zoom_factor         = augmentation_params[5]
    gauss_noise_stddev  = augmentation_params[6]
    chan_shift_factor   = augmentation_params[7]
    sim_poor_scan       = augmentation_params[8]
    gamma_correct       = augmentation_params[9]
    
    # Master Probability of Augmentations
    if tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)>(1-prob):    

        # Extract Original Images/Labels
        input_image_1  = tf.identity(features["image"])
        target_label_1 = tf.identity(targets["detection"])

        # Scaling Probability and Augmentation
        if (zoom_factor!=0.00):
            zoom_prob      = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)>(tx_prob)                                                     
            scale          = tf.random.uniform(shape=[], minval=tf.cast(float(input_image_1.get_shape()[1]), dtype=tf.int32),\
                                                         maxval=tf.cast(tf.math.ceil(input_image_1.get_shape()[1]*zoom_factor), dtype=tf.int32), dtype=tf.int32)          
            input_image_1  = tf.cond(zoom_prob, lambda: zoom_4D_tensor(input_image_1, scale=scale), lambda: input_image_1)                    

        # Horizontal Flipping Along Axial Plane Probability and Augmentation
        if (axial_hflip==True):
            flip_prob      = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)>(0.50)                                                        
            input_image_1  = tf.cond(flip_prob, lambda: axial_4D_hflip(input_image_1), lambda: input_image_1)                           
        
        # Rotation Probability, Offset Value and Augmentation
        if (rotation_degree!=0):
            rot_prob       = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)>(tx_prob)                                                      
            angle          = tf.random.uniform(shape=[], minval=-rotation_degree, maxval=rotation_degree, dtype=tf.float32)                                  
            input_image_1  = tf.cond(rot_prob, lambda: rotate_4D_tensor(input_image_1,  angle=angle), lambda: input_image_1)                               

        # Translation Probability, Offset Values and Augmentation    
        if (translate_factor!=0.00): 
            trans_prob     = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)>(tx_prob)                                                      
            pad_top        = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.math.ceil(input_image_1.get_shape()[1]*translate_factor), dtype=tf.int32), dtype=tf.int32)   
            pad_bottom     = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.math.ceil(input_image_1.get_shape()[1]*translate_factor), dtype=tf.int32), dtype=tf.int32)   
            pad_right      = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.math.ceil(input_image_1.get_shape()[2]*translate_factor), dtype=tf.int32), dtype=tf.int32)   
            pad_left       = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.math.ceil(input_image_1.get_shape()[2]*translate_factor), dtype=tf.int32), dtype=tf.int32)   
            input_image_1  = tf.cond(trans_prob, lambda: translate_4D_tensor(input_image_1, pad_top=pad_top,     pad_bottom=pad_bottom,                    
                                                                       pad_right=pad_right, pad_left=pad_left), lambda: input_image_1)  

        if (train_obj=='lesion'):
            # Simulate Inter-Sequence Registration Error
            if (chan_shift_factor!=0):
                sim_reg_prob   = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)>(tx_prob)
                cs_pad_top     = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.math.ceil(input_image_1.get_shape()[1]*chan_shift_factor), dtype=tf.int32), dtype=tf.int32)  
                cs_pad_bottom  = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.math.ceil(input_image_1.get_shape()[1]*chan_shift_factor), dtype=tf.int32), dtype=tf.int32)  
                cs_pad_right   = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.math.ceil(input_image_1.get_shape()[2]*chan_shift_factor), dtype=tf.int32), dtype=tf.int32)  
                cs_pad_left    = tf.random.uniform(shape=[], minval=0, maxval=tf.cast(tf.math.ceil(input_image_1.get_shape()[2]*chan_shift_factor), dtype=tf.int32), dtype=tf.int32)  
                input_image_1  = tf.cond(sim_reg_prob, lambda: channel_shift_4D_tensor(input_image_1,    pad_top=cs_pad_top,    pad_bottom=cs_pad_bottom,                    
                                                                                 pad_right=cs_pad_right, pad_left=cs_pad_left), lambda: input_image_1)

        # Gamma Correction
        if (np.sum(gamma_correct)!=0):
            gamma_prob     = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)>(tx_prob)                                                     
            gamma          = tf.random.uniform(shape=[], minval=gamma_correct[0], maxval=gamma_correct[1], dtype=tf.float32)                                                       
            input_image_1  = tf.cond(gamma_prob, lambda: gamma_shift_4D_tensor(input_image_1, gamma=gamma, train_obj=train_obj), lambda: input_image_1)

        # Simulate Poor Quality Scan
        if (sim_poor_scan!=False):
            poor_scan_prob = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)>(tx_prob)                                                        
            input_image_1  = tf.cond(poor_scan_prob, lambda: sim_poor_scan_4D_tensor(input_image_1, train_obj=train_obj), lambda: input_image_1)   

        # Additive Gaussian Noise Probability and Augmentation
        if (gauss_noise_stddev!=0):
            gauss_prob     = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)>(tx_prob)                                                      
            stddev         = tf.random.uniform(shape=[], minval=0, maxval=gauss_noise_stddev, dtype=tf.float32)                                              
            input_image_1  = tf.cond(gauss_prob, lambda: gaussian_noise_4D_tensor(input_image_1, stddev=stddev, train_obj=train_obj), lambda: input_image_1)   

        # Label Augmentations
        if (zoom_factor!=0.00):  target_label_1 = tf.cond(zoom_prob, lambda: zoom_4D_tensor(target_label_1,   scale=scale), lambda: target_label_1)
        if (axial_hflip==True):  target_label_1 = tf.cond(flip_prob, lambda: axial_4D_hflip(target_label_1),                lambda: target_label_1)                
        if (rotation_degree!=0): target_label_1 = tf.cond(rot_prob,  lambda: rotate_4D_tensor(target_label_1, angle=angle), lambda: target_label_1)        

        if (translate_factor!=0.00):
            target_label_1   = tf.cond(trans_prob, lambda: translate_4D_tensor(target_label_1, 
               pad_top=pad_top, pad_bottom=pad_bottom, pad_right=pad_right, pad_left=pad_left), lambda: target_label_1)

        # Sanity-Check (Label Swaps From Augmentations)
        if debug_on:
            label_swap_flag = tf.cond(tf.math.ceil(tf.math.reduce_max(targets["detection"]))==tf.math.ceil(tf.math.reduce_max(target_label_1)), lambda: 1, lambda: 0)
            if (label_swap_flag==0):
                tf.print(tf.math.ceil(tf.math.reduce_max(targets["detection"])), output_stream=sys.stdout)
                tf.print(tf.math.ceil(tf.math.reduce_max(target_label_1)),       output_stream=sys.stdout)

        features["image"]    = tf.identity(input_image_1)  
        targets["detection"] = tf.identity(target_label_1)  

    return features, targets





# Scaling Augmentation w/ 4D Tensors
def zoom_4D_tensor(input_tensor, scale=1.00):
    output_list = []
        
    for i in range(input_tensor.get_shape()[0]):
        scaled_img  = tf.image.resize(input_tensor[i], tf.stack([scale, scale]))
        cropped_img = tf.image.crop_to_bounding_box(scaled_img, 
                                                    scale-input_tensor.get_shape()[1], 
                                                    scale-input_tensor.get_shape()[2], 
                                                    input_tensor.get_shape()[1], 
                                                    input_tensor.get_shape()[2])
        output_list.append(cropped_img)
    output = tf.stack(output_list)
    
    return tf.cast(output, dtype=tf.float32)


# Horizontal Flip Augmentation w/ 4D Tensors
def axial_4D_hflip(input_tensor):
    output_list = []
        
    for i in range(input_tensor.get_shape()[0]):
        output_list.append(tf.image.flip_left_right(input_tensor[i]))
    output = tf.stack(output_list)
    
    return tf.cast(output, dtype=tf.float32)


# Translation Augmentation w/ 4D Tensors
def translate_4D_tensor(input_tensor, pad_mode='SYMMETRIC',
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
    
    return tf.cast(output, dtype=tf.float32)


# Simulate Poor Inter-Sequence Registration w/ 4D Tensors (Note: Only Works for 3-Channel Images)
def channel_shift_4D_tensor(input_tensor, pad_mode='SYMMETRIC',
                            pad_top=0, pad_bottom=0, pad_right=0, pad_left=0):
    # Randomly Select One Channel/Sequence
    num_channels   = input_tensor.get_shape()[-1]
    select_channel = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
    shift_channel  = input_tensor[...,select_channel:select_channel+1]

    # Translation + Padding
    x = pad_to_bounding_box(shift_channel, pad_top, pad_left, 
                            shift_channel.get_shape()[1] + pad_bottom + pad_top, 
                            shift_channel.get_shape()[2] + pad_right  + pad_left,
                            pad_mode=pad_mode)
    
    # Cropping to Original Shape
    x = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right, 
                                      shift_channel.get_shape()[1], 
                                      shift_channel.get_shape()[2])

    # Recombine Image
    if (num_channels==3):  # Standard: 3 MRI Sequences
        if   select_channel==0: output = tf.concat([x,input_tensor[...,1:2],input_tensor[...,2:3]], axis=-1)
        elif select_channel==1: output = tf.concat([input_tensor[...,0:1],x,input_tensor[...,2:3]], axis=-1)
        elif select_channel==2: output = tf.concat([input_tensor[...,0:1],input_tensor[...,1:2],x], axis=-1)
        else:                   output = tf.identity(input_tensor)  
    if (num_channels>=4):  # Probabilistic: 3 MRI Sequences + Labels
        if   select_channel==0: output = tf.concat([x,input_tensor[...,1:2],input_tensor[...,2:3],input_tensor[...,3:num_channels]], axis=-1)
        elif select_channel==1: output = tf.concat([input_tensor[...,0:1],x,input_tensor[...,2:3],input_tensor[...,3:num_channels]], axis=-1)
        elif select_channel==2: output = tf.concat([input_tensor[...,0:1],input_tensor[...,1:2],x,input_tensor[...,3:num_channels]], axis=-1)
        else:                   output = tf.identity(input_tensor)  

    return tf.cast(tf.reshape(output, input_tensor.get_shape()), dtype=tf.float32)


# Rotation Augmentation w/ 4D Tensors
def rotate_4D_tensor(input_tensor, pad_mode='SYMMETRIC', angle=0):   
    
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

    return tf.cast(output, dtype=tf.float32)


# Simulate Poor Quality Scan (Note: Only Works for 3-Channel bpMRI or 1-Channel T2W)
def sim_poor_scan_4D_tensor(input_tensor, train_obj='lesion'):    
    # Downsample (via Linear Interp.) and Upsample (via Nearest Neighborhood Interp.)
    num_channels = input_tensor.get_shape()[-1]

    if (train_obj=='lesion'):
        x_0 = sim_poor_scan_3D_tensor(input_tensor[...,0:1])
        x_1 = sim_poor_scan_3D_tensor(input_tensor[...,1:2])
        x_2 = sim_poor_scan_3D_tensor(input_tensor[...,2:3])
    
        # Standard: 3 MRI Sequences
        if (num_channels==3): return tf.cast(tf.reshape(tf.concat([x_0,x_1,x_2], axis=-1),input_tensor.get_shape()), tf.float32)
        # Probabilistic: 3 MRI Sequences + Labels
        if (num_channels>=4): return tf.cast(tf.reshape(tf.concat([x_0,x_1,x_2,input_tensor[...,3:num_channels]], axis=-1),input_tensor.get_shape()), tf.float32)

    elif (train_obj=='zonal'):
        x_0 = sim_poor_scan_3D_tensor(input_tensor[...,0:1])
    
        # Standard: 1 MRI Sequence
        if (num_channels==1): return tf.cast(tf.reshape(x_0,input_tensor.get_shape()), tf.float32)
        # Probabilistic: 1 MRI Sequence + Labels
        if (num_channels>=2): return tf.cast(tf.reshape(tf.concat([x_0,input_tensor[...,1:num_channels]], axis=-1),input_tensor.get_shape()), tf.float32)


# Channel-Wise Simulate Poor Quality Scan
def sim_poor_scan_3D_tensor(x):
    if tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)>(0.50):
        
        x_ = tf.image.resize(x, (int(x.get_shape()[1]*0.75), int(x.get_shape()[1]*0.75)), method='bilinear')
        x_ = tf.image.resize(x_,(int(x.get_shape()[1]),      int(x.get_shape()[1])),      method='nearest')

        return tf.cast(x_, tf.float32)
    else: return tf.cast(tf.identity(x), tf.float32)


# Gamma Boost/Suppress (Note: Only Works for 3-Channel bpMRI or 1-Channel T2W)
def gamma_shift_4D_tensor(input_tensor, gamma=1, train_obj='lesion'):
    num_channels = input_tensor.get_shape()[-1]

    if (train_obj=='lesion'):
        x_0 = gamma_shift_3D_tensor(input_tensor[...,0:1],gamma)
        x_1 = gamma_shift_3D_tensor(input_tensor[...,1:2],gamma)
        x_2 = gamma_shift_3D_tensor(input_tensor[...,2:3],gamma)
    
        # Standard: 3 MRI Sequences
        if (num_channels==3): return tf.cast(tf.reshape(tf.concat([x_0,x_1,x_2], axis=-1),input_tensor.get_shape()), tf.float32)
        # Probabilistic: 3 MRI Sequences + Labels
        if (num_channels>=4): return tf.cast(tf.reshape(tf.concat([x_0,x_1,x_2,input_tensor[...,3:num_channels]], axis=-1),input_tensor.get_shape()), tf.float32)

    elif (train_obj=='zonal'):
        x_0 = gamma_shift_3D_tensor(input_tensor[...,0:1],gamma)
    
        # Standard: 1 MRI Sequence
        if (num_channels==1): return tf.cast(tf.reshape(x_0,input_tensor.get_shape()), tf.float32)
        # Probabilistic: 1 MRI Sequence + Labels
        if (num_channels>=2): return tf.cast(tf.reshape(tf.concat([x_0,input_tensor[...,1:num_channels]], axis=-1),input_tensor.get_shape()), tf.float32)
    

# Channel-Wise Gamma Boost/Suppress
def gamma_shift_3D_tensor(x, gamma):
    if tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)>(0.50):
        mn, sd   = tf.math.reduce_mean(x), tf.math.reduce_std(x)
        rnge     = tf.math.reduce_max(x) - tf.math.reduce_min(x)
        x_       = tf.math.pow(((x - tf.math.reduce_min(x)) /\
                      tf.cast(tf.math.reduce_max(x) - tf.math.reduce_min(x)+1e-8, tf.float32)), gamma)\
                      * (tf.math.reduce_max(x)-tf.math.reduce_min(x)) + tf.math.reduce_min(x)

        x_       = x_ -  tf.math.reduce_mean(x_)
        x_       = x_ / (tf.math.reduce_std(x_) + 1e-8) * sd    # Retain Original Intensity Distribution Shape
        x_0      = x_ + mn
        return tf.cast(x_0, tf.float32)
    else: return tf.cast(tf.identity(x), tf.float32)


# Additive Gaussian Noise (Note: Only Works for 3-Channel bpMRI or 1-Channel T2W)
def gaussian_noise_4D_tensor(input_tensor, stddev=1.0, train_obj='lesion'):

    if (train_obj=='lesion'):
        noise  = tf.random.normal(shape=input_tensor[...,:3].get_shape(), 
                                  mean=0.0, stddev=stddev, dtype=tf.float32)
        output = tf.concat([tf.add(input_tensor[...,:3],noise),input_tensor[...,3:]], axis=-1)

    if (train_obj=='zonal'):
        noise  = tf.random.normal(shape=input_tensor[...,:1].get_shape(), 
                                  mean=0.0, stddev=stddev, dtype=tf.float32)
        output = tf.concat([tf.add(input_tensor[...,:1],noise),input_tensor[...,1:]], axis=-1)

    return tf.cast(tf.reshape(output, input_tensor.get_shape()), dtype=tf.float32)



# Modified Native 'tf.image.pad_to_bounding_box' Function
def pad_to_bounding_box(image, offset_height, offset_width, target_height, 
                        target_width, pad_mode='CONSTANT'):

    image       = tf.convert_to_tensor(image, name='image')
    is_batch    = True
    image_shape = image.get_shape()
    if image_shape.ndims == 3:
      is_batch  = False
      image     = tf.expand_dims(image, 0)
    elif image_shape.ndims is None:
      is_batch  = False
      image     = tf.expand_dims(image, 0)
      image.set_shape([None] * 4)
    elif image_shape.ndims != 4:
      raise ValueError('\'image\' must have either 3 or 4 dimensions.')

    batch, height, width, depth = _ImageDimensions(image, rank=4)

    after_padding_width  = target_width  - offset_width  - width
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

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
import os
import cv2
from skimage.measure import regionprops
from shutil import copyfile
import tensorflow as tf

'''
Prostate Cancer/Zonal Segmentation in bpMRI
Script:         Data Generators
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Task 1: Background(0), TZ(1), PZ(2);
                Task 2: Background(0), csPCa (1);
Update:         10/09/2021

'''


# Custom Model Data Generator for Non-Serial Data
def custom_data_generator(data_xlsx, train_obj='zonal', probabilistic=False, mode='train'):
    """
    Custom Generator that uses Paths to Image + Labels (via XLSX) to 
    Load Data of Size [depth,height,width,#channels] and Sequentially Yield the Same.

    Note: 
    1)  Takes Preprocessed NumPy Data as Input.
    2)  Compatible with tf.data.Dataset Wrappers for Python Generators.
    """
    # Load I/O Datasheet + Initialize Arrays
    all_data   = pd.read_excel(data_xlsx)
    
    i = 0
    while True:
        # Restart Counter
        if ((i+1)>len(all_data['image_path'])):  i = 0 

        # Prepare Model I/O
        while True: # To Counter {BlockingIOError: Resource temporarily unavailable}
            try:
                
                # Anatomical Segmentation (WG,TZ,PZ)
                if (train_obj=='zonal'):
                    image                        = np.load(all_data['image_path'][i])[:,:,:,:1]    
                    if not (mode=='test'): zones = np.load(all_data['zones_path'][i]).astype(np.uint8)
                    else:                  zones = np.zeros_like(image[...,0])

                    tz, pz    = zones.copy() , zones.copy()  
                    tz[zones!=1], pz[zones!=2] = 0,0 
                    tz[zones==1], pz[zones==2] = 1,1                                       # Binarize TZ/PZ Annotations Independently
                    tz, pz = contour_smoothening(tz), contour_smoothening(pz)              # Smoothen Contour Definitions
                    label  = np.stack([np.ones_like(zones)-tz-pz, tz, pz], axis=-1)        # One-Hot Encoding
                
                # Diagnostic Segmentation (csPCa)
                if (train_obj=='lesion'):
                    image                          = np.load(all_data['image_path'][i])    
                    if not (mode=='test'): lesions = np.load(all_data['label_path'][i])
                    else:                  lesions = np.zeros_like(image[...,0])

                    lesions[lesions<=1] = 0                         
                    lesions[lesions>=2] = 1                                                # Binarize Annotation (csPCa: GGG>=2)
                    lesions = contour_smoothening(lesions)                                 # Smoothen Contour Definitions
                    label   = np.stack([np.ones_like(lesions)-lesions, lesions], axis=-1)  # One-Hot Encoding
                break
            except: continue
        i += 1

        if (mode=='test')|(mode=='valid'): postq_lbl = np.zeros_like(label)[:,:,:,1:]
        else:                              postq_lbl = label.copy()[:,:,:,1:] 

        # Bayesian/Probabilistic Segmentation
        if probabilistic:          
            yield {"image":     np.concatenate((image.copy(),postq_lbl), axis=-1)},{
                   "detection": label.copy(),
                   "KL":        np.zeros(shape=label.shape)}
        # Standard/Deterministic Segmentation
        else:
            yield {"image":     image.copy()},{
                   "detection": label.copy()}


# Smoothen Annotation Contours Post-Resampling
def contour_smoothening(label, kernel_2d=(7,7), iterations=1):
    for _ in range(iterations):
        for k in range(label.shape[0]):
            label[k] = cv2.GaussianBlur(label[k].copy().astype(np.uint8), 
                                        kernel_2d, cv2.BORDER_DEFAULT)
    return label
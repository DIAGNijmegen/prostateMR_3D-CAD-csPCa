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
import tensorflow as tf

'''
Prostate Cancer Detection or Zonal Segmentation in MRI
Script:         Misc. Utilities
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Task 1: Benign(0), Malignant(1)
                Task 2: Whole-Gland(0), Transitional Zone(1),
                        Peripheral Zone (2)
Update:         03/10/2021

'''


# TensorFlow Device Configuration 
def setup_device(gpuid=None):
    devices = []

    if gpuid is not None and not isinstance(gpuid, str): gpuid = str(gpuid)
    if gpuid is not None: nb_devices = len(gpuid.split(','))
    else:                 nb_devices = 1

    if gpuid is not None and (gpuid!='-1'):
        # Define GPU Devices for Mirrored Strategy
        if (nb_devices>1):
            for i in range(nb_devices): devices.append('/gpu:'+str(i))
        else:                           devices = '/gpu:0'
        
        # Set Number of Visible GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

        # GPU Memory Configuration Differs Between TF 1 and 2
        if hasattr(tf, 'ConfigProto'):
            config                          = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement     = True
            config.log_device_placement     = True
            tf.keras.backend.set_session(tf.Session(config=config))
        else:
            tf.config.set_soft_device_placement(True)
            for pd in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(pd, True)
    else:
        devices = '/cpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    return devices, nb_devices

# Print Overview Of Run using CL Arguments
def print_overview(args):
    # Training Objective
    if   (args.TRAIN_OBJ=='zonal'):  print("Training Objective: Anatomical (WG,TZ,PZ) Segmentation")
    elif (args.TRAIN_OBJ=='lesion'): print("Training Objective: Diagnostic (csPCa) Segmentation")
    
    # Probabilistic Outputs 
    if args.UNET_PROBABILISTIC:     print("Probabilistic Outputs: Enabled")
    else:                           print("Probabilistic Outputs: Disabled")

    # Deep Supervision
    if args.UNET_DEEP_SUPERVISION:  print("Deep Supervision: Enabled")
    else:                           print("Deep Supervision: Disabled")

    # Number of Classes at Train-Time
    if   (args.TRAIN_OBJ=='zonal'):    print("Number of Classes at Train-Time: 3")       
    elif (args.TRAIN_OBJ=='lesion'):   print("Number of Classes at Train-Time: 2")      
    
    # Training Hyperparameters
    print("Batch Size:",            args.BATCH_SIZE)
    print("Initial Learning Rate:", args.BASE_LR)
    
    # Optimizer
    if   (args.OPTIMIZER=='adam'):     print("Optimizer: Adam w/ AMSGrad")
    elif (args.OPTIMIZER=='momentum'): print("Optimizer: Stochastic Gradient Descent w/ Nesterov Momentum")
    
    # Loss Function
    if   (args.LOSS_MODE=='distribution_focal'):  print("Loss Function: Distribution/Focal Loss (alpha="+str(args.FOCAL_LOSS_ALPHA)+"; gamma="+str(args.FOCAL_LOSS_GAMMA)+")")
    elif (args.LOSS_MODE=='region_boundary'):     print("Loss Function: Soft Dice + Boundary/Surface Loss")
    
    # Train-Time Augmentations
    if (args.AUGM_PARAMS[0]>0): 
        print("Train-Time Augmentations: \
          (Master Probability: "+        str(args.AUGM_PARAMS[0])+\
        "; Transformation Probability: "+str(args.AUGM_PARAMS[1])+\
        "; X-Y Translation: "+           str(args.AUGM_PARAMS[2])+\
        "; Rotation Degree: ±"+          str(args.AUGM_PARAMS[3])+\
        "; Horizontal Flip: "+           str(args.AUGM_PARAMS[4])+\
        "; Zoom Factor: "+               str(args.AUGM_PARAMS[5])+\
        "; Gaussian Noise: stddev<="+    str(args.AUGM_PARAMS[6])+\
        "; Channel Shift: "+             str(args.AUGM_PARAMS[7])+\
        "; Simulate Poor Scan Quality: "+str(args.AUGM_PARAMS[8])+\
        "; Gamma Shifts: "+              str(args.AUGM_PARAMS[9])+")")
    else:                       print("Train-Time Augmentations: Disabled")
    
    # Dropout Mode
    if   (args.UNET_DROPOUT_MODE=='standard'):     print("Dropout: Standard (rate={:.2f})".format(args.UNET_DROPOUT_RATE))
    elif (args.UNET_DROPOUT_MODE=='monte-carlo'):  print("Dropout: Monte Carlo (rate={:.2f})".format(args.UNET_DROPOUT_RATE))
    elif (args.UNET_DROPOUT_RATE==0):              print("Dropout: Disabled")

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
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from shutil import copyfile
import tensorflow as tf
import model.unets as unets
import functools
print = functools.partial(print, flush=True)
from deploy_FROC import compute_FROC

'''
Prostate Cancer Detection or Zonal Segmentation in MRI
Script:         Train-Time Callbacks
Contributor:    anindox8, joeranbosma
Target Organ:   Prostate
Target Classes: Task 1: Benign(0), Malignant(1)
                Task 2: Whole-Gland(0), Transitional Zone(1),
                        Peripheral Zone (2)
Update:         03/10/2021

'''


# Dice Coefficient for 3D Volumes
def dice_3d(predictions, labels):
    epsilon     =  1e-7
    dice_num    =  np.sum(predictions[labels==1])*2.0 
    dice_denom  =  np.sum(predictions) + np.sum(labels)
    return ((dice_num+epsilon)/(dice_denom+epsilon)).astype(np.float32)


# Export Weights Every N Epochs
class WeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self, model, min_epoch, weights_num_epochs, weights_dir, init_epoch=0, weights_overwrite=True):
        self.model   = model
        self.N       = weights_num_epochs
        self.M       = min_epoch
        self.D       = weights_dir+'/model_weights.h5'
        self.O       = weights_overwrite
        self.epoch   = init_epoch

    def on_epoch_end(self, epoch, logs={}):

        if ((self.epoch+1)%self.N==0)&(self.epoch!=0)&((self.epoch+1)>=self.M):
            name     =  self.D
            name     =  name.split('.h5')[0] + '_%03d.h5' % (self.epoch+1)
            
            # To Counter {BlockingIOError: Resource temporarily unavailable}
            while True:
                try:
                    tf.keras.models.save_model(self.model, name)
                    print('Model Weights Saved: ', name)
                    break
                except: continue
            if self.O:
                name =  self.D
                name =  name.split('.h5')[0] + '_%03d.h5' % ((self.epoch+1)-self.N)
                
                while True:
                    try:
                        if os.path.exists(name): os.remove(name)
                        break
                    except: continue
        self.epoch  += 1


# Custom Learning Rate Scheduler
class ReduceLR_Schedule(tf.keras.callbacks.Callback):
    """
    Reduce learning rate when model performance has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates.
    """
    def __init__(self, lr_rates, epoch_points):
        self.lr_rates         = lr_rates
        self.epoch_points     = epoch_points
          
    def on_epoch_begin(self, epoch, logs=None):

        assert (len(self.epoch_points)==len(self.lr_rates))

        if ((epoch+1)>=self.epoch_points[0])&((epoch+1)<self.epoch_points[1]): new_lr = self.lr_rates[0]
        if ((epoch+1)>=self.epoch_points[1])&((epoch+1)<self.epoch_points[2]): new_lr = self.lr_rates[1]
        if ((epoch+1)>=self.epoch_points[2])&((epoch+1)<self.epoch_points[3]): new_lr = self.lr_rates[2]
        if ((epoch+1)>=self.epoch_points[3]):                                  new_lr = self.lr_rates[3]
        
        if ((epoch+1)==self.epoch_points[0])|((epoch+1)==self.epoch_points[1])|((epoch+1)==self.epoch_points[2])|((epoch+1)==self.epoch_points[3]):
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print('\nEpoch %03d: ReduceLR_Schedule reducing learning '
                  'rate to %s.' % (epoch+1, new_lr))


# Custom Learning Rate Scheduler
class PolyLR_Schedule(tf.keras.callbacks.Callback):
    """
    Reduce learning rate as per the nn-U-Net training heuristic.
    """
    def __init__(self, initial_lr, exponent, max_epochs):
        self.initial_lr  = initial_lr
        self.exponent    = exponent
        self.max_epochs  = max_epochs
          
    def on_epoch_begin(self, epoch, logs=None):
        new_lr           = self.initial_lr * (1-epoch/self.max_epochs)**self.exponent

        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        print('\nEpoch %03d: PolyLR_Schedule reducing learning '
              'rate to %s.' % (epoch+1, new_lr))


# Cyclic Learning Rate Scheduler
class CyclicLR(tf.keras.callbacks.Callback):
    """
    Instead of monotonically decreasing the learning rate, this method 
    lets the learning rate cyclically vary between reasonable boundary
    values. Training with cyclical learning rates instead of fixed values
    achieves improved classification accuracy without a need to tune and
    often in fewer iterations.

    [1] L.N. Smith (2017), "Cyclical Learning Rates for Training Neural Networks", IEEE WACV
    
    """
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        self._reset()
    
    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}
        if self.clr_iterations == 0:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())


# Load Model Weights and Restart/Resume Training
def ResumeTraining(model, weights_dir, resume=True, prefix='model_weights'):
    weights_dir = '/'+prefix+'.h5'
    init_epoch  = 0    

    for f in os.listdir(weights_dir.split(prefix)[0]):
        if (resume) & (weights_dir.split('.h5')[0] in (weights_dir.split(prefix)[0]+f)) & ('.xlsx' not in (weights_dir.split(prefix)[0]+f)):
            temp_epoch = int(((weights_dir.split(prefix)[0]+f).split(weights_dir.split('.h5')[0]+'_')[1]).split('.h5')[0])
            if (temp_epoch > init_epoch): init_epoch = temp_epoch

    for f in os.listdir(weights_dir.split(prefix)[0]):
        if (resume) & (weights_dir.split('.h5')[0] in (weights_dir.split(prefix)[0]+f)) & ('.xlsx' not in (weights_dir.split(prefix)[0]+f)):
            if (init_epoch == int(((weights_dir.split(prefix)[0]+f).split(weights_dir.split('.h5')[0]+'_')[1]).split('.h5')[0])):
   
                print('Loading Model Weights...')
                model = unets.networks.M1.load(path=weights_dir.split(prefix)[0]+f)
                print('Complete: ', weights_dir.split(prefix)[0]+f)
   
    if (init_epoch==0): print('Begin Training @ Epoch ',  init_epoch)
    else:               print('Resume Training @ Epoch ', init_epoch)

    return model, init_epoch





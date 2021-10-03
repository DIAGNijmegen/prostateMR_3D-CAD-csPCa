import sys
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
import tensorflow as tf
import tensorflow.keras.layers as KL


'''
Prostate Cancer/Zonal Segmentation in bpMRI
Script:         Train Model for Zonal Segmentation
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Task 1: Background(0), TZ(1), PZ(2);
                Task 2: Background(0), csPCa (1);
Update:         10/09/2021

'''


class Focal:
    """
    [1] T.Y. Lin et al. (2017), "Focal Loss for Dense Object Detection", IEEE CVPR. 
    
    Note: Function is Applicable for Binary/One-Hot Objectives Only.
    """
    def __init__(self, alpha=[0.25, 0.75], gamma=2.00):
        self.alpha    = alpha
        self.gamma    = gamma

    def loss(self, y_true, y_pred):
        class_weights = tf.convert_to_tensor(self.alpha, dtype=tf.float32)        
        y_pred       /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)                        
        y_pred        = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon()) 
        ce            = tf.math.multiply(tf.cast(y_true, tf.float32), -tf.math.log(y_pred))
        gamma_weight  = tf.math.multiply(tf.cast(y_true, tf.float32),  tf.math.pow(tf.math.subtract(1.0, y_pred), self.gamma))
        fl            = tf.math.multiply(class_weights, tf.math.multiply(gamma_weight, ce))
        return tf.reduce_mean(fl)


class WeightedCategoricalCE:
    """
    Weighted Adaptation of tf.keras.losses.CategoricalCrossentropy {alpha: np.array([C1,C2,C3,...])}
    
    Note: Function is Applicable for Binary/One-Hot Objectives Only.
    """
    def __init__(self, alpha=[0.25, 0.75]):
        self.alpha    = alpha
            
    def loss(self, y_true, y_pred):
        class_weights = tf.convert_to_tensor(self.alpha, dtype=tf.float32)
        y_pred       /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)                        
        y_pred        = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon()) 
        ce            = tf.math.multiply(tf.cast(y_true, tf.float32), -tf.math.log(y_pred))
        wce           = tf.math.multiply(ce, class_weights)
        return tf.reduce_mean(wce)


class EvidenceLowerBound:
    """
    Dummy Wrapper for Averaging Sample-Wise KL/ELBO Losses into Batch-Wise Metric. Actual Loss Value Computed 
    Inside Model (Refer to Tensor 'kl' in Function 'm1' of 'model.unets.networks') and Passed via 'y_pred'.

    [1] S. Kohl et al. (2018), "A Probabilistic U-Net for Segmentation of Ambiguous Images", NeurIPS.
    
    Note: Function is Applicable for Binary/One-Hot Objectives Only.
    """
    def __init__(self, beta=1.00):
        self.beta     = beta

    def loss(self, y_true, y_pred):
        return (self.beta * tf.reduce_mean(y_pred))


class SoftDicePlusBoundarySurface:
    """
    Soft Dice Loss + Boundary/Surface Loss for Multi-Class Segmentation 
    
    [1] H. Kervadec et al. (2021), "Boundary Loss for Highly Unbalanced Segmentation", MedIA.
    [Ref: https://github.com/keras-team/keras/issues/9395#issuecomment-370971561]
    [Ref: https://github.com/LIVIAETS/boundary-loss/]
    
    Note: Function is Applicable for Binary/One-Hot Objectives Only.
    """
    def __init__(self, loss_weights=[1.00, 1.50], smooth=tf.keras.backend.epsilon()):
        self.smooth       = smooth
        self.loss_weights = loss_weights

    def calc_dist_map(self, seg):
        num_classes = int(seg.shape[-1])
        res         = np.zeros_like(seg)
        
        for c in range(num_classes):
            posmask        = seg[...,c].astype(np.bool)
            if posmask.any():
                negmask    = ~posmask
                res[...,c] = distance(negmask)    * negmask \
                           -(distance(posmask)-1) * posmask
        return res
    
    def calc_dist_map_batch(self, y_true):
        y_true_numpy = y_true.numpy()
        return np.array([self.calc_dist_map(y) for y in y_true_numpy]).astype(np.float32)

    def dice_loss(self, y_true, y_pred):
        y_pred    /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)                        
        y_pred     = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon()) 
        y_true_f   = tf.keras.backend.flatten(y_true[...,1:])
        y_pred_f   = tf.keras.backend.flatten(y_pred[...,1:])
        intersect  = tf.reduce_sum(y_true_f * y_pred_f, axis=-1)
        denom      = tf.reduce_sum(y_true_f + y_pred_f, axis=-1)
        return 1-tf.reduce_mean((2. * intersect / (denom + self.smooth)))

    def boundary_surface_loss(self, y_true, y_pred):
        y_pred         /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)                        
        y_pred          = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon()) 
        y_true_dist_map = tf.py_function(func=self.calc_dist_map_batch, inp=[y_true[...,1:]], Tout=tf.float32)
        return tf.reduce_mean((y_pred[...,1:]*y_true_dist_map))

    def loss(self, y_true, y_pred):
        return (self.loss_weights[0] * self.dice_loss(y_true, y_pred)) + \
               (self.loss_weights[1] * self.boundary_surface_loss(y_true, y_pred))


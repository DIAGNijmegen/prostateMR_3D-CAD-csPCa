import sys
import numpy as np
import tensorflow as tf

'''
Prostate Cancer Detection in bpMRI
Script:         TensorFlow Losses
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         08/04/2021

'''



class Focal:
    """
    Focal Loss for Segmentation with Severe Class Imbalance
    [1] T.Y. Lin et al. (2017), "Focal Loss for Dense Object Detection", IEEE CVPR. 
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha         = alpha
        self.gamma         = gamma

    def loss(self, y_true, y_pred):
        ce                 = tf.math.multiply(tf.cast(y_true, tf.float32), -tf.math.log(y_pred))
        weight             = tf.math.multiply(tf.cast(y_true, tf.float32),  tf.math.pow(tf.math.subtract(1.0, y_pred), self.gamma))
        fl                 = tf.reduce_max(tf.math.multiply(self.alpha, tf.math.multiply(weight, ce)), axis=1)
        return tf.reduce_mean(fl)


class Weighted_Categorical_CE:
    """
    Weighted Adaptation of tf.keras.losses.CategoricalCrossentropy {weights: np.array([C1,C2,C3,...])}
    """
    def __init__(self, class_weights):
        self.weights      =  class_weights
            
    def loss(self, y_true, y_pred):
        weights       =  tf.convert_to_tensor(self.weights, dtype=tf.float32)
        y_pred       /=  tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)                        
        y_pred        =  tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon()) 
        loss          =  tf.math.multiply(tf.cast(y_true, tf.float32), tf.math.log(y_pred))
        weighted      =  tf.math.multiply(loss, weights)
        loss          = -tf.keras.backend.sum(loss, -1)
        return loss
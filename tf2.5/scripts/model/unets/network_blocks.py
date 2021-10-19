import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from .modelio import LoadableModel, store_config_args
import sonnet as snt

'''
Prostate Cancer Detection or Zonal Segmentation in MRI
Script:         Network Architecture
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Task 1: Benign(0), Malignant(1)
                Task 2: Whole-Gland(0), Transitional Zone(1),
                        Peripheral Zone (2)
Update:         03/10/2021

'''


# Feature Extraction Blocks
# 3D SEResNet BottleNeck Module -----------------------------------------------------------------------------------------------------------------------------------------
class SEResNetBottleNeck(tf.keras.layers.Layer):
    """
    [1] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    """
    def __init__(self, filters, kernel_size, strides, conv_params, reduction):
        
        super(SEResNetBottleNeck, self).__init__()

        self.filters     = filters
        self.kernel_size = kernel_size
        self.strides     = strides
        self.conv_params = conv_params
        self.reduction   = reduction

        self.conv1 = tf.keras.layers.Conv3D(filters=self.filters//4, kernel_size=self.kernel_size, strides=self.strides, **self.conv_params)
        self.norm1 = tfa.layers.InstanceNormalization()
        self.conv2 = tf.keras.layers.Conv3D(filters=self.filters//4, kernel_size=(3,3,3), strides=(1,1,1), **self.conv_params)
        self.norm2 = tfa.layers.InstanceNormalization()
        self.conv3 = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=(1,1,1), strides=(1,1,1), **self.conv_params)
        self.norm3 = tfa.layers.InstanceNormalization()    
        self.conv4 = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, **self.conv_params)
        self.norm4 = tfa.layers.InstanceNormalization()
        self.conv6 = tf.keras.layers.Conv3D(filters=self.filters//self.reduction, kernel_size=(1,1,1), strides=(1,1,1))
        self.conv7 = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=(1,1,1), strides=(1,1,1))

    def call(self, input_tensor):
        x        = tf.identity(input_tensor)
        residual = tf.identity(input_tensor)

        # Bottleneck
        x  = self.conv1(x)
        x  = self.norm1(x)
        x  = tf.keras.activations.relu(x, alpha=0.1)
        x  = self.conv2(x)
        x  = self.norm2(x)
        x  = tf.keras.activations.relu(x, alpha=0.1)
        x  = self.conv3(x)
        x_ = self.norm3(x)

        # Replicate Operations with Residual Connection (change in #num_filters or spatial_dims)
        if (x_.get_shape()[-1]!=residual.get_shape()[-1]):
            residual = self.conv4(residual)
            residual = self.norm4(residual)

        # Attention Module
        x = tf.keras.layers.GlobalAveragePooling3D()(x_)
        x = tf.keras.layers.Lambda(function=ExpandDims)(x)
        x = self.conv6(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = self.conv7(x)
        x = tf.keras.activations.sigmoid(x)
        x = tf.math.multiply(x_,x)

        # Residual Addition
        x = tf.math.multiply(x,residual) 
        x = tf.keras.activations.relu(x, alpha=0.1)

        return x

def ExpandDims(x):
    return x[:,None,None,None,:]
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


# 3D Attention Gating Module -----------------------------------------------------------------------------------------------------------------------------------------
class GridAttentionBlock3D(tf.keras.layers.Layer):
    '''
    [1] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    '''
    def __init__(self, inter_channels, sub_samp, conv_params):

        super(GridAttentionBlock3D, self).__init__()

        self.conv_params    = conv_params
        self.inter_channels = inter_channels
        self.sub_samp       = sub_samp

        self.conv1 = tf.keras.layers.Conv3D(filters=self.inter_channels, kernel_size=self.sub_samp, strides=self.sub_samp, **self.conv_params)
        self.conv2 = tf.keras.layers.Conv3D(filters=self.inter_channels, kernel_size=(1,1,1),  strides=(1,1,1),  **self.conv_params)
        self.conv3 = tf.keras.layers.Conv3D(filters=1, kernel_size=(1,1,1), strides=(1,1,1), **self.conv_params)
        self.conv4 = tf.keras.layers.Conv3D(filters=self.inter_channels, kernel_size=(1,1,1), strides=(1,1,1), **self.conv_params)
        self.norm4 = tfa.layers.InstanceNormalization()
    
    def call(self, conv_tensor, gating_tensor):
        x = conv_tensor
        g = gating_tensor
        
        # Attention Gating Function (theta^T * x_ij + phi^T * gating_signal + bias)
        theta_x     = self.conv1(x)
        phi_g       = self.conv2(g)
        scale_z     = theta_x.get_shape()[1] // phi_g.get_shape()[1]
        scale_x     = theta_x.get_shape()[2] // phi_g.get_shape()[2]
        scale_y     = theta_x.get_shape()[3] // phi_g.get_shape()[3]
        phi_g       = tf.keras.layers.UpSampling3D(size=(scale_z,scale_x,scale_y))(phi_g)
        f           = tf.keras.layers.LeakyReLU(alpha=0.1)(theta_x+phi_g)
        psi_f       = self.conv3(f)
        sigm_psi_f  = tf.keras.layers.Activation('sigmoid')(psi_f)
        scale_z     = x.get_shape()[1] // sigm_psi_f.get_shape()[1]
        scale_x     = x.get_shape()[2] // sigm_psi_f.get_shape()[2]
        scale_y     = x.get_shape()[3] // sigm_psi_f.get_shape()[3]
        sigm_psi_f  = tf.keras.layers.UpSampling3D(size=(scale_z,scale_x,scale_y))(sigm_psi_f)
        y           = sigm_psi_f * x
    
        # Output Projection
        W_y         = self.conv4(y)
        W_y         = self.norm4(W_y)
    
        return W_y, sigm_psi_f

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Probabilistic Blocks
# Monte Carlo Dropout ---------------------------------------------------------------------------------------------------------------------------------------------------
class MonteCarloDropout(tf.keras.layers.Layer):
    def __init__(self, rate):
        super(MonteCarloDropout, self).__init__()
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, rate=self.rate)


# Convolutional Encoder to Parametrize Gaussian Distribution with Axis-Aligned Covariance Matrix
def AxisAligned3DConvGaussian(filters            =  (32,64,128,256,512), 
                              strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2)),           
                              kernel_sizes       = ((1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3)),
                              se_reduction       =  (8,8,8,8,8),
                              proba_event_shape  =   256,
                              kernel_initializer =   tf.keras.initializers.Orthogonal(gain=1.0),
                              bias_initializer   =   tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
                              kernel_regularizer =   tf.keras.regularizers.l2(1e-4),
                              bias_regularizer   =   tf.keras.regularizers.l2(1e-4),
                              latent_tensor_name =  'latent_distribution'):
    """
    [1] S. Kohl et al. (2018), "A Probabilistic U-Net for Segmentation of Ambiguous Images", NeurIPS.
    """
    def layer(image, segmentation):
        # Preamble
        if (segmentation!=None): x = tf.concat([image, tf.cast(segmentation,tf.float32)], axis=-1)
        else:                    x = image
        
        conv_params = {'padding':           'same',
                       'kernel_initializer': kernel_initializer,
                       'bias_initializer':   bias_initializer,
                       'kernel_regularizer': kernel_regularizer,
                       'bias_regularizer':   bias_regularizer}
        
        # Preliminary Convolutional Layer
        x     = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=kernel_sizes[0], strides=strides[0], **conv_params)(x)
        x     = tfa.layers.InstanceNormalization()(x)
        x     = tf.keras.layers.LeakyReLU(alpha=0.1)(x)  
        
        # Encoder: Backbone SE-Residual Blocks for Feature Extraction
        conv1 = SEResNetBottleNeck(filters=filters[1], kernel_size=kernel_sizes[1], 
                                   strides=strides[1], reduction  =se_reduction[1], conv_params=conv_params)(x)  
        conv2 = SEResNetBottleNeck(filters=filters[2], kernel_size=kernel_sizes[2],
                                   strides=strides[2], reduction  =se_reduction[2], conv_params=conv_params)(conv1)  
        conv3 = SEResNetBottleNeck(filters=filters[3], kernel_size=kernel_sizes[3],
                                   strides=strides[3], reduction  =se_reduction[3], conv_params=conv_params)(conv2)  
        convm = SEResNetBottleNeck(filters=filters[4], kernel_size=kernel_sizes[4],
                                   strides=strides[4], reduction  =se_reduction[4], conv_params=conv_params)(conv3)

        # Latent Prior Distribution
        encoding     = tf.reduce_mean(convm, axis=[1,2,3], keepdims=True)
        mu_log_sigma = tf.keras.layers.Conv3D(filters=proba_event_shape*2, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(encoding)
        mu_log_sigma = tf.squeeze(mu_log_sigma, axis=[1,2,3])
        return mu_log_sigma
    return layer


# Integrate Latent Distribution Features and Derive Logits
class Conv1x1x1withLatentDist(snt.Module):
    """
    [1] S. Kohl et al. (2018), "A Probabilistic U-Net for Segmentation of Ambiguous Images", NeurIPS.
    """        
    def __init__(self,
                 num_classes        =  2, 
                 num_channels       =  256,
                 kernel_initializer =  tf.keras.initializers.Orthogonal(gain=1.0),
                 bias_initializer   =  tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
                 kernel_regularizer =  tf.keras.regularizers.l2(1e-4),
                 bias_regularizer   =  tf.keras.regularizers.l2(1e-4),
                 logits_tensor_name = 'latent_logits'):
        
        super(Conv1x1x1withLatentDist, self).__init__()

        self.num_classes        = num_classes       
        self.num_channels       = num_channels      
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer  
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer   = bias_regularizer  
        self.logits_tensor_name = logits_tensor_name

        conv_params = {'padding':           'same',
                       'kernel_initializer': self.kernel_initializer,
                       'bias_initializer':   self.bias_initializer,
                       'kernel_regularizer': self.kernel_regularizer,
                       'bias_regularizer':   self.bias_regularizer}

        # Final Convolutional Layers    
        self.conv1  = tf.keras.layers.Conv3D(filters=self.num_channels//4,  kernel_size=(1,3,3), strides=(1,1,1), **conv_params)
        self.norm1  = tfa.layers.InstanceNormalization()
        self.conv2  = tf.keras.layers.Conv3D(filters=self.num_channels//16, kernel_size=(1,3,3), strides=(1,1,1), **conv_params)
        self.norm2  = tfa.layers.InstanceNormalization()
        self.logits = tf.keras.layers.Conv3D(filters=self.num_classes,      kernel_size=(1,1,1), strides=(1,1,1), **conv_params, 
                                             name=self.logits_tensor_name)

    def __call__(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.norm1(x, training=training)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = tf.keras.activations.relu(x, alpha=0.1)

        return self.logits(x)


# Integrate Hierarchical Latent Distribution Features and Derive Logits
class StitchingProbDecoder(snt.Module):
    """
    [1] S. Kohl et al. (2019), "A Hierarchical Probabilistic U-Net for Modeling Multi-Scale Ambiguities", NeurIPS.
    """
    def __init__(self,
                 num_classes        =   2, 
                 filters            =  (32,64,128,256,512), 
                 strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2)),
                 kernel_sizes       = ((1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3)),
                 kernel_initializer =  tf.keras.initializers.Orthogonal(gain=1.0),
                 bias_initializer   =  tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
                 kernel_regularizer =  tf.keras.regularizers.l2(1e-4),
                 bias_regularizer   =  tf.keras.regularizers.l2(1e-4)):

        super(StitchingProbDecoder, self).__init__()

        self.num_classes        = num_classes       
        self.filters            = filters     
        self.strides            = strides     
        self.kernel_sizes       = kernel_sizes
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer  
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer   = bias_regularizer  

        conv_params = {'padding':           'same',
                       'kernel_initializer': self.kernel_initializer,
                       'bias_initializer':   self.bias_initializer,
                       'kernel_regularizer': self.kernel_regularizer,
                       'bias_regularizer':   self.bias_regularizer}

        self.logits  = tf.keras.layers.Conv3D(filters=self.num_classes, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)

    def __call__(self, decoder_features):
        return self.logits(decoder_features)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------





























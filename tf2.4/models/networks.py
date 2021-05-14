import numpy as np
import tensorflow as tf
from .modelio import LoadableModel, store_config_args

'''
Prostate Cancer Detection in bpMRI
Script:         Model Definition
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         08/04/2021

'''



# Dual-Attention U-Net for PCa Detection ------------------------------------------------------------------------------------------------------------------------------
class M1(LoadableModel):
    '''
    [1] Z. Zhou et al. (2019), "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", IEEE TMI.
    [2] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    [3] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    [4] K. He et al.(2016), "Deep Residual Learning for Image Recognition", IEEE CVPR.
    [5] K. He et al.(2016), "Identity Mappings in Deep Residual Networks", ECCV.
    '''
    @store_config_args
    def __init__(self,
                 input_shape,
                 src_feats,
                 num_classes,
                 dropout_rate       =   0.10,  
                 dropout_mode       =  'standard',    
                 filters            =  (16,32,64,128,256),           
                 strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,1,1)),           
                 se_reduction       =  (8,8,8,8,8),         
                 att_sub_samp       = ((1,1,1),(1,1,1),(1,1,1)),      
                 bias_initializer   =   tf.zeros_initializer(),
                 bias_regularizer   =   tf.keras.regularizers.l2(1e-3),         
                 kernel_initializer =   tf.initializers.GlorotUniform(),
                 kernel_regularizer =   tf.keras.regularizers.l2(1e-3),
                 label_encoding     =  'one_hot',
                 cascaded           =   False,
                 anatomical_prior   =   False):

        # Ensure Correct Dimensionality
        ndims = len(input_shape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    
        # Input Layer Definition
        source = tf.keras.Input(shape=(*input_shape, src_feats+1 if anatomical_prior else src_feats), name='input_image')

        # Standard Single-Stage Model
        if not cascaded:            
            # Dual-Attention U-Net Model Definition
            m1_model    = m1(inputs             =  source, 
                             num_classes        =  num_classes, 
                             dropout_mode       =  dropout_mode,
                             dropout_rate       =  dropout_rate,          
                             filters            =  filters,            
                             strides            =  strides,            
                             se_reduction       =  se_reduction,          
                             att_sub_samp       =  att_sub_samp,             
                             kernel_initializer =  kernel_initializer,    
                             bias_initializer   =  bias_initializer,     
                             kernel_regularizer =  kernel_regularizer,    
                             bias_regularizer   =  bias_regularizer)
    
            super().__init__(name='cad', inputs=[source], outputs=[m1_model['logits']])
    
            # Cache Pointers to Layers/Tensors for Future Reference
            self.references           = LoadableModel.ReferenceContainer()
            self.references.m1_model  = m1_model
            self.encoding             = label_encoding
            self.cascaded             = cascaded

        # Cascaded Two-Stage Model
        else:            
            # First-Stage Dual-Attention U-Net Model Definition
            m1_stage1   = m1(inputs             =  source, 
                             num_classes        =  num_classes, 
                             dropout_mode       =  dropout_mode,
                             dropout_rate       =  dropout_rate,          
                             filters            = [x//2 for x in filters],            
                             strides            =  strides,            
                             se_reduction       =  se_reduction,          
                             att_sub_samp       =  att_sub_samp,                   
                             kernel_initializer =  kernel_initializer,    
                             bias_initializer   =  bias_initializer,     
                             kernel_regularizer =  kernel_regularizer,    
                             bias_regularizer   =  bias_regularizer,
                             target_tensor_name = 'stage_1_label')

            # Second-Stage Dual-Attention U-Net Model Definition
            m1_stage2   = m1(inputs             =  tf.keras.layers.concatenate([tf.expand_dims(
                                                   m1_stage1['y_softmax'][:,:,:,:,1],axis=-1), source], axis=-1), 
                             num_classes        =  num_classes, 
                             dropout_rate       =  dropout_rate, 
                             dropout_mode       =  dropout_mode,         
                             filters            =  filters,            
                             strides            =  strides,            
                             se_reduction       =  se_reduction,          
                             att_sub_samp       =  att_sub_samp,               
                             kernel_initializer =  kernel_initializer,    
                             bias_initializer   =  bias_initializer,     
                             kernel_regularizer =  kernel_regularizer,    
                             bias_regularizer   =  bias_regularizer,
                             target_tensor_name = 'stage_2_label')
    
            super().__init__(name='cad', inputs=[source], outputs=[m1_stage1['logits'], m1_stage2['logits']])
    
            # Cache Pointers to Layers/Tensors for Future Reference
            self.references           = LoadableModel.ReferenceContainer()
            self.references.m1_stage1 = m1_stage1
            self.references.m1_stage2 = m1_stage2
            self.encoding             = label_encoding
            self.cascaded             = cascaded

    def get_detect_model(self):
        """
        Returns Reconfigured Model to Predict Lesion Probabilities Only.
        """
        if self.cascaded:
            if (self.encoding=='ordinal'): return tf.keras.Model(self.inputs, [self.references.m1_stage1['y_sigmoid'], self.references.m1_stage2['y_sigmoid']])
            if (self.encoding=='one_hot'): return tf.keras.Model(self.inputs, [self.references.m1_stage1['y_softmax'], self.references.m1_stage2['y_softmax']])
        else:
            if (self.encoding=='ordinal'): return tf.keras.Model(self.inputs, self.references.m1_model['y_sigmoid'])
            if (self.encoding=='one_hot'): return tf.keras.Model(self.inputs, self.references.m1_model['y_softmax'])
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
































# 3D SEResNet BottleNeck Module -------------------------------------------------------------------------------------------------------------------------------------
def SEResNetBottleNeck(filters, conv_params, reduction=16, strides=(1,1,1)):
    """
    [1] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    """
    def layer(input_tensor):     # Define Operations as a Layer
        x        = input_tensor
        residual = input_tensor

        # Bottleneck
        x = tf.keras.layers.Conv3D(filters=filters//4, kernel_size=(1,1,1), strides=strides, **conv_params)(x)
        x = tf.keras.layers.BatchNormalization(epsilon=9.999999747378752e-06)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Conv3D(filters=filters//4, kernel_size=(3,3,3), strides=(1,1,1), **conv_params)(x)
        x = tf.keras.layers.BatchNormalization(epsilon=9.999999747378752e-06)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Conv3D(filters=filters, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(x)
        x = tf.keras.layers.BatchNormalization(epsilon=9.999999747378752e-06)(x)

        # Replicate Operations with Residual Connection (change in #num_filters or spatial_dims)
        x_channels = x.get_shape()[-1]
        r_channels = residual.get_shape()[-1]
        if (strides!=1)|(x_channels!=r_channels):
            residual = tf.keras.layers.Conv3D(filters=x_channels, kernel_size=(1,1,1), strides=strides, **conv_params)(residual)
            residual = tf.keras.layers.BatchNormalization(epsilon=9.999999747378752e-06)(residual)

        # Attention Module
        x = ChannelSE(reduction=reduction)(x)

        # Residual Addition
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return x
    return layer

# Squeeze-and-Excitation Block
def ChannelSE(reduction=16):
    """
    [1] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    """
    def layer(input_tensor):     # Define Operations as a Layer
        channels = input_tensor.get_shape()[-1]
        x        = input_tensor

        # Squeeze-and-Excitation Block (originally derived from PyTorch)
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        x = tf.keras.layers.Lambda(function=ExpandDims)(x)
        x = tf.keras.layers.Conv3D(filters=channels//reduction, kernel_size=(1,1,1), strides=(1,1,1))(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Conv3D(filters=channels, kernel_size=(1,1,1), strides=(1,1,1))(x)
        x = tf.keras.layers.Activation('sigmoid')(x)

        # Attention
        x = tf.keras.layers.Multiply()([input_tensor, x])
        return x
    return layer    

def ExpandDims(x):
    return x[:,None,None,None,:]
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


# 3D Attention Gating Module -----------------------------------------------------------------------------------------------------------------------------------------
def GridGating3D(input_tensor, output_filters, conv_params, kernel_size=(1,1,1)):
    '''
    [1] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    '''
    x = tf.keras.layers.Conv3D(filters=output_filters, kernel_size=kernel_size, strides=(1,1,1), **conv_params)(input_tensor)
    x = tf.keras.layers.BatchNormalization(epsilon=9.999999747378752e-06)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x

def GridAttentionBlock3D(conv_tensor, gating_tensor, conv_params, inter_channels=None, sub_samp=(2,2,2)):
    '''
    [1] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    '''
    x = conv_tensor
    g = gating_tensor

    if (inter_channels==None):
        inter_channels = (x.get_shape()[-1]) // 2
        if (inter_channels==0): inter_channels = 1

    # Attention Gating Function (theta^T * x_ij + phi^T * gating_signal + bias)
    theta_x     = tf.keras.layers.Conv3D(filters=inter_channels, kernel_size=sub_samp, strides=sub_samp, **conv_params)(x)
    phi_g       = tf.keras.layers.Conv3D(filters=inter_channels, kernel_size=(1,1,1),  strides=(1,1,1),  **conv_params)(g)
    scale_z     = theta_x.get_shape()[1] // phi_g.get_shape()[1]
    scale_x     = theta_x.get_shape()[2] // phi_g.get_shape()[2]
    scale_y     = theta_x.get_shape()[3] // phi_g.get_shape()[3]
    phi_g       = tf.keras.layers.UpSampling3D(size=(scale_z,scale_x,scale_y))(phi_g)
    f           = tf.keras.layers.LeakyReLU(alpha=0.1)(theta_x+phi_g)
    psi_f       = tf.keras.layers.Conv3D(filters=1, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(f)
    sigm_psi_f  = tf.keras.layers.Activation('sigmoid')(psi_f)
    scale_z     = x.get_shape()[1] // sigm_psi_f.get_shape()[1]
    scale_x     = x.get_shape()[2] // sigm_psi_f.get_shape()[2]
    scale_y     = x.get_shape()[3] // sigm_psi_f.get_shape()[3]
    sigm_psi_f  = tf.keras.layers.UpSampling3D(size=(scale_z,scale_x,scale_y))(sigm_psi_f)
    y           = sigm_psi_f * x

    # Output Projection
    W_y         = tf.keras.layers.Conv3D(filters=(x.get_shape()[-1]), kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(y)
    W_y         = tf.keras.layers.BatchNormalization(epsilon=9.999999747378752e-06)(W_y)

    return W_y, sigm_psi_f

# Monte Carlo Dropout
class MonteCarloDropout(tf.keras.layers.Layer):
    def __init__(self, rate):
        super(MonteCarloDropout, self).__init__()
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, rate=self.rate)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------




# Dual-Attention U-Net --------------------------------------------------------------------------------------------------------------------------------------------------
def m1(inputs, num_classes, 
       dropout_mode       = 'standard',
       dropout_rate       =  0.10,
       filters            = (16,32,64,128,256), 
       strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,1,1)),
       se_reduction       = (8,8,8,8,8),
       att_sub_samp       = ((2,2,2),(2,2,2),(1,1,1)),
       kernel_initializer = tf.initializers.GlorotUniform(),
       bias_initializer   = tf.zeros_initializer(),
       kernel_regularizer = tf.keras.regularizers.l2(1e-3),
       bias_regularizer   = tf.keras.regularizers.l2(1e-3),
       target_tensor_name = 'detection_label'):
    """
    [1] Z. Zhou et al. (2019), "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", IEEE TMI.
    [2] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    [3] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    [4] K. He et al.(2016), "Deep Residual Learning for Image Recognition", IEEE CVPR.
    [5] K. He et al.(2016), "Identity Mappings in Deep Residual Networks", ECCV.
    """
    # Preamble
    x           = inputs
    outputs     = {}
    
    conv_params = {'padding':           'same',
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer':   bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer':   bias_regularizer}

    if   (dropout_mode=='standard'):     DropoutFunc = tf.keras.layers.Dropout
    elif (dropout_mode=='monte-carlo'):  DropoutFunc = MonteCarloDropout

    # Preliminary Convolutional Layer
    x     = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(3,3,3), strides=strides[0], **conv_params)(x)
    x     = tf.keras.layers.BatchNormalization(epsilon=9.999999747378752e-06)(x)
    x     = tf.keras.layers.LeakyReLU(alpha=0.1)(x)  

    # Encoder: Backbone SE-Residual Blocks for Feature Extraction
    conv1 = SEResNetBottleNeck(filters=filters[1], strides=strides[1], reduction=se_reduction[1], conv_params=conv_params)(x)  
    conv2 = SEResNetBottleNeck(filters=filters[2], strides=strides[2], reduction=se_reduction[2], conv_params=conv_params)(conv1)  
    conv3 = SEResNetBottleNeck(filters=filters[3], strides=strides[3], reduction=se_reduction[3], conv_params=conv_params)(conv2)  

    # Middle Stage
    conv3 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(1,2,2))(conv3)
    pool3 = DropoutFunc(dropout_rate)(pool3)
    convm = tf.keras.layers.Conv3D(filters=filters[4], strides=strides[4], kernel_size=(3,3,3),    **conv_params)(pool3)
    convm = SEResNetBottleNeck(filters=filters[4], strides=strides[4], reduction=se_reduction[4], conv_params=conv_params)(convm)
    convm = tf.keras.layers.LeakyReLU(alpha=0.1)(convm)

    # Grid Attention Gating
    att_gate        = GridGating3D(input_tensor=convm, output_filters=filters[3], kernel_size=(1,1,1), conv_params=conv_params)
    att_conv1,att_1 = GridAttentionBlock3D(conv_tensor=conv1, gating_tensor=att_gate, inter_channels=filters[1], sub_samp=att_sub_samp[0], conv_params=conv_params)
    att_conv2,att_2 = GridAttentionBlock3D(conv_tensor=conv2, gating_tensor=att_gate, inter_channels=filters[2], sub_samp=att_sub_samp[1], conv_params=conv_params)
    att_conv3,att_3 = GridAttentionBlock3D(conv_tensor=conv3, gating_tensor=att_gate, inter_channels=filters[3], sub_samp=att_sub_samp[2], conv_params=conv_params)

    # Decoder: Nested U-Net - Stage 3
    deconv3     = tf.keras.layers.Conv3DTranspose(filters=filters[3], kernel_size=(3,3,3), strides=(1,2,2), padding="same")(convm)
    deconv3_up1 = tf.keras.layers.Conv3DTranspose(filters=filters[3], kernel_size=(3,3,3), strides=(2,2,2), padding="same")(deconv3)
    deconv3_up2 = tf.keras.layers.Conv3DTranspose(filters=filters[3], kernel_size=(3,3,3), strides=(1,2,2), padding="same")(deconv3_up1)
    uconv3      = tf.keras.layers.concatenate([deconv3, att_conv3])    
    uconv3      = DropoutFunc(dropout_rate)(uconv3)
    uconv3      = tf.keras.layers.Conv3D(filters=filters[3], strides=(1,1,1), kernel_size=(3,3,3), **conv_params)(uconv3)
    uconv3      = SEResNetBottleNeck(filters=filters[3], strides=(1,1,1), reduction=se_reduction[3], conv_params=conv_params)(uconv3)
    uconv3      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv3)
  
    # Decoder: Nested U-Net - Stage 2
    deconv2     = tf.keras.layers.Conv3DTranspose(filters=filters[2], kernel_size=(3,3,3), strides=(2,2,2), padding="same")(uconv3)
    deconv2_up1 = tf.keras.layers.Conv3DTranspose(filters=filters[2], kernel_size=(3,3,3), strides=(1,2,2), padding="same")(deconv2)
    uconv2      = tf.keras.layers.concatenate([deconv2, deconv3_up1, att_conv2]) 
    uconv2      = DropoutFunc(dropout_rate)(uconv2)
    uconv2      = tf.keras.layers.Conv3D(filters=filters[2], strides=(1,1,1), kernel_size=(3,3,3), **conv_params)(uconv2)
    uconv2      = SEResNetBottleNeck(filters=filters[2], strides=(1,1,1), reduction=se_reduction[2], conv_params=conv_params)(uconv2)
    uconv2      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv2)

    # Decoder: Nested U-Net - Stage 1
    deconv1     = tf.keras.layers.Conv3DTranspose(filters=filters[1], kernel_size=(3,3,3), strides=(1,2,2), padding="same")(uconv2)
    uconv1      = tf.keras.layers.concatenate([deconv1, deconv2_up1, deconv3_up2, att_conv1])
    uconv1      = DropoutFunc(dropout_rate)(uconv1)
    uconv1      = tf.keras.layers.Conv3D(filters=filters[1], strides=(1,1,1), kernel_size=(3,3,3), **conv_params)(uconv1)
    uconv1      = SEResNetBottleNeck(filters=filters[1], strides=(1,1,1), reduction=se_reduction[1], conv_params=conv_params)(uconv1)
    uconv1      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv1)

    # Decoder: Nested U-Net - Stage 0
    uconv0      = tf.keras.layers.Conv3DTranspose(filters=filters[0], kernel_size=(3,3,3), strides=(1,2,2), padding="same")(uconv1)   
    uconv0      = DropoutFunc(dropout_rate)(uconv0)
    uconv0      = tf.keras.layers.Conv3D(filters=filters[0], strides=(1,1,1), kernel_size=(3,3,3), **conv_params)(uconv0)
    uconv0      = SEResNetBottleNeck(filters=filters[0], strides=(1,1,1), reduction=se_reduction[0], conv_params=conv_params)(uconv0)
    uconv0      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv0)
    uconv0      = DropoutFunc(dropout_rate/2)(uconv0)

    # Final Convolutional Layer [Logits] + Softmax/Argmax
    y__         = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=(1,1,1), strides=(1,1,1), name=target_tensor_name, **conv_params)(uconv0)
    y_prob      = tf.nn.softmax(y__)
    y_          = tf.argmax(y__, axis=-1) \
                        if num_classes>1  \
                        else tf.cast(tf.greater_equal(y__[..., 0], 0.5), tf.int32)
    
    outputs['pre_logits'] = uconv0
    outputs['logits']     = y__
    outputs['y_softmax']  = y_prob    
    outputs['y_sigmoid']  = tf.keras.activations.sigmoid(y__)    
    outputs['y_']         = y_

    return outputs  
# -------------------------------------------------------------------------------------------------------------------------------------------------------------













































































































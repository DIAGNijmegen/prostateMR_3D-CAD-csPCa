from __future__ import unicode_literals, absolute_import, print_function, division
from models.model_ops import upsample_and_concat, linear_upsample_3d
import tensorflow as tf
import numpy as np


'''
Prostate Cancer Detection in bpMRI
Script:         Detector Definition
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         26/10/2020

'''



# Pre-Activation 3D Residual Block-----------------------------------------------------------------------------------------------------------------------------
def residual_block_3d(input_tensor, input_channels=None, output_channels=None,
                      kernel_size=(3,3,3), stride=(1,1,1),
                      kernel_initializer = tf.initializers.variance_scaling(distribution='uniform'), 
                      kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), 
                      mode               = tf.estimator.ModeKeys.EVAL):
    """
    [1] K. He et al.(2016), "Deep Residual Learning for Image Recognition", IEEE CVPR.
    [2] K. He et al.(2016), "Identity Mappings in Deep Residual Networks", ECCV.
    """
    # Define Target Channel Shapes
    if output_channels is None:
      output_channels = input_tensor.get_shape()[-1].value
    if input_channels is None:
      input_channels  = input_tensor.get_shape()[-1].value

    conv_params  = {'padding':           'same',
                    'kernel_initializer': kernel_initializer,
                    'kernel_regularizer': kernel_regularizer,
                    'data_format':       'channels_last'}

    # First Convolutional Layer
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN)(input_tensor)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv3D(filters=input_channels, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(x)

    # Second Convolutional Layer    
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv3D(filters=input_channels, kernel_size=kernel_size, strides=stride, **conv_params)(x)

    # Third Convolutional Layer
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv3D(filters=output_channels, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(x)

    if (input_channels!=output_channels)|(stride!=1):
      input_tensor = tf.keras.layers.Conv3D(filters=output_channels, kernel_size=kernel_size, strides=stride, **conv_params)(input_tensor)

    # Residual Addition
    x = tf.keras.layers.Add()([x,input_tensor])
    return x
# -------------------------------------------------------------------------------------------------------------------------------------------------------------


# 3D Attention Block ------------------------------------------------------------------------------------------------------------------------------------------
def attention_block_3d(input_tensor, input_channels=None, output_channels=None, 
                       encoder_depth=1, p=1,t=2,r=1, 
                       kernel_initializer = tf.initializers.variance_scaling(distribution='uniform'), 
                       kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), 
                       mode               = tf.estimator.ModeKeys.EVAL): 
    """
    [1] F. Wang et al.(2017), "Residual Attention Network for Image Classification", IEEE CVPR.
    """
    # Define Target Channel Shapes
    if output_channels is None:
      output_channels = input_tensor.get_shape()[-1].value
    if input_channels is None:
      input_channels  = input_tensor.get_shape()[-1].value


    conv_params  = {'padding':           'same',
                    'kernel_initializer': kernel_initializer,
                    'kernel_regularizer': kernel_regularizer,
                    'data_format':       'channels_last'}

    # First Residual Block & Trunk Branch
    for i in range(p): input_tensor       = residual_block_3d(input_tensor, mode=mode)                                
    output_trunk                          = input_tensor                                                   
    for i in range(t): output_trunk       = residual_block_3d(output_trunk, output_channels=output_channels, mode=mode)   
    output_soft_mask                      = tf.keras.layers.MaxPool3D(padding='same')(input_tensor)
    for i in range(r): output_soft_mask   = residual_block_3d(output_soft_mask, mode=mode)
  
    # Encoder Stage
    skip_connections = []
    for i in range(encoder_depth - 1):
      output_skip_connection              = residual_block_3d(output_soft_mask, mode=mode)                               
      skip_connections.append(output_skip_connection)
      output_soft_mask                    = tf.keras.layers.MaxPool3D(padding='same')(output_soft_mask)
      for _ in range(r): output_soft_mask = residual_block_3d(output_soft_mask, mode=mode)

    # Decoder Stage
    skip_connections                      = list(reversed(skip_connections))
    for i in range(encoder_depth-1):
      for _ in range(r): output_soft_mask = residual_block_3d(output_soft_mask, mode=mode)
      output_soft_mask                    = tf.keras.layers.UpSampling3D(data_format='channels_last')(output_soft_mask)
      output_soft_mask                    = tf.keras.layers.Add()([output_soft_mask,skip_connections[i]])                

    # Final Upsampling
    for i in range(r): output_soft_mask   = residual_block_3d(output_soft_mask, mode=mode)
    output_soft_mask                      = tf.keras.layers.UpSampling3D(data_format='channels_last')(output_soft_mask)

    # Final Convolutional Layers
    output_soft_mask = tf.keras.layers.Conv3D(filters=output_channels, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(output_soft_mask)
    output_soft_mask = tf.keras.layers.Conv3D(filters=output_channels, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(output_soft_mask)
    output_soft_mask = tf.keras.layers.Activation("sigmoid")(output_soft_mask)
    
    # Attention Mechanism: {(1+output_soft_mask)*output_trunk}
    output           = tf.keras.layers.Lambda(lambda x: x+1)(output_soft_mask)
    output           = tf.keras.layers.Multiply()([output, output_trunk])

    # Last Residual Block
    for i in range(p): output = residual_block_3d(output, mode=mode)
    return output
# -------------------------------------------------------------------------------------------------------------------------------------------------------------


# 3D SEResNet BottleNeck Module -------------------------------------------------------------------------------------------------------------------------------
def seresnet_bottleneck_3d(filters, reduction=16, strides=(1,1,1), groups=None, mode=tf.estimator.ModeKeys.EVAL):
    """
    [1] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    """
    def layer(input_tensor):     # Define Operations as a Layer
        x        = input_tensor
        residual = input_tensor

        # Bottleneck
        x = tf.keras.layers.Conv3D(filters=filters//4, kernel_size=(1,1,1), strides=strides, use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.ZeroPadding3D(padding=(1,1,1))(x)
        x = tf.keras.layers.Conv3D(filters=filters//4, kernel_size=(3,3,3), strides=(1,1,1), use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv3D(filters=filters, kernel_size=(1,1,1), strides=(1,1,1), use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)

        # Replicate Operations with Residual Connection (change in #num_filters or spatial_dims)
        x_channels = x.get_shape()[-1].value
        r_channels = residual.get_shape()[-1].value
        if (strides!=1)|(x_channels!=r_channels):
            residual = tf.keras.layers.Conv3D(filters=x_channels, kernel_size=(1,1,1), strides=strides, use_bias=False, kernel_initializer='he_uniform')(residual)
            residual = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(residual)

        # Attention Module
        x = ChannelSE(reduction=reduction)(x)

        # Residual Addition
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    return layer

# Squeeze-and-Excitation Block
def ChannelSE(reduction=16):
    """
    [1] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    """
    def layer(input_tensor):     # Define Operations as a Layer
        channels = input_tensor.get_shape()[-1].value
        x        = input_tensor

        # Squeeze-and-Excitation Block (originally derived from PyTorch)
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        x = tf.keras.layers.Lambda(function=se_expand_dims)(x)
        x = tf.keras.layers.Conv3D(filters=channels//reduction, kernel_size=(1,1,1), strides=(1,1,1), use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv3D(filters=channels, kernel_size=(1,1,1), strides=(1,1,1), use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Activation('sigmoid')(x)

        # Attention
        x = tf.keras.layers.Multiply()([input_tensor, x])
        return x
    return layer    

def se_expand_dims(x):
    return x[:,None,None,None,:]
# -------------------------------------------------------------------------------------------------------------------------------------------------------------


# 3D Attention Gating Module ----------------------------------------------------------------------------------------------------------------------------------
def GridGating3D(input_tensor, output_filters, kernel_size=(1,1,1), mode=tf.estimator.ModeKeys.EVAL):
    '''
    [1] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    '''
    x = tf.keras.layers.Conv3D(filters=output_filters, kernel_size=kernel_size, strides=(1,1,1), use_bias=False, kernel_initializer='he_uniform')(input_tensor)
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def GridAttentionBlock3D(conv_tensor, gating_tensor, inter_channels=None, 
                         sub_samp=(2,2,2), mode=tf.estimator.ModeKeys.EVAL):
    '''
    [1] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    '''
    # Preamble
    x = conv_tensor
    g = gating_tensor

    if (inter_channels==None):
      inter_channels = (x.get_shape()[-1].value) // 2
      if (inter_channels==0): inter_channels = 1

    # Attention Gating Function (theta^T * x_ij + phi^T * gating_signal + bias)
    theta_x     = tf.keras.layers.Conv3D(filters=inter_channels, kernel_size=sub_samp, strides=sub_samp, use_bias=False, kernel_initializer='he_uniform')(x)
    phi_g       = tf.keras.layers.Conv3D(filters=inter_channels, kernel_size=(1,1,1),  strides=(1,1,1),  use_bias=True,  kernel_initializer='he_uniform')(g)
    scale_z     = theta_x.get_shape()[1].value//phi_g.get_shape()[1].value
    scale_x     = theta_x.get_shape()[2].value//phi_g.get_shape()[2].value
    scale_y     = theta_x.get_shape()[3].value//phi_g.get_shape()[3].value
    phi_g       = tf.keras.layers.UpSampling3D(size=(scale_z,scale_x,scale_y))(phi_g)
    f           = tf.keras.layers.Activation('relu')(theta_x+phi_g)
    psi_f       = tf.keras.layers.Conv3D(filters=1, kernel_size=(1,1,1), strides=(1,1,1), use_bias=True, kernel_initializer='he_uniform')(f)
    sigm_psi_f  = tf.keras.layers.Activation('sigmoid')(psi_f)
    scale_z     = x.get_shape()[1].value//sigm_psi_f.get_shape()[1].value
    scale_x     = x.get_shape()[2].value//sigm_psi_f.get_shape()[2].value
    scale_y     = x.get_shape()[3].value//sigm_psi_f.get_shape()[3].value
    sigm_psi_f  = tf.keras.layers.UpSampling3D(size=(scale_z,scale_x,scale_y))(sigm_psi_f)
    y           = sigm_psi_f * x

    # Output Projection
    W_y         = tf.keras.layers.Conv3D(filters=(x.get_shape()[-1].value), kernel_size=(1,1,1), strides=(1,1,1), use_bias=False, kernel_initializer='he_uniform')(y)
    W_y         = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(W_y)

    return W_y, sigm_psi_f
# -------------------------------------------------------------------------------------------------------------------------------------------------------------










# Dual-Attention U-Net --------------------------------------------------------------------------------------------------------------------------------------- 
def m1(inputs, num_classes, dropout_rate = 0.1,
       filters            = (16,32,64,128,256), 
       strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,1,1)),
       reduction          = (8,8,8,8,8),
       att_sub_samp       = ((2,2,2),(2,2,2),(1,1,1)),
       use_bias           = False,
       kernel_initializer = tf.initializers.variance_scaling(distribution='uniform'),
       bias_initializer   = tf.zeros_initializer(),
       kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
       bias_regularizer   = tf.contrib.layers.l2_regularizer(1e-3),
       mode               = tf.estimator.ModeKeys.EVAL):
    """
    [1] Z. Zhou et al. (2019), "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", IEEE TMI.
    [2] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    [3] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    [4] K. He et al.(2016), "Deep Residual Learning for Image Recognition", IEEE CVPR.
    [5] K. He et al.(2016), "Identity Mappings in Deep Residual Networks", ECCV.
    """
    # Preamble
    x             = inputs
    outputs       = {}
    
    conv_params = {'padding':           'same',
                   'use_bias':           use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer':   bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer':   bias_regularizer}

    # Preliminary Convolutional Layer
    x     = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(3,3,3), strides=strides[0], **conv_params)(x)
    x     = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
    x     = tf.keras.layers.Activation('relu')(x)  

    # Encoder: Backbone SE-Residual Blocks for Feature Extraction
    conv1 = seresnet_bottleneck_3d(filters=filters[1], strides=strides[1], reduction=reduction[1], mode=mode)(x)  
    conv2 = seresnet_bottleneck_3d(filters=filters[2], strides=strides[2], reduction=reduction[2], mode=mode)(conv1)  
    conv3 = seresnet_bottleneck_3d(filters=filters[3], strides=strides[3], reduction=reduction[3], mode=mode)(conv2)  

    # Middle Stage
    conv3 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(1,2,2))(conv3)
    pool3 = tf.keras.layers.Dropout(dropout_rate)(pool3)
    convm = tf.keras.layers.Conv3D(filters=filters[4], strides=strides[4], kernel_size=(3,3,3),    **conv_params)(pool3)
    convm = seresnet_bottleneck_3d(filters=filters[4], strides=strides[4], reduction=reduction[4], mode=mode)(convm)
    convm = tf.keras.layers.LeakyReLU(alpha=0.1)(convm)

    # Grid Attention Gating
    att_gate         = GridGating3D(input_tensor=convm, output_filters=filters[3], kernel_size=(1,1,1), mode=mode)
    att_conv1, att_1 = GridAttentionBlock3D(conv_tensor=conv1, gating_tensor=att_gate, inter_channels=filters[1], sub_samp=att_sub_samp[0], mode=mode)
    att_conv2, att_2 = GridAttentionBlock3D(conv_tensor=conv2, gating_tensor=att_gate, inter_channels=filters[2], sub_samp=att_sub_samp[1], mode=mode)
    att_conv3, att_3 = GridAttentionBlock3D(conv_tensor=conv3, gating_tensor=att_gate, inter_channels=filters[3], sub_samp=att_sub_samp[2], mode=mode)


    # Decoder: Nested U-Net - Stage 3
    deconv3     = tf.keras.layers.Conv3DTranspose(filters=filters[3], kernel_size=(3,3,3), strides=(1,2,2), padding="same")(convm)
    deconv3_up1 = tf.keras.layers.Conv3DTranspose(filters=filters[3], kernel_size=(3,3,3), strides=(2,2,2), padding="same")(deconv3)
    deconv3_up2 = tf.keras.layers.Conv3DTranspose(filters=filters[3], kernel_size=(3,3,3), strides=(1,2,2), padding="same")(deconv3_up1)
    uconv3      = tf.keras.layers.concatenate([deconv3, att_conv3])    
    uconv3      = tf.keras.layers.Dropout(dropout_rate)(uconv3)
    uconv3      = tf.keras.layers.Conv3D(filters=filters[3], strides=(1,1,1), kernel_size=(3,3,3),    **conv_params)(uconv3)
    uconv3      = seresnet_bottleneck_3d(filters=filters[3], strides=(1,1,1), reduction=reduction[3], mode=mode)(uconv3)
    uconv3      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv3)
  

    # Decoder: Nested U-Net - Stage 2
    deconv2     = tf.keras.layers.Conv3DTranspose(filters=filters[2], kernel_size=(3,3,3), strides=(2,2,2), padding="same")(uconv3)
    deconv2_up1 = tf.keras.layers.Conv3DTranspose(filters=filters[2], kernel_size=(3,3,3), strides=(1,2,2), padding="same")(deconv2)
    uconv2      = tf.keras.layers.concatenate([deconv2, deconv3_up1, att_conv2]) 
    uconv2      = tf.keras.layers.Dropout(dropout_rate)(uconv2)
    uconv2      = tf.keras.layers.Conv3D(filters=filters[2], strides=(1,1,1), kernel_size=(3,3,3),    **conv_params)(uconv2)
    uconv2      = seresnet_bottleneck_3d(filters=filters[2], strides=(1,1,1), reduction=reduction[2], mode=mode)(uconv2)
    uconv2      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv2)


    # Decoder: Nested U-Net - Stage 1
    deconv1     = tf.keras.layers.Conv3DTranspose(filters=filters[1], kernel_size=(3,3,3), strides=(1,2,2), padding="same")(uconv2)
    uconv1      = tf.keras.layers.concatenate([deconv1, deconv2_up1, deconv3_up2, att_conv1])
    uconv1      = tf.keras.layers.Dropout(dropout_rate)(uconv1)
    uconv1      = tf.keras.layers.Conv3D(filters=filters[1], strides=(1,1,1), kernel_size=(3,3,3),    **conv_params)(uconv1)
    uconv1      = seresnet_bottleneck_3d(filters=filters[1], strides=(1,1,1), reduction=reduction[1], mode=mode)(uconv1)
    uconv1      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv1)


    # Decoder: Nested U-Net - Stage 0
    uconv0      = tf.keras.layers.Conv3DTranspose(filters=filters[0], kernel_size=(3,3,3), strides=(1,2,2), padding="same")(uconv1)   
    uconv0      = tf.keras.layers.Dropout(dropout_rate)(uconv0)
    uconv0      = tf.keras.layers.Conv3D(filters=filters[0], strides=(1,1,1), kernel_size=(3,3,3),    **conv_params)(uconv0)
    uconv0      = seresnet_bottleneck_3d(filters=filters[0], strides=(1,1,1), reduction=reduction[0], mode=mode)(uconv0)
    uconv0      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv0)
    uconv0      = tf.keras.layers.Dropout(dropout_rate/2)(uconv0)


    # Final Convolutional Layer [Logits] + Softmax/Argmax
    y__         = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(uconv0)
    y_prob      = tf.nn.softmax(y__)
    y_          = tf.argmax(y__, axis=-1) \
                        if num_classes>1  \
                        else tf.cast(tf.greater_equal(y__[..., 0], 0.5), tf.int32)
    
    # Model Summary
    print('Input Volume:---------------------------', inputs.get_shape())
    print('Initial Convolutional Layer:------------', x.get_shape())
    print('Encoder: Stage 1; SE-Residual Block:----', conv1.get_shape())
    print('Encoder: Stage 2; SE-Residual Block:----', conv2.get_shape())
    print('Encoder: Stage 3; SE-Residual Block:----', conv3.get_shape())
    print('Middle: High-Dim Latent Features:-------', convm.get_shape())
    print('Attention Gating: Middle Gate:----------', att_gate.get_shape())
    print('Attention Gating: Stage 1:--------------', att_conv1.get_shape())
    print('Attention Gating: Stage 2:--------------', att_conv2.get_shape())    
    print('Attention Gating: Stage 3:--------------', att_conv3.get_shape())    
    print('Decoder: Stage 3; Nested U-Net:---------', uconv3.get_shape())
    print('Decoder: Stage 2; Nested U-Net:---------', uconv2.get_shape())
    print('Decoder: Stage 1; Nested U-Net:---------', uconv1.get_shape())    
    print('Decoder: Stage 0; Nested U-Net:---------', uconv0.get_shape())    
    print('Final Convolutional Layer [Logits]:-----', y__.get_shape())

    outputs['logits']    = y__
    outputs['y_prob']    = y_prob    
    outputs['y_']        = y_

    return outputs  
# -------------------------------------------------------------------------------------------------------------------------------------------------------------



# Cascaded Dual-Attention U-Net ------------------------------------------------------------------------------------------------------------------------------- 
def cascaded_m1(inputs, num_classes, dropout_rate = 0.1,
                filters            = (16,32,64,128,256), 
                strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,1,1)),
                reduction          = (8,8,8,8,8),
                att_sub_samp       = ((2,2,2),(2,2,2),(1,1,1)),
                use_bias           = False,
                kernel_initializer = tf.initializers.variance_scaling(distribution='uniform'),
                bias_initializer   = tf.zeros_initializer(),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                bias_regularizer   = tf.contrib.layers.l2_regularizer(1e-3),
                mode               = tf.estimator.ModeKeys.EVAL):

    x                             = inputs
    outputs                       = {}

    print('Stage 1: Coarse Segmentation')
    stage1_output_ops = m1(
        inputs                    = x,
        num_classes               = num_classes,
        dropout_rate              = 0.50,
        filters                   = [x//2 for x in filters],
        strides                   = strides,
        reduction                 = reduction,
        att_sub_samp              = att_sub_samp,
        mode                      = mode,
        kernel_initializer        = kernel_initializer,
        kernel_regularizer        = kernel_regularizer) 

    print('Stage 2: Fine Segmentation')
    stage2_output_ops = m1(
        inputs                    = tf.keras.layers.concatenate([x, stage1_output_ops['y_prob']]),
        num_classes               = num_classes,
        dropout_rate              = 0.50,
        filters                   = filters,
        strides                   = strides,
        reduction                 = reduction,
        att_sub_samp              = att_sub_samp,
        mode                      = mode,
        kernel_initializer        = kernel_initializer,
        kernel_regularizer        = kernel_regularizer) 

    outputs['stage1_logits']      = stage1_output_ops['logits'] 
    outputs['stage1_y_prob']      = stage1_output_ops['y_prob']    
    outputs['stage1_y_']          = stage1_output_ops['y_']     
    outputs['stage2_logits']      = stage2_output_ops['logits'] 
    outputs['stage2_y_prob']      = stage2_output_ops['y_prob']    
    outputs['stage2_y_']          = stage2_output_ops['y_']    

    return outputs
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
























































# 2D SEResNet BottleNeck Module -------------------------------------------------------------------------------------------------------------------------------
def seresnet_bottleneck_2d(filters, reduction=16, strides=(1,1), groups=None, mode=tf.estimator.ModeKeys.EVAL):
    """
    [1] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    """
    def layer(input_tensor):     # Define Operations as a Layer
        x        = input_tensor
        residual = input_tensor

        # Bottleneck
        x = tf.keras.layers.Conv2D(filters=filters//4, kernel_size=(1,1), strides=strides, use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
        x = tf.keras.layers.Conv2D(filters=filters//4, kernel_size=(3,3), strides=(1,1), use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)

        # Replicate Operations with Residual Connection (change in #num_filters or spatial_dims)
        x_channels = x.get_shape()[-1].value
        r_channels = residual.get_shape()[-1].value
        if (strides!=1)|(x_channels!=r_channels):
            residual = tf.keras.layers.Conv2D(filters=x_channels, kernel_size=(1,1), strides=strides, use_bias=False, kernel_initializer='he_uniform')(residual)
            residual = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(residual)

        # Attention Module
        x = ChannelSE_2d(reduction=reduction)(x)

        # Residual Addition
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    return layer

# Squeeze-and-Excitation Block
def ChannelSE_2d(reduction=16):
    """
    [1] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    """
    def layer(input_tensor):     # Define Operations as a Layer
        channels = input_tensor.get_shape()[-1].value
        x        = input_tensor

        # Squeeze-and-Excitation Block (originally derived from PyTorch)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Lambda(function=se_expand_dims_2d)(x)
        x = tf.keras.layers.Conv2D(filters=channels//reduction, kernel_size=(1,1), strides=(1,1), use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters=channels, kernel_size=(1,1), strides=(1,1), use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Activation('sigmoid')(x)

        # Attention
        x = tf.keras.layers.Multiply()([input_tensor, x])
        return x
    return layer    

def se_expand_dims_2d(x):
    return x[:,None,None,:]


# 2D Attention Gating Module ----------------------------------------------------------------------------------------------------------------------------------
def GridGating2D(input_tensor, output_filters, kernel_size=(1,1), mode=tf.estimator.ModeKeys.EVAL):
    '''
    [1] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    '''
    x = tf.keras.layers.Conv2D(filters=output_filters, kernel_size=kernel_size, strides=(1,1), use_bias=False, kernel_initializer='he_uniform')(input_tensor)
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def GridAttentionBlock2D(conv_tensor, gating_tensor, inter_channels=None, 
                         sub_samp=(2,2), mode=tf.estimator.ModeKeys.EVAL):
    '''
    [1] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    '''
    # Preamble
    x = conv_tensor
    g = gating_tensor

    if (inter_channels==None):
      inter_channels = (x.get_shape()[-1].value) // 2
      if (inter_channels==0): inter_channels = 1

    # Attention Gating Function (theta^T * x_ij + phi^T * gating_signal + bias)
    theta_x     = tf.keras.layers.Conv2D(filters=inter_channels, kernel_size=sub_samp, strides=sub_samp, use_bias=False, kernel_initializer='he_uniform')(x)
    phi_g       = tf.keras.layers.Conv2D(filters=inter_channels, kernel_size=(1,1),    strides=(1,1),    use_bias=True,  kernel_initializer='he_uniform')(g)
    scale_x     = theta_x.get_shape()[1].value//phi_g.get_shape()[1].value
    scale_y     = theta_x.get_shape()[2].value//phi_g.get_shape()[2].value
    phi_g       = tf.keras.layers.UpSampling2D(size=(scale_x,scale_y))(phi_g)
    f           = tf.keras.layers.Activation('relu')(theta_x+phi_g)
    psi_f       = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), use_bias=True, kernel_initializer='he_uniform')(f)
    sigm_psi_f  = tf.keras.layers.Activation('sigmoid')(psi_f)
    scale_x     = x.get_shape()[1].value//sigm_psi_f.get_shape()[1].value
    scale_y     = x.get_shape()[2].value//sigm_psi_f.get_shape()[2].value
    sigm_psi_f  = tf.keras.layers.UpSampling2D(size=(scale_x,scale_y))(sigm_psi_f)
    y           = sigm_psi_f * x

    # Output Projection
    W_y         = tf.keras.layers.Conv2D(filters=(x.get_shape()[-1].value), kernel_size=(1,1), strides=(1,1), use_bias=False, kernel_initializer='he_uniform')(y)
    W_y         = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(W_y)

    return W_y, sigm_psi_f
# -------------------------------------------------------------------------------------------------------------------------------------------------------------



# SE-Residual 2D U-Net++ w/ Grid Attention Gating ------------------------------------------------------------------------------------------------------------- 
def se_ag_residual_unet_plus_2d(inputs, num_classes, dropout_rate = 0.1,
               filters            = (16,32,64,128,256), 
               strides            = ((1,1),(2,2),(2,2),(2,2),(1,1)),
               reduction          = (8,8,8,8,8),
               att_sub_samp       = ((2,2),(2,2),(1,1)),
               use_bias           = False,
               kernel_initializer = tf.initializers.variance_scaling(distribution='uniform'),
               bias_initializer   = tf.zeros_initializer(),
               kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
               bias_regularizer   = tf.contrib.layers.l2_regularizer(1e-3),
               mode               = tf.estimator.ModeKeys.EVAL):
    """
    [1] Z. Zhou et al. (2019), "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", IEEE TMI.
    [2] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    [3] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    [4] K. He et al.(2016), "Deep Residual Learning for Image Recognition", IEEE CVPR.
    [5] K. He et al.(2016), "Identity Mappings in Deep Residual Networks", ECCV.
    """
    # Preamble
    x             = inputs
    outputs       = {}
    
    conv_params = {'padding':           'same',
                   'use_bias':           use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer':   bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer':   bias_regularizer}

    # Preliminary Convolutional Layer
    x     = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(3,3), strides=strides[0], **conv_params)(x)
    x     = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
    x     = tf.keras.layers.Activation('relu')(x)  

    # Encoder: Backbone SE-Residual Blocks for Feature Extraction
    conv1 = seresnet_bottleneck_2d(filters=filters[1], strides=strides[1], reduction=reduction[1], mode=mode)(x)  
    conv2 = seresnet_bottleneck_2d(filters=filters[2], strides=strides[2], reduction=reduction[2], mode=mode)(conv1)  
    conv3 = seresnet_bottleneck_2d(filters=filters[3], strides=strides[3], reduction=reduction[3], mode=mode)(conv2)  

    # Middle Stage
    conv3 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv3)
    pool3 = tf.keras.layers.Dropout(dropout_rate)(pool3)
    convm = tf.keras.layers.Conv2D(filters=filters[4], strides=strides[4], kernel_size=(3,3),    **conv_params)(pool3)
    convm = seresnet_bottleneck_2d(filters=filters[4], strides=strides[4], reduction=reduction[4], mode=mode)(convm)
    convm = tf.keras.layers.LeakyReLU(alpha=0.1)(convm)

    # Grid Attention Gating
    att_gate         = GridGating2D(input_tensor=convm, output_filters=filters[3], kernel_size=(1,1), mode=mode)
    att_conv1, att_1 = GridAttentionBlock2D(conv_tensor=conv1, gating_tensor=att_gate, inter_channels=filters[1], sub_samp=att_sub_samp[0], mode=mode)
    att_conv2, att_2 = GridAttentionBlock2D(conv_tensor=conv2, gating_tensor=att_gate, inter_channels=filters[2], sub_samp=att_sub_samp[1], mode=mode)
    att_conv3, att_3 = GridAttentionBlock2D(conv_tensor=conv3, gating_tensor=att_gate, inter_channels=filters[3], sub_samp=att_sub_samp[2], mode=mode)


    # Decoder: Nested U-Net - Stage 3
    deconv3     = tf.keras.layers.Conv2DTranspose(filters=filters[3], kernel_size=(3,3), strides=(2,2), padding="same")(convm)
    deconv3_up1 = tf.keras.layers.Conv2DTranspose(filters=filters[3], kernel_size=(3,3), strides=(2,2), padding="same")(deconv3)
    deconv3_up2 = tf.keras.layers.Conv2DTranspose(filters=filters[3], kernel_size=(3,3), strides=(2,2), padding="same")(deconv3_up1)
    uconv3      = tf.keras.layers.concatenate([deconv3, att_conv3])    
    uconv3      = tf.keras.layers.Dropout(dropout_rate)(uconv3)
    uconv3      = tf.keras.layers.Conv2D(filters=filters[3], strides=(1,1), kernel_size=(3,3),    **conv_params)(uconv3)
    uconv3      = seresnet_bottleneck_2d(filters=filters[3], strides=(1,1), reduction=reduction[3], mode=mode)(uconv3)
    uconv3      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv3)
  

    # Decoder: Nested U-Net - Stage 2
    deconv2     = tf.keras.layers.Conv2DTranspose(filters=filters[2], kernel_size=(3,3), strides=(2,2), padding="same")(uconv3)
    deconv2_up1 = tf.keras.layers.Conv2DTranspose(filters=filters[2], kernel_size=(3,3), strides=(2,2), padding="same")(deconv2)
    uconv2      = tf.keras.layers.concatenate([deconv2, deconv3_up1, att_conv2]) 
    uconv2      = tf.keras.layers.Dropout(dropout_rate)(uconv2)
    uconv2      = tf.keras.layers.Conv2D(filters=filters[2], strides=(1,1), kernel_size=(3,3),    **conv_params)(uconv2)
    uconv2      = seresnet_bottleneck_2d(filters=filters[2], strides=(1,1), reduction=reduction[2], mode=mode)(uconv2)
    uconv2      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv2)


    # Decoder: Nested U-Net - Stage 1
    deconv1     = tf.keras.layers.Conv2DTranspose(filters=filters[1], kernel_size=(3,3), strides=(2,2), padding="same")(uconv2)
    uconv1      = tf.keras.layers.concatenate([deconv1, deconv2_up1, deconv3_up2, att_conv1])
    uconv1      = tf.keras.layers.Dropout(dropout_rate)(uconv1)
    uconv1      = tf.keras.layers.Conv2D(filters=filters[1], strides=(1,1), kernel_size=(3,3),    **conv_params)(uconv1)
    uconv1      = seresnet_bottleneck_2d(filters=filters[1], strides=(1,1), reduction=reduction[1], mode=mode)(uconv1)
    uconv1      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv1)


    # Decoder: Nested U-Net - Stage 0
    uconv0      = tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=(3,3), strides=(2,2), padding="same")(uconv1)   
    uconv0      = tf.keras.layers.Dropout(dropout_rate)(uconv0)
    uconv0      = tf.keras.layers.Conv2D(filters=filters[0], strides=(1,1), kernel_size=(3,3),    **conv_params)(uconv0)
    uconv0      = seresnet_bottleneck_2d(filters=filters[0], strides=(1,1), reduction=reduction[0], mode=mode)(uconv0)
    uconv0      = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv0)
    uconv0      = tf.keras.layers.Dropout(dropout_rate/2)(uconv0)


    # Final Convolutional Layer [Logits] + Softmax/Argmax
    y__         = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1), **conv_params)(uconv0)
    y_prob      = tf.nn.softmax(y__)
    y_          = tf.argmax(y__, axis=-1) \
                        if num_classes>1  \
                        else tf.cast(tf.greater_equal(y__[..., 0], 0.5), tf.int32)
    
    # Model Summary
    print('Input Volume:---------------------------', inputs.get_shape())
    print('Initial Convolutional Layer:------------', x.get_shape())
    print('Encoder: Stage 1; SE-Residual Block:----', conv1.get_shape())
    print('Encoder: Stage 2; SE-Residual Block:----', conv2.get_shape())
    print('Encoder: Stage 3; SE-Residual Block:----', conv3.get_shape())
    print('Middle: High-Dim Latent Features:-------', convm.get_shape())
    print('Attention Gating: Middle Gate:----------', att_gate.get_shape())
    print('Attention Gating: Stage 1:--------------', att_conv1.get_shape())
    print('Attention Gating: Stage 2:--------------', att_conv2.get_shape())    
    print('Attention Gating: Stage 3:--------------', att_conv3.get_shape())    
    print('Decoder: Stage 3; Nested U-Net:---------', uconv3.get_shape())
    print('Decoder: Stage 2; Nested U-Net:---------', uconv2.get_shape())
    print('Decoder: Stage 1; Nested U-Net:---------', uconv1.get_shape())    
    print('Decoder: Stage 0; Nested U-Net:---------', uconv0.get_shape())    
    print('Final Convolutional Layer [Logits]:-----', y__.get_shape())

    outputs['logits']    = y__
    outputs['y_prob']    = y_prob    
    outputs['y_']        = y_

    return outputs  
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

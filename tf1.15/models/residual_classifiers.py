from __future__ import unicode_literals, absolute_import, print_function, division
import tensorflow as tf
import numpy as np

'''
Prostate Cancer Detection in bpMRI
Script:         Classifier Definitions
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         05/06/2020

'''


# 3D ResNet Model [Ref:DLTK] ---------------------------------------------------------------------------------------------------------------------------------
def resnet_3d(inputs, num_classes,
              num_res_units      =  1,
              filters            = (16, 32, 64, 128),
              strides            = ((1,1,1),(2,2,2),(2,2,2),(2,2,2)),
              mode               = tf.estimator.ModeKeys.EVAL,
              use_bias           = False,
              activation         = tf.nn.relu6,
              kernel_initializer = tf.initializers.variance_scaling(distribution='uniform'),
              bias_initializer   = tf.zeros_initializer(),
              kernel_regularizer = None, 
              bias_regularizer   = None,
              dropout_mode       = False):
    """
    [1] K. He et al.(2016), "Deep Residual Learning for Image Recognition", IEEE CVPR.
    [2] K. He et al.(2016), "Identity Mappings in Deep Residual Networks", ECCV.
    """
    # Verify Input Shape
    outputs = {}
    assert len(strides) == len(filters)
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'
    # Define Activation
    relu_op = tf.nn.relu6
    # Define Convolution Parameters
    conv_params = {'padding':           'same',
                   'use_bias':           use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer':   bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer':   bias_regularizer}
    x = inputs

    # Inital Convolution with 'filters[0]'
    k = [s * 2 if s > 1 else 3 for s in strides[0]]
    x = tf.layers.conv3d(x, filters[0], k, strides[0], **conv_params)
    tf.logging.info('Init conv tensor shape {}'.format(x.get_shape()))

    # Residual Feature Encoding Blocks ('num_res_units' at 'res_scales')
    res_scales    = [x]
    saved_strides = []
    # For Each Scale
    for res_scale in range(1, len(filters)):
        with tf.variable_scope('unit_{}_0'.format(res_scale)):
            x = vanilla_residual_unit_3d(inputs      = x,
                                         out_filters = filters[res_scale],
                                         strides     = strides[res_scale],
                                         activation  = activation,
                                         mode        = mode)
        saved_strides.append(strides[res_scale])
        # For Each Residual Block
        for i in range(1, num_res_units):
            with tf.variable_scope('unit_{}_{}'.format(res_scale, i)):
                x = vanilla_residual_unit_3d(inputs      = x,
                                             out_filters = filters[res_scale],
                                             strides     = (1,1,1),
                                             activation  = activation,
                                             mode        = mode)
        res_scales.append(x)
        tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(res_scale,x.get_shape()))

    # Global Average Pooling
    with tf.variable_scope('pool'):
        x    = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
        x    = relu_op(x)
        axis = tuple(range(len(x.get_shape().as_list())))[1:-1]
        x    = tf.reduce_mean(x, axis=axis, name='global_avg_pool')
        tf.logging.info('Global pool shape {}'.format(x.get_shape()))

    # Fully Connected Layers
    with tf.variable_scope('last'):
        x = tf.layers.dense(inputs             = x,
                            units              = num_classes,
                            activation         = None,
                            use_bias           = conv_params['use_bias'],
                            kernel_initializer = conv_params['kernel_initializer'],
                            bias_initializer   = conv_params['bias_initializer'],
                            kernel_regularizer = conv_params['kernel_regularizer'],
                            bias_regularizer   = conv_params['bias_regularizer'],
                            name               = 'hidden_units')
        if dropout_mode:
          x = tf.layers.Dropout(0.50)(x)
        tf.logging.info('Output tensor shape {}'.format(x.get_shape()))

    # Model Outputs [Softmax Class Probabilities('y_prob'), Hard Prediction('y_')]
    outputs['logits'] = x
    with tf.variable_scope('pred'):
        outputs['y_prob'] = tf.nn.softmax(x)             
        outputs['y_']     = tf.argmax(x, axis=-1) \
                             if   num_classes > 1 \
                             else tf.cast(tf.greater_equal(x[..., 0], 0.5), tf.int32)
    return outputs



# 3D Residual Unit [Ref:DLTK]
def vanilla_residual_unit_3d(inputs, out_filters,
                             kernel_size        = (3,3,3),
                             strides            = (1,1,1),
                             mode               = tf.estimator.ModeKeys.EVAL,
                             use_bias           = False,
                             activation         = tf.nn.relu6,
                             kernel_initializer = tf.initializers.variance_scaling(distribution='uniform'),
                             bias_initializer   = tf.zeros_initializer(),
                             kernel_regularizer = None,
                             bias_regularizer   = None):
    """
    [1] K. He et al.(2016), "Deep Residual Learning for Image Recognition", IEEE CVPR.
    [2] K. He et al.(2016), "Identity Mappings in Deep Residual Networks", ECCV.
    """
    # Define Pooling Operation
    pool_op     = tf.layers.max_pooling3d
    # Define Convolution Parameters
    conv_params = {'padding':           'same',
                   'use_bias':           use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer':   bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer':   bias_regularizer}
    # Verify Input Shape
    in_filters = inputs.get_shape().as_list()[-1]
    assert in_filters == inputs.get_shape().as_list()[-1], \
        'Module was initialised for a different input shape'
    x      = inputs
    orig_x = x
    # Handle Strided Convolutions
    if np.prod(strides) != 1: orig_x = pool_op(inputs=orig_x, pool_size=strides, strides=strides, padding='valid')

    # Sub-Unit 0
    with tf.variable_scope('sub_unit0'):
        # Adjust Strided Convolution Kernel Size to Prevent Losing Information
        k = [s * 2 if s > 1 else k for k, s in zip(kernel_size, strides)]
        # Layer Definitions for Sub-Unit
        x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
        x = activation(x)
        x = tf.layers.conv3d(inputs=x, filters=out_filters, kernel_size=k, strides=strides,**conv_params)

    # Sub-Unit 1
    with tf.variable_scope('sub_unit1'):
        # Layer Definitions for Sub-Unit
        x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
        x = activation(x)
        x = tf.layers.conv3d(inputs=x, filters=out_filters, kernel_size=kernel_size, strides=(1, 1, 1),**conv_params)

    # Residual Addition
    with tf.variable_scope('sub_unit_add'):
        # Handle Differences in I/O Filter Sizes
        if in_filters < out_filters:
            orig_x = tf.pad(tensor=orig_x, paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + 
                                                    [[int(np.floor((out_filters - in_filters) / 2.)),
                                                    int(np.ceil((out_filters - in_filters) / 2.))]])
        elif in_filters > out_filters:
            orig_x = tf.layers.conv3d(inputs=orig_x, filters=out_filters, kernel_size=kernel_size, strides=(1, 1, 1),**conv_params)
        x += orig_x
    return x

# ------------------------------------------------------------------------------------------------------------------------------------------------------------







# 3D Residual Attention Model for 8x64x64 Patches [Ref: 3D modified adaptation of github.com/qubvel/residual_attention_network] ------------------------------
def resnet_attention_3d(inputs, num_classes=2, p=1, t=2, r=1,
                         kernel_initializer = tf.initializers.variance_scaling(distribution='uniform'), 
                         kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
                         mode               = tf.estimator.ModeKeys.EVAL):
    """
    [1] F. Wang et al.(2017), "Residual Attention Network for Image Classification", IEEE CVPR.
    """
    outputs      = {}
    x            = inputs
    conv_params  = {'padding':           'same',
                    'kernel_initializer': kernel_initializer,
                    'kernel_regularizer': kernel_regularizer,
                    'data_format':       'channels_last'}

    # Preliminary Convolutional Layer
    x = tf.keras.layers.Conv3D(filters=16, kernel_size=(5,5,5), strides=(1,1,1), **conv_params)(x)
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2))(x)                                                             

    # Residual and Attention Blocks
    x = residual_block_3d(x,  input_channels=16,  output_channels=32, stride=(1,2,2), mode=mode)                      
    x = attention_block_3d(x, input_channels=32,  output_channels=64, p=p, t=t, r=r,  mode=mode)                    

    # Visualize Post-Attention Feature Maps
    with tf.device('/cpu:0'):
      disp_filter_num = 0
      x_out           = tf.expand_dims(x[:,:,:,:,disp_filter_num],axis=-1)
      shape           = x_out.get_shape().as_list()
      ydim            = shape[2]
      xdim            = shape[3]
      featuremaps     = shape[4]
      x_out           = tf.slice(x_out,(0,0,0,0,0),(1,1,-1,-1,-1))
      x_out           = tf.reshape(x_out,(ydim,xdim,featuremaps))
      ydim           += 2
      xdim           += 2
      x_out           = tf.image.resize_image_with_crop_or_pad(x_out,ydim,xdim)
      x_out           = tf.reshape(x_out,(ydim,xdim,1,1)) 
      x_out           = tf.transpose(x_out,(2,0,3,1))
      x_out           = tf.reshape(x_out,(1,1*ydim,1*xdim,1))
      tf.summary.image('Post-Attention Feature Maps', x_out, 50)

    # Final Pooling and Fully Connected Layers
    x = tf.keras.layers.GlobalAveragePooling3D()(x)                                                                                                                                                                  
    x = tf.keras.layers.Dense(num_classes,       activation=None, 
                              kernel_initializer = kernel_initializer, 
                              kernel_regularizer = kernel_regularizer)(x)
    
    # Model Outputs [Softmax Class Probabilities('y_prob'), Hard Prediction('y_')]
    outputs['logits'] = x
    outputs['y_prob'] = tf.nn.softmax(x)             
    outputs['y_']     = tf.argmax(x, axis=-1) \
                         if   num_classes > 1 \
                         else tf.cast(tf.greater_equal(x[...,0], 0.5), tf.int32)
    return outputs


# Pre-Activation 3D Residual Block
def residual_block_3d(input_tensor, input_channels=None, output_channels=None, num_classes=10, 
                      kernel_size=(3,3,3), stride=(1,1,1), SE_mode=False, reduction=4, 
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
      input_channels  = output_channels // 4

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

    if (SE_mode==True): x = ChannelSE(reduction=reduction)(x)

    # Residual Addition
    x = tf.keras.layers.Add()([x,input_tensor])
    return x


# 3D Attention Block
def attention_block_3d(input_tensor, input_channels=None, output_channels=None, 
                       encoder_depth=1, p=1,t=2,r=1, SE_mode=False, reduction=4, 
                       kernel_initializer = tf.initializers.variance_scaling(distribution='uniform'), 
                       kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), 
                       mode               = tf.estimator.ModeKeys.EVAL): 
    """
    [1] F. Wang et al.(2017), "Residual Attention Network for Image Classification", IEEE CVPR.
    """
    # Define Target Channel Shapes
    if input_channels is None:
      input_channels  = input_tensor.get_shape()[-1].value
    if output_channels is None:
      output_channels = input_channels

    conv_params  = {'padding':           'same',
                    'kernel_initializer': kernel_initializer,
                    'kernel_regularizer': kernel_regularizer,
                    'data_format':       'channels_last'}

    # First Residual Block & Trunk Branch
    for i in range(p): input_tensor       = residual_block_3d(input_tensor, SE_mode=SE_mode, reduction=reduction, mode=mode)                                
    output_trunk                          = input_tensor                                                   
    for i in range(t): output_trunk       = residual_block_3d(output_trunk, output_channels=output_channels, SE_mode=SE_mode, reduction=reduction, mode=mode)   
    output_soft_mask                      = tf.keras.layers.MaxPool3D(padding='same')(input_tensor)
    for i in range(r): output_soft_mask   = residual_block_3d(output_soft_mask,  SE_mode=SE_mode, reduction=reduction, mode=mode)
  
    # Encoder Stage
    skip_connections = []
    for i in range(encoder_depth - 1):
      output_skip_connection              = residual_block_3d(output_soft_mask, SE_mode=SE_mode, reduction=reduction, mode=mode)                               
      skip_connections.append(output_skip_connection)
      output_soft_mask                    = tf.keras.layers.MaxPool3D(padding='same')(output_soft_mask)
      for _ in range(r): output_soft_mask = residual_block_3d(output_soft_mask, SE_mode=SE_mode, reduction=reduction, mode=mode)

    # Decoder Stage
    skip_connections                      = list(reversed(skip_connections))
    for i in range(encoder_depth-1):
      for _ in range(r): output_soft_mask = residual_block_3d(output_soft_mask, SE_mode=SE_mode, reduction=reduction, mode=mode)
      output_soft_mask                    = tf.keras.layers.UpSampling3D(data_format='channels_last')(output_soft_mask)
      output_soft_mask                    = tf.keras.layers.Add()([output_soft_mask,skip_connections[i]])                

    # Final Upsampling
    for i in range(r): output_soft_mask   = residual_block_3d(output_soft_mask, SE_mode=SE_mode, reduction=reduction, mode=mode)
    output_soft_mask                      = tf.keras.layers.UpSampling3D(data_format='channels_last')(output_soft_mask)

    # Final Convolutional Layers
    output_soft_mask = tf.keras.layers.Conv3D(filters=output_channels, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(output_soft_mask)
    output_soft_mask = tf.keras.layers.Conv3D(filters=output_channels, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(output_soft_mask)
    output_soft_mask = tf.keras.layers.Activation("sigmoid")(output_soft_mask)
    
    # Attention Mechanism: {(1+output_soft_mask)*output_trunk}
    output           = tf.keras.layers.Lambda(lambda x: x+1)(output_soft_mask)
    output           = tf.keras.layers.Multiply()([output, output_trunk])

    # Last Residual Block
    for i in range(p): output = residual_block_3d(output, SE_mode=SE_mode, reduction=reduction, mode=mode)
    return output
# ------------------------------------------------------------------------------------------------------------------------------------------------------------







# 3D Squeeze-Excitation ResNet Architecture for 8x64x64 Patches [Ref: 3D modified adaptation of github.com/qubvel/classification_models] ---------------------
def seresnet_3d(inputs=None,      num_classes=2,    model_config='seresnet', groups=[2,2],     stride_scale=0,
                init_filters=64,  stages=(3,4,6,3), dropout=False,           reduction=[8,8],  strides=(1,1,1), 
                kernel_initializer = tf.initializers.variance_scaling(distribution='uniform'),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), 
                mode               = tf.estimator.ModeKeys.EVAL):
    """
    [1] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    [2] S. Xie et al.(2017), "Aggregated Residual Transformations for Deep Neural Networks", IEEE CVPR.
    """
    # Define Model Configuration (SEResNet50/SEResNeXt50) 
    if   (model_config=='seresnet'):   se_residual_block_3d = seresnet_bottleneck_3d
    elif (model_config=='seresnext'):  se_residual_block_3d = seresnext_bottleneck_3d

    x       = inputs
    outputs = {}

    # Preliminary Convolutional Layer
    x  = tf.keras.layers.Conv3D(filters=init_filters, kernel_size=(5,5,5), strides=(1,1,1), use_bias=False, padding='same', kernel_initializer='he_uniform')(x)
    x  = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
    x0 = tf.keras.layers.Activation('relu')(x)
    x  = tf.keras.layers.MaxPooling3D(pool_size=(1,2,2))(x0)

    # ResNet Core 
    x1 = se_residual_block_3d(filters=init_filters*2, reduction=reduction[0], strides=(1,2,2), groups=groups[0], mode=mode)(x)
    x2 = se_residual_block_3d(filters=init_filters*4, reduction=reduction[1], strides=strides, groups=groups[1], mode=mode)(x1)

    # Final Pooling and Fully Connected Layers
    x = tf.keras.layers.GlobalAveragePooling3D()(x2)
    if (dropout>0): x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(num_classes, activation=None, 
                              kernel_initializer = kernel_initializer, 
                              kernel_regularizer = kernel_regularizer)(x)
    
    # Model Outputs [Softmax Class Probabilities('y_prob'), Hard Prediction('y_')]
    outputs['logits'] = x
    outputs['y_prob'] = tf.nn.softmax(x)             
    outputs['y_']     = tf.argmax(x, axis=-1) \
                         if   num_classes > 1 \
                         else tf.cast(tf.greater_equal(x[...,0], 0.5), tf.int32)
    
    # Map Convolutional Layer Outputs (GradCAM)
    outputs['x0']      = x0
    outputs['x1']      = x1
    outputs['x2']      = x2
    outputs['grad_x0'] = tf.gradients(outputs['y_prob'][0,1], x0)[0]
    outputs['grad_x1'] = tf.gradients(outputs['y_prob'][0,1], x1)[0]
    outputs['grad_x2'] = tf.gradients(outputs['y_prob'][0,1], x2)[0]
    return outputs


# 3D SEResNet BottleNeck Module
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


# 3D SEResNeXt BottleNeck Module
def seresnext_bottleneck_3d(filters, reduction=16, strides=(1,1,1), groups=2, base_width=4, mode=tf.estimator.ModeKeys.EVAL):
    """
    [1] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    [2] S. Xie et al.(2017), "Aggregated Residual Transformations for Deep Neural Networks", IEEE CVPR.
    """
    def layer(input_tensor):     # Define Operations as a Layer
        x        = input_tensor
        residual = input_tensor
        width    = (filters//4) * base_width * groups // 2

        # Bottleneck
        x = tf.keras.layers.Conv3D(filters=width, kernel_size=(1,1,1), strides=(1,1,1), use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = GroupConv3D(filters=width, kernel_size=(1,1,1), strides=strides, groups=groups, kernel_initializer='he_uniform', use_bias=False)(x)
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

# Grouped Convolution Layer implemented as a Slice, Conv3D and Concatenate layers
def GroupConv3D(filters, kernel_size, strides=(1,1,1), groups=2, kernel_initializer='he_uniform', use_bias=False, activation='linear', padding='same'):
    """
    [1] S. Xie et al.(2017), "Aggregated Residual Transformations for Deep Neural Networks", IEEE CVPR.
    """
    def layer(input_tensor):     # Define Operations as a Layer
        inp_ch = int(input_tensor.get_shape()[-1].value // groups)  # Channels Per Group
        out_ch = int(filters // groups)                             # Output Channels
        x      = input_tensor
        blocks = []

        for c in range(groups):
            x = input_tensor[:,:,:,:,(c*inp_ch):((c+1)*inp_ch)]
            x = tf.keras.layers.Conv3D(filters=out_ch,    kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_initializer, 
                                       use_bias=use_bias, activation=activation,   padding=padding)(x)
            blocks.append(x)
        
        x = tf.keras.layers.Concatenate(axis=4)(blocks)
        return x
    return layer

# Squeeze-and-Excitation Block [Ref: 3D modified adaptation of github.com/qubvel/classification_models, github.com/Cadene/pretrained-models.pytorch]
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
# -----------------------------------------------------------------------------------------------------------------------------------------------------------







# 3D Inception-ResNet Architecture for 8x64x64 Patches [Ref: 3D modified adaptation of github.com/qubvel/segmentation_models/backbones] ---------------------
def inception_resnet_3d(inputs=None,  num_classes=2, stem_filters=16, num_IRA_blocks=5, num_IRB_blocks=5, num_IRC_blocks=5,  
                        branch_0_filters    = (16,     40,        16,32),
                        branch_1_filters    = (8,16,   12,24,48,  32,96),
                        branch_2_filters    = (8,8,16,            32,64,96),
                        branch_pool_filters = (16,     40,        32),
                        dropout             = False,  
                        kernel_initializer  = tf.initializers.variance_scaling(distribution='uniform'),
                        kernel_regularizer  = tf.contrib.layers.l2_regularizer(1e-3), 
                        mode                = tf.estimator.ModeKeys.EVAL):
    """
    [1] C. Szegedy et al.(2017), "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning", AAAI.
    [2] C. Szegedy et al.(2016), "Rethinking the Inception Architecture for Computer Vision", IEEE CVPR.
    [3] C. Szegedy et al.(2015), "Going Deeper with Convolutions", IEEE CVPR.    
    """
    x       = inputs
    outputs = {}

    conv_params  = {'padding':           'same',
                    'kernel_initializer': kernel_initializer,
                    'kernel_regularizer': kernel_regularizer,
                    'data_format':       'channels_last'}

    # Stem/Preliminary Convolutional Layer(s) [Spatial Output: 8x32x32, Channels: 16]
    x            = convolutional_block_3d(input_tensor=x, filters=stem_filters, kernel_size=(5,5,5), strides=(1,1,1), conv_params=conv_params, mode=mode)
    x0           = tf.keras.layers.MaxPooling3D(pool_size=(1,2,2))(x)
    x            = x0

    # Inception-A Block (Mixed 5b) [Spatial Output: 8x32x32, Channels: 64]
    branch_0    = convolutional_block_3d(input_tensor=x,        filters=branch_0_filters[0], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=x,        filters=branch_1_filters[0], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=branch_1, filters=branch_1_filters[1], kernel_size=(5,5,5), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=x,        filters=branch_2_filters[0], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=branch_2, filters=branch_2_filters[1], kernel_size=(3,3,3), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=branch_2, filters=branch_2_filters[2], kernel_size=(3,3,3), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_pool = tf.keras.layers.AveragePooling3D(pool_size=(1,3,3), strides=(1,1,1), padding='same')(x)
    branch_pool = convolutional_block_3d(input_tensor=branch_pool, filters=branch_pool_filters[0], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branches    = [branch_0, branch_1, branch_2, branch_pool] 
    x1          = tf.keras.layers.Concatenate(axis=-1, name='mixed_5b')(branches)
    x           = x1

    # 5x Inception-ResNet-A Block (block35) [Spatial Output: 8x32x32, Channels: 64]
    for block_idx in range(num_IRA_blocks):  x = inception_resnet_block_3d(x, scale=0.17, conv_params=conv_params, block_type='block35')


    # Reduction-A Block (Mixed 6a) [Spatial Output: 8x16x16, Channels: 128]
    branch_0    = convolutional_block_3d(input_tensor=x,        filters=branch_0_filters[1], kernel_size=(3,3,3), strides=(1,2,2), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=x,        filters=branch_1_filters[2], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=branch_1, filters=branch_1_filters[3], kernel_size=(3,3,3), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=branch_1, filters=branch_1_filters[4], kernel_size=(3,3,3), strides=(1,2,2), conv_params=conv_params, mode=mode)
    branch_pool = tf.keras.layers.MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2), padding='same')(x)
    branch_pool = convolutional_block_3d(input_tensor=branch_pool, filters=branch_pool_filters[1], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branches    = [branch_0, branch_1, branch_pool] 
    x2          = tf.keras.layers.Concatenate(axis=-1, name='mixed_6a')(branches)
    x           = x2

    # 5x Inception-ResNet-B Block (block17): [Spatial Output: 8x16x16, Channels: 128]
    for block_idx in range(num_IRB_blocks):  x = inception_resnet_block_3d(x, scale=0.10, conv_params=conv_params, block_type='block17')


    # Reduction-B Block (Mixed 7a) [Spatial Output: 4x8x8, Channels: 256]
    branch_0    = convolutional_block_3d(input_tensor=x,        filters=branch_0_filters[2], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_0    = convolutional_block_3d(input_tensor=branch_0, filters=branch_0_filters[3], kernel_size=(3,3,3), strides=(2,2,2), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=x,        filters=branch_1_filters[5], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=branch_1, filters=branch_1_filters[6], kernel_size=(3,3,3), strides=(2,2,2), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=x,        filters=branch_2_filters[3], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=branch_2, filters=branch_2_filters[4], kernel_size=(3,3,3), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=branch_2, filters=branch_2_filters[5], kernel_size=(3,3,3), strides=(2,2,2), conv_params=conv_params, mode=mode)
    branch_pool = tf.keras.layers.MaxPooling3D(pool_size=(2,3,3), strides=(2,2,2), padding='same')(x)
    branch_pool = convolutional_block_3d(input_tensor=branch_pool, filters=branch_pool_filters[2], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branches    = [branch_0, branch_1, branch_2, branch_pool] 
    x3          = tf.keras.layers.Concatenate(axis=-1, name='mixed_7a')(branches)
    x           = x3

    # 5x Inception-ResNet-C Block (block8): [Spatial Output: 4x8x8, Channels: 256]
    for block_idx in range(num_IRC_blocks):  x = inception_resnet_block_3d(x, scale=0.20, conv_params=conv_params, block_type='block8')

    # Final Pooling and Fully Connected Layers
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    if (dropout>0): x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(num_classes, activation=None, 
                              kernel_initializer = kernel_initializer, 
                              kernel_regularizer = kernel_regularizer)(x)
    
    # Model Outputs [Softmax Class Probabilities('y_prob'), Hard Prediction('y_')]
    outputs['logits'] = x
    outputs['y_prob'] = tf.nn.softmax(x)             
    outputs['y_']     = tf.argmax(x, axis=-1) \
                         if   num_classes > 1 \
                         else tf.cast(tf.greater_equal(x[...,0], 0.5), tf.int32)

    # Model Summary
    print('Input Volume:---------------------------', inputs.get_shape())
    print('Stem Layer:-----------------------------', x0.get_shape())
    print('Inception-A Block (Mixed 5b)------------', x1.get_shape())
    print('Reduction-A Block (Mixed 6a)------------', x2.get_shape())
    print('Reduction-B block (Mixed 7a)------------', x3.get_shape())
    print('Final Convolutional Layer [Logits]:-----', x.get_shape())

    return outputs


# 3D Inception-ResNet Block
def inception_resnet_block_3d(input_tensor, scale, block_type, conv_params, mode=tf.estimator.ModeKeys.EVAL):
    
    if   (block_type=='block35'):
        branch_0 = convolutional_block_3d(input_tensor=input_tensor, filters=16,  kernel_size=(1,1,1), conv_params=conv_params, mode=mode)
        branch_1 = convolutional_block_3d(input_tensor=input_tensor, filters=8,   kernel_size=(1,1,1), conv_params=conv_params, mode=mode)
        branch_1 = convolutional_block_3d(input_tensor=branch_1,     filters=16,  kernel_size=(3,3,3), conv_params=conv_params, mode=mode)
        branch_2 = convolutional_block_3d(input_tensor=input_tensor, filters=8,   kernel_size=(1,1,1), conv_params=conv_params, mode=mode)
        branch_2 = convolutional_block_3d(input_tensor=branch_2,     filters=16,  kernel_size=(3,3,3), conv_params=conv_params, mode=mode)
        branch_2 = convolutional_block_3d(input_tensor=branch_2,     filters=32,  kernel_size=(3,3,3), conv_params=conv_params, mode=mode)
        branches = [branch_0, branch_1, branch_2]
    elif (block_type=='block17'):
        branch_0 = convolutional_block_3d(input_tensor=input_tensor, filters=64,  kernel_size=(1,1,1), conv_params=conv_params, mode=mode)
        branch_1 = convolutional_block_3d(input_tensor=input_tensor, filters=16,  kernel_size=(1,1,1), conv_params=conv_params, mode=mode)
        branch_1 = convolutional_block_3d(input_tensor=branch_1,     filters=32,  kernel_size=(1,1,7), conv_params=conv_params, mode=mode)
        branch_1 = convolutional_block_3d(input_tensor=branch_1,     filters=64,  kernel_size=(1,7,1), conv_params=conv_params, mode=mode)
        branches = [branch_0, branch_1]
    elif (block_type=='block8'):
        branch_0 = convolutional_block_3d(input_tensor=input_tensor, filters=128, kernel_size=(1,1,1), conv_params=conv_params, mode=mode)
        branch_1 = convolutional_block_3d(input_tensor=input_tensor, filters=32,  kernel_size=(1,1,1), conv_params=conv_params, mode=mode)
        branch_1 = convolutional_block_3d(input_tensor=branch_1,     filters=64,  kernel_size=(1,1,3), conv_params=conv_params, mode=mode)
        branch_1 = convolutional_block_3d(input_tensor=branch_1,     filters=128, kernel_size=(1,3,1), conv_params=conv_params, mode=mode)
        branches = [branch_0, branch_1]

    mixed = tf.keras.layers.Concatenate(axis=-1, name='mixed')(branches)
    up    = convolutional_block_3d(input_tensor=mixed, filters=input_tensor.get_shape()[-1], kernel_size=(1,1,1), conv_params=conv_params, mode=mode)
    x     = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=input_tensor.get_shape()[1:], 
                                                                                        arguments   ={'scale': scale})([input_tensor, up])
    x     = tf.keras.layers.Activation('relu')(x)

    return x


# 3D Convolutional Block
def convolutional_block_3d(input_tensor, filters, kernel_size, strides=(1,1,1), conv_params=None, mode=tf.estimator.ModeKeys.EVAL):

    if (conv_params==None):
      conv_params  = {'padding':           'same',
                      'kernel_initializer': tf.initializers.variance_scaling(distribution='uniform'),
                      'kernel_regularizer': tf.contrib.layers.l2_regularizer(1e-3),
                      'data_format':       'channels_last'}

    x = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=False, **conv_params)(input_tensor)
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x
# ------------------------------------------------------------------------------------------------------------------------------------------------------------







# 3D Inception-ResNet Architecture for 8x64x64 Patches [Ref: 3D modified adaptation of github.com/qubvel/segmentation_models/backbones] ---------------------
def se_inception_resnet_3d(inputs=None,  num_classes=2, stem_filters=16, num_IRA_blocks=5, num_IRB_blocks=5, num_IRC_blocks=5,  
                           branch_0_filters    = (16,     40,        16,32),
                           branch_1_filters    = (8,16,   12,24,48,  32,96),
                           branch_2_filters    = (8,8,16,            32,64,96),
                           branch_pool_filters = (16,     40,        32),
                           dropout             = False,  
                           reduction           = [8,8,8],
                           kernel_initializer  = tf.initializers.variance_scaling(distribution='uniform'),
                           kernel_regularizer  = tf.contrib.layers.l2_regularizer(1e-3), 
                           mode                = tf.estimator.ModeKeys.EVAL):
    """
    [1] C. Szegedy et al.(2017), "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning", AAAI.
    [2] C. Szegedy et al.(2016), "Rethinking the Inception Architecture for Computer Vision", IEEE CVPR.
    [3] C. Szegedy et al.(2015), "Going Deeper with Convolutions", IEEE CVPR.    
    [4] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    """
    x       = inputs
    outputs = {}

    conv_params  = {'padding':           'same',
                    'kernel_initializer': kernel_initializer,
                    'kernel_regularizer': kernel_regularizer,
                    'data_format':       'channels_last'}

    # Stem/Preliminary Convolutional Layer(s) [Spatial Output: 8x32x32, Channels: 16]
    x           = convolutional_block_3d(input_tensor=x, filters=stem_filters, kernel_size=(5,5,5), strides=(1,1,1), conv_params=conv_params, mode=mode)
    x0          = tf.keras.layers.MaxPooling3D(pool_size=(1,2,2))(x)
    x           = x0

    # Inception-A Block (Mixed 5b) [Spatial Output: 8x32x32, Channels: 64]
    branch_0    = convolutional_block_3d(input_tensor=x,        filters=branch_0_filters[0], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=x,        filters=branch_1_filters[0], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=branch_1, filters=branch_1_filters[1], kernel_size=(5,5,5), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=x,        filters=branch_2_filters[0], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=branch_2, filters=branch_2_filters[1], kernel_size=(3,3,3), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=branch_2, filters=branch_2_filters[2], kernel_size=(3,3,3), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_pool = tf.keras.layers.AveragePooling3D(pool_size=(1,3,3), strides=(1,1,1), padding='same')(x)
    branch_pool = convolutional_block_3d(input_tensor=branch_pool, filters=branch_pool_filters[0], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branches    = [branch_0, branch_1, branch_2, branch_pool] 
    x1          = tf.keras.layers.Concatenate(axis=-1, name='mixed_5b')(branches)
    x           = x1

    # 5x Inception-ResNet-A Block (block35) [Spatial Output: 8x32x32, Channels: 64]
    for block_idx in range(num_IRA_blocks):  x = inception_resnet_block_3d(x, scale=0.17, conv_params=conv_params, block_type='block35')
    # Squeeze-and-Excitation Channel-Wise Attention
    x           = ChannelSE(reduction=reduction[0])(x)

    # Reduction-A Block (Mixed 6a) [Spatial Output: 8x16x16, Channels: 128]
    branch_0    = convolutional_block_3d(input_tensor=x,        filters=branch_0_filters[1], kernel_size=(3,3,3), strides=(1,2,2), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=x,        filters=branch_1_filters[2], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=branch_1, filters=branch_1_filters[3], kernel_size=(3,3,3), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=branch_1, filters=branch_1_filters[4], kernel_size=(3,3,3), strides=(1,2,2), conv_params=conv_params, mode=mode)
    branch_pool = tf.keras.layers.MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2), padding='same')(x)
    branch_pool = convolutional_block_3d(input_tensor=branch_pool, filters=branch_pool_filters[1], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branches    = [branch_0, branch_1, branch_pool] 
    x2          = tf.keras.layers.Concatenate(axis=-1, name='mixed_6a')(branches)
    x           = x2

    # 5x Inception-ResNet-B Block (block17): [Spatial Output: 8x16x16, Channels: 128]
    for block_idx in range(num_IRB_blocks):  x = inception_resnet_block_3d(x, scale=0.10, conv_params=conv_params, block_type='block17')
    # Squeeze-and-Excitation Channel-Wise Attention
    x           = ChannelSE(reduction=reduction[1])(x)

    # Reduction-B Block (Mixed 7a) [Spatial Output: 4x8x8, Channels: 256]
    branch_0    = convolutional_block_3d(input_tensor=x,        filters=branch_0_filters[2], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_0    = convolutional_block_3d(input_tensor=branch_0, filters=branch_0_filters[3], kernel_size=(3,3,3), strides=(2,2,2), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=x,        filters=branch_1_filters[5], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_1    = convolutional_block_3d(input_tensor=branch_1, filters=branch_1_filters[6], kernel_size=(3,3,3), strides=(2,2,2), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=x,        filters=branch_2_filters[3], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=branch_2, filters=branch_2_filters[4], kernel_size=(3,3,3), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branch_2    = convolutional_block_3d(input_tensor=branch_2, filters=branch_2_filters[5], kernel_size=(3,3,3), strides=(2,2,2), conv_params=conv_params, mode=mode)
    branch_pool = tf.keras.layers.MaxPooling3D(pool_size=(2,3,3), strides=(2,2,2), padding='same')(x)
    branch_pool = convolutional_block_3d(input_tensor=branch_pool, filters=branch_pool_filters[2], kernel_size=(1,1,1), strides=(1,1,1), conv_params=conv_params, mode=mode)
    branches    = [branch_0, branch_1, branch_2, branch_pool] 
    x3          = tf.keras.layers.Concatenate(axis=-1, name='mixed_7a')(branches)
    x           = x3

    # 5x Inception-ResNet-C Block (block8): [Spatial Output: 4x8x8, Channels: 256]
    for block_idx in range(num_IRC_blocks):  x = inception_resnet_block_3d(x, scale=0.20, conv_params=conv_params, block_type='block8')
    # Squeeze-and-Excitation Channel-Wise Attention
    x           = ChannelSE(reduction=reduction[2])(x)

    # Final Pooling and Fully Connected Layers
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    if (dropout>0): x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(num_classes, activation=None, 
                              kernel_initializer = kernel_initializer, 
                              kernel_regularizer = kernel_regularizer)(x)
    
    # Model Outputs [Softmax Class Probabilities('y_prob'), Hard Prediction('y_')]
    outputs['logits'] = x
    outputs['y_prob'] = tf.nn.softmax(x)             
    outputs['y_']     = tf.argmax(x, axis=-1) \
                         if   num_classes > 1 \
                         else tf.cast(tf.greater_equal(x[...,0], 0.5), tf.int32)

    # Model Summary
    print('Input Volume:---------------------------', inputs.get_shape())
    print('Stem Layer:-----------------------------', x0.get_shape())
    print('Inception-A Block (Mixed 5b)------------', x1.get_shape())
    print('Reduction-A Block (Mixed 6a)------------', x2.get_shape())
    print('Reduction-B block (Mixed 7a)------------', x3.get_shape())
    print('Final Convolutional Layer [Logits]:-----', x.get_shape())

    return outputs
# ------------------------------------------------------------------------------------------------------------------------------------------------------------

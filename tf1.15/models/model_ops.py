from __future__ import unicode_literals, absolute_import, print_function, division
import tensorflow as tf
import numpy as np


'''
Prostate Cancer Detection in bpMRI
Script:         Model Ops
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         06/05/2020

'''

# Upsample and Concatenate with Skip-Connection [Ref:DLTK]
def upsample_and_concat(input1, input2, strides=(2,2,2)):
    assert len(input1.get_shape().as_list()) == 5, \
        'ERROR: Inputs Must Have A Rank Of 5.'
    assert len(input1.get_shape().as_list()) == len(input2.get_shape().as_list()), \
        'ERROR: Tensors To-Be-Concatenated Have Different Ranks.'
    
    # Linearly Upsample Input 1 via Transposed Convolution
    input1 = linear_upsample_3d(input1, shape_as=input2, strides=strides)
    # Concatenate Input 1 with 2 (Skip-Connection)
    return tf.concat(axis=-1, values=[input2, input1])


# Linearly Upsample via Transposed Convolution
def linear_upsample_3d(inputs, shape_as, strides=(2,2,2),
                       use_bias=False, trainable=False):
    static_inp_shape = tuple(inputs.get_shape().as_list()) 
    dyn_inp_shape    = tf.shape(inputs)                     
    rank             = len(static_inp_shape)
    num_filters      = static_inp_shape[-1]
    strides_5d       = [1, ] + list(strides) + [1, ]
    kernel_size      = [2 * s if s > 1 else 1 for s in strides]

    kernel = get_linear_upsampling_kernel(kernel_spatial_shape = kernel_size,
                                          out_filters          = num_filters,
                                          in_filters           = num_filters,
                                          trainable            = trainable)
    dyn_out_shape        = [dyn_inp_shape[i] * strides_5d[i] for i in range(rank)]
    dyn_out_shape[-1]    = num_filters
    static_out_shape     = [static_inp_shape[i] * strides_5d[i]
                            if isinstance(static_inp_shape[i], int)
                            else None for i in range(rank)]
    static_out_shape[-1] = num_filters

    # For Odd-Numbered Z Dimension [Static: Compt. Graph; Dynamic: Actual O/P]
    if ((static_inp_shape[1]%2)!=0): 
      static_out_shape[1] = (tuple(shape_as.get_shape().as_list()))[1]
      dyn_out_shape[1]    = (tuple(shape_as.get_shape().as_list()))[1]

    upsampled = tf.nn.conv3d_transpose(value=inputs, filter=kernel,
                                       output_shape = dyn_out_shape,
                                       strides      = strides_5d,
                                       padding      = "SAME")
    upsampled.set_shape(static_out_shape)
    return upsampled


# Generate Upsampling Kernel Shapes/Weights
def get_linear_upsampling_kernel(kernel_spatial_shape,
                                 out_filters,
                                 in_filters,
                                 trainable=False):
    rank = len(list(kernel_spatial_shape))
    assert 1 < rank < 4, \
        'ERROR: Transposed Convolutions are Only Supported in 2D and 3D.'

    kernel_shape =  tuple(kernel_spatial_shape + [out_filters, in_filters])
    size         =  kernel_spatial_shape
    factor       = (np.array(size) + 1) // 2
    center       =  np.zeros_like(factor, np.float)

    for i in range(len(factor)):
        if (size[i]%2)==1: center[i] = factor[i] - 1
        else:              center[i] = factor[i] - 0.5

    weights = np.zeros(kernel_shape)
    if rank == 2:
        og     =  np.ogrid[:size[0], :size[1]]
        x_filt = (1 - abs(og[0] - center[0]) / np.float(factor[0]))
        y_filt = (1 - abs(og[1] - center[1]) / np.float(factor[1]))
        filt   = x_filt * y_filt

        for i in range(out_filters): weights[:,:,i,i] = filt
    
    else:
        og     =  np.ogrid[:size[0], :size[1], :size[2]]
        x_filt = (1 - abs(og[0] - center[0]) / np.float(factor[0]))
        y_filt = (1 - abs(og[1] - center[1]) / np.float(factor[1]))
        z_filt = (1 - abs(og[2] - center[2]) / np.float(factor[2]))
        filt   = x_filt * y_filt * z_filt

        for i in range(out_filters): weights[:,:,:,i,i] = filt

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return tf.get_variable(name        = "linear_up_kernel",
                           initializer = init,
                           shape       = weights.shape,
                           trainable   = trainable)

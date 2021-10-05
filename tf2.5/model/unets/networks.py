import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from .modelio import LoadableModel, store_config_args

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


# 3D U-Net Wrapper for Configs ----------------------------------------------------------------------------------------------------------------------------------------
class M1(LoadableModel):
    '''
    [1] Z. Zhou et al. (2019), "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", IEEE TMI.
    [2] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    [3] S. Kohl et al. (2018), "A Probabilistic U-Net for Segmentation of Ambiguous Images", NeurIPS.
    [4] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    [5] K. He et al.(2016), "Deep Residual Learning for Image Recognition", IEEE CVPR.
    [6] K. He et al.(2016), "Identity Mappings in Deep Residual Networks", ECCV.
    '''
    @store_config_args
    def __init__(self,
                 input_spatial_dims,
                 input_channels,
                 num_classes,
                 dropout_rate       =   0.50,  
                 dropout_mode       =  'standard',    
                 filters            =  (32,64,128,256,512),           
                 strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2)),           
                 kernel_sizes       = ((1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3)),
                 se_reduction       =  (8,8,8,8,8),         
                 att_sub_samp       = ((1,1,1),(1,1,1),(1,1,1),(1,1,1)),      
                 kernel_initializer =   tf.keras.initializers.Orthogonal(gain=1.0, seed=8),
                 bias_initializer   =   tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=8),
                 kernel_regularizer =   tf.keras.regularizers.l2(1e-4),
                 bias_regularizer   =   tf.keras.regularizers.l2(1e-4),         
                 cascaded           =   False,
                 probabilistic      =   False,
                 deep_supervision   =   False,
                 proba_event_shape  =   256,
                 name               =  'UNET-TYPE-M1'):

        # Ensure Correct Dimensionality
        ndims = len(input_spatial_dims)
        assert ndims in [1,2,3], 'Variable (ndims) should be  1, 2 or 3. Found: %d.'%ndims
        
        # Standard Single-Stage Model
        if (cascaded==False):
            # Input Layer Definition
            image = tf.keras.Input(shape=(*input_spatial_dims, input_channels), name='image')
            
            # Dual-Attention U-Net Model Definition
            m1_model    = m1(inputs             = image, 
                             num_classes        = num_classes, 
                             dropout_mode       = dropout_mode,
                             dropout_rate       = dropout_rate,          
                             filters            = filters,            
                             strides            = strides,
                             kernel_sizes       = kernel_sizes,            
                             se_reduction       = se_reduction,          
                             att_sub_samp       = att_sub_samp,             
                             kernel_initializer = kernel_initializer,    
                             bias_initializer   = bias_initializer,     
                             kernel_regularizer = kernel_regularizer,    
                             bias_regularizer   = bias_regularizer,
                             probabilistic      = probabilistic,
                             proba_event_shape  = proba_event_shape)
            
            # For Probabilisitic Outputs
            if probabilistic:
                # Define Model I/O Tensors
                super().__init__(name=name, inputs=[image], outputs=[tf.keras.layers.Lambda(lambda x: x, name='detection')(m1_model['prob_softmax']), 
                                                                     tf.keras.layers.Lambda(lambda x: x, name='KL')(m1_model['prob_kl'])])

                # Cache Pointers to Probabilistic Layers/Tensors for Future Reference
                self.references               = LoadableModel.ReferenceContainer()
                self.references.infer_softmax = tf.keras.activations.softmax(m1_model['prob_infer_conv'])
                self.references.kl            = m1_model['prob_kl']
            
            else:
                # Define Model I/O Tensors                
                super().__init__(name=name, inputs=[image], outputs=[tf.keras.layers.Lambda(lambda x: x, name='detection')(m1_model['y_softmax'])])
                self.references               = LoadableModel.ReferenceContainer()

            # Cache Pointers to Common Layers/Tensors for Future Reference
            self.references.cascaded       = cascaded
            self.references.probabilistic  = probabilistic
            self.references.m1_model       = m1_model

        # Cascaded Two-Stage Model
        else:
            # Input Layers Definition
            image_v1 = tf.keras.Input(shape=(*input_spatial_dims, input_channels), name='image_1')
            image_v2 = tf.keras.Input(shape=(*input_spatial_dims, input_channels), name='image_2')
            
            # First-Stage Dual-Attention U-Net Model Definition
            m1_stage1   = m1(inputs             = image_v1, 
                             num_classes        = num_classes, 
                             dropout_mode       = dropout_mode,
                             dropout_rate       = dropout_rate,          
                             filters            = filters,            
                             strides            = strides,
                             kernel_sizes       = kernel_sizes,            
                             se_reduction       = se_reduction,          
                             att_sub_samp       = att_sub_samp,                   
                             kernel_initializer = kernel_initializer,    
                             bias_initializer   = bias_initializer,     
                             kernel_regularizer = kernel_regularizer,    
                             bias_regularizer   = bias_regularizer,
                             probabilistic      = probabilistic,
                             proba_event_shape  = proba_event_shape)

            # Second-Stage Dual-Attention U-Net Model Definition
            m1_stage2   = m1(inputs             = tf.keras.layers.concatenate([(m1_stage1['prob_softmax'][:,:,:,:,:num_classes-1] \
                                                  if probabilistic else m1_stage1['y_softmax'][:,:,:,:,:num_classes-1]), image_v2], axis=-1), 
                             num_classes        = num_classes, 
                             dropout_rate       = dropout_rate, 
                             dropout_mode       = dropout_mode,         
                             filters            = filters,            
                             strides            = strides,  
                             kernel_sizes       = kernel_sizes,          
                             se_reduction       = se_reduction,          
                             att_sub_samp       = att_sub_samp,               
                             kernel_initializer = kernel_initializer,    
                             bias_initializer   = bias_initializer,     
                             kernel_regularizer = kernel_regularizer,    
                             bias_regularizer   = bias_regularizer,
                             probabilistic      = probabilistic,
                             proba_event_shape  = proba_event_shape)

            # Coupled Inference via Specified Aggregation Strategy
            prior_pred_train, joint_pred_train     = self.decision_fusion(prior_softmax     = m1_stage1['prob_softmax'][:,:,:,:,num_classes-1] if probabilistic \
                                                                                              else m1_stage1['y_softmax'][:,:,:,:,num_classes-1],
                                                                          follow_up_softmax = m1_stage2['prob_softmax'][:,:,:,:,num_classes-1] if probabilistic \
                                                                                              else m1_stage2['y_softmax'][:,:,:,:,num_classes-1],
                                                                          strategy          = cascaded)
            if probabilistic:
                prior_pred_infer, joint_pred_infer = self.decision_fusion(prior_softmax     = tf.keras.activations.softmax(m1_stage1['prob_infer_conv'])[:,:,:,:,num_classes-1],
                                                                          follow_up_softmax = tf.keras.activations.softmax(m1_stage2['prob_infer_conv'])[:,:,:,:,num_classes-1],
                                                                          strategy          = cascaded)
                # Define Model I/O Tensors
                super().__init__(name=name, inputs=[image_v1,image_v2], outputs=[tf.keras.layers.Lambda(lambda x: x, name='detection_1')([prior_pred_train]),
                                                                                 tf.keras.layers.Lambda(lambda x: x, name='detection_2')([joint_pred_train]),
                                                                                 tf.keras.layers.Lambda(lambda x: x, name='KL_1')(m1_stage1['prob_kl']), 
                                                                                 tf.keras.layers.Lambda(lambda x: x, name='KL_2')(m1_stage2['prob_kl'])])
                # Cache Pointers to Probabilistic Layers/Tensors for Future Reference            
                self.references                 = LoadableModel.ReferenceContainer()
                self.references.infer_softmax_1 = tf.keras.activations.softmax(m1_stage1['prob_infer_conv'])
                self.references.infer_softmax_2 = tf.keras.activations.softmax(m1_stage2['prob_infer_conv'])
                self.references.kl_1            = m1_stage1['prob_kl']
                self.references.kl_2            = m1_stage2['prob_kl']

            else:
                # Define Model I/O Tensors
                super().__init__(name=name, inputs=[image_v1,image_v2], outputs=[tf.keras.layers.Lambda(lambda x: x, name='detection_1')([prior_pred_train]),
                                                                                 tf.keras.layers.Lambda(lambda x: x, name='detection_2')([joint_pred_train])])
                self.references                 = LoadableModel.ReferenceContainer()

            print('Cascade Prior Prediction (Softmax):-----', joint_pred_train.shape)
            print('Cascade Follow-Up Prediction (Softmax):-', joint_pred_train.shape)

            # Cache Pointers to Common Layers/Tensors for Future Reference
            self.references.m1_stage1     = m1_stage1
            self.references.m1_stage2     = m1_stage2
            self.references.cascaded      = cascaded
            self.references.probabilistic = probabilistic

    # Returns Reconfigured Model to Predict Segment Probabilities Only
    def get_detect_model(self):
        if (self.references.cascaded!=False):
            if self.references.probabilistic: 
                return tf.keras.Model(self.inputs, [self.references.infer_softmax_1,\
                                                    self.references.infer_softmax_2])
            else:                             
                return tf.keras.Model(self.inputs, [self.references.m1_stage1['y_softmax'][:,:,:,:,:num_classes],\
                                                    self.references.m1_stage2['y_softmax'][:,:,:,:,:num_classes]])
        else:
            if self.references.probabilistic: return tf.keras.Model(self.inputs,  self.references.infer_softmax)
            else:                             return tf.keras.Model(self.inputs,  self.references.m1_model['y_softmax'][:,:,:,:,:num_classes])

    # Decision Fusion for Two-Stage Cascaded Network
    def decision_fusion(self, prior_softmax, follow_up_softmax, strategy='identity'):
    
        # Possible Aggregation Strategies
        if   (strategy=='identity'): joint_pred = tf.expand_dims((follow_up_softmax), axis=-1)
        elif (strategy=='noisy-or'): joint_pred = tf.expand_dims((1-((1-prior_softmax)*(1-follow_up_softmax))), axis=-1)
        elif (strategy=='bayes'):
            joint_pred = tf.expand_dims((((prior_softmax*follow_up_softmax)+1e-9) \
                                       / ((prior_softmax*follow_up_softmax)+1e-9 + ((1-prior_softmax)*(1-follow_up_softmax)))), axis=-1)
        
        # Output Prediction (+ Stage 1 Prediction for Deep Supervision)
        prior_pred = tf.keras.layers.concatenate((tf.expand_dims((1-prior_softmax), axis=-1),
                                                  tf.expand_dims((prior_softmax),   axis=-1)))
        joint_pred = tf.keras.layers.concatenate((1-joint_pred,joint_pred))
    
        return prior_pred, joint_pred
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------































# 3D SEResNet BottleNeck Module -------------------------------------------------------------------------------------------------------------------------------------
def SEResNetBottleNeck(filters, kernel_size, conv_params, reduction=16, strides=(1,1,1)):
    """
    [1] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    """
    def layer(input_tensor):     # Define Operations as a Layer
        x        = input_tensor
        residual = input_tensor

        # Bottleneck
        x = tf.keras.layers.Conv3D(filters=filters//4, kernel_size=kernel_size, strides=strides, **conv_params)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Conv3D(filters=filters//4, kernel_size=(3,3,3), strides=(1,1,1), **conv_params)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Conv3D(filters=filters, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(x)
        x = tfa.layers.InstanceNormalization()(x)

        # Replicate Operations with Residual Connection (change in #num_filters or spatial_dims)
        x_channels = x.get_shape()[-1]
        r_channels = residual.get_shape()[-1]
        if (strides!=1)|(x_channels!=r_channels):
            residual = tf.keras.layers.Conv3D(filters=x_channels, kernel_size=kernel_size, strides=strides, **conv_params)(residual)
            residual = tfa.layers.InstanceNormalization()(residual)

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
    W_y         = tfa.layers.InstanceNormalization()(W_y)

    return W_y, sigm_psi_f
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


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
                              kernel_initializer =   tf.keras.initializers.Orthogonal(gain=1.0, seed=8),
                              bias_initializer   =   tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=8),
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
class Conv1x1x1withLatentDist(tf.keras.Model):
    """
    [1] S. Kohl et al. (2018), "A Probabilistic U-Net for Segmentation of Ambiguous Images", NeurIPS.
    """        
    def __init__(self,
                 num_classes        =  2, 
                 num_channels       =  256,
                 kernel_initializer =  tf.keras.initializers.Orthogonal(gain=1.0, seed=8),
                 bias_initializer   =  tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=8),
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
        self.conv1 = tf.keras.layers.Conv3D(filters=self.num_channels//4,  kernel_size=(1,3,3), strides=(1,1,1), **conv_params)
        self.norm1 = tfa.layers.InstanceNormalization()
        self.conv2 = tf.keras.layers.Conv3D(filters=self.num_channels//16, kernel_size=(1,3,3), strides=(1,1,1), **conv_params)
        self.norm2 = tfa.layers.InstanceNormalization()
        self.conv3 = tf.keras.layers.Conv3D(filters=self.num_classes,      kernel_size=(1,1,1), strides=(1,1,1), **conv_params, 
                                            name=self.logits_tensor_name)

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.norm1(x, training=training)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        return self.conv3(x)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# (Probabilistic) 3D U-Net ----------------------------------------------------------------------------------------------------------------------------------------------
def m1(inputs, num_classes, 
       dropout_mode       =  'standard',
       dropout_rate       =   0.50,
       filters            =  (32,64,128,256,512), 
       strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2)),           
       kernel_sizes       = ((1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3)),
       se_reduction       =  (8,8,8,8,8),
       att_sub_samp       = ((1,1,1),(1,1,1),(1,1,1),(1,1,1)),
       kernel_initializer =   tf.keras.initializers.Orthogonal(gain=1.0, seed=8),
       bias_initializer   =   tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=8),
       kernel_regularizer =   tf.keras.regularizers.l2(1e-4),
       bias_regularizer   =   tf.keras.regularizers.l2(1e-4),
       deep_supervision   =   False,
       probabilistic      =   False,
       proba_event_shape  =   256,
       summary            =   True):
    """
    [1] Z. Zhou et al. (2019), "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", IEEE TMI.
    [2] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    [3] S. Kohl et al. (2018), "A Probabilistic U-Net for Segmentation of Ambiguous Images", NeurIPS.
    [4] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    [5] K. He et al.(2016), "Deep Residual Learning for Image Recognition", IEEE CVPR.
    [6] K. He et al.(2016), "Identity Mappings in Deep Residual Networks", ECCV.
    
    U-Net Schematic:
        Resol. 1  (x)------------->(att_conv0)-->(deconv2_up2)-->(deconv1_up1)-->(deconv0)-->(uconv0)-->(y__)
        Resol. 2   |---->(conv1)-->(att_conv1)-->(deconv3_up2)-->(deconv2_up1)-->(deconv1)-->(uconv1)
        Resol. 3            |----->(conv2)------>(att_conv2)---->(deconv3_up1)-->(deconv2)-->(uconv2)
        Resol. 4                      |--------->(conv3)-------->(att_conv3)---->(deconv3)-->(uconv3)
        Resol. 5                                    |----------->(convm)-------------|
    """
    # Preamble
    if probabilistic: x_ =  inputs[:,:,:,:,:-(num_classes-1)]
    else:             x_ =  inputs
    outputs              = {}
    conv_params          = {'padding':           'same',
                            'kernel_initializer': kernel_initializer,
                            'bias_initializer':   bias_initializer,
                            'kernel_regularizer': kernel_regularizer,
                            'bias_regularizer':   bias_regularizer}

    if   (dropout_mode=='standard'):     DropoutFunc = tf.keras.layers.Dropout
    elif (dropout_mode=='monte-carlo'):  DropoutFunc = MonteCarloDropout

    assert len(filters)==5, "ERROR: Expected Tuple/Array with 5 Values (One Per Resolution)."
    assert len(se_reduction)==5, "ERROR: Expected Tuple/Array with 5 Values (One Per Resolution)."
    assert [len(a) for a in att_sub_samp]==[3,3,3,3], "ERROR: Expected 4x3 Tuple/Array (3D Sub-Sampling Factors for 4 Attention Gates)."
    assert [len(s) for s in strides]==[3,3,3,3,3], "ERROR: Expected 5x3 Tuple/Array (3D Strides for 5 Resolutions)."
    assert [len(k) for k in kernel_sizes]==[3,3,3,3,3], "ERROR: Expected 5x3 Tuple/Array (3D Kernels for 5 Resolutions)."


    # Preliminary Convolutional Layer
    x     = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=kernel_sizes[0], strides=strides[0], **conv_params)(x_)
    x     = tfa.layers.InstanceNormalization()(x)
    x     = tf.keras.layers.LeakyReLU(alpha=0.1)(x)  

    # Encoder: Backbone SE-Residual Blocks for Feature Extraction
    conv1 = SEResNetBottleNeck(filters=filters[1], kernel_size=kernel_sizes[1], strides=strides[1], reduction=se_reduction[1], conv_params=conv_params)(x)  
    conv1 = DropoutFunc(dropout_rate)(conv1)
    conv2 = SEResNetBottleNeck(filters=filters[2], kernel_size=kernel_sizes[2], strides=strides[2], reduction=se_reduction[2], conv_params=conv_params)(conv1)  
    conv2 = DropoutFunc(dropout_rate)(conv2)
    conv3 = SEResNetBottleNeck(filters=filters[3], kernel_size=kernel_sizes[3], strides=strides[3], reduction=se_reduction[3], conv_params=conv_params)(conv2)  
    conv3 = DropoutFunc(dropout_rate)(conv3)
    convm = SEResNetBottleNeck(filters=filters[4], kernel_size=kernel_sizes[4], strides=strides[4], reduction=se_reduction[4], conv_params=conv_params)(conv3)
    convm = DropoutFunc(dropout_rate)(convm)

    # Grid-Based Attention Gating
    att_conv0,att_0 = GridAttentionBlock3D(conv_tensor=x,     gating_tensor=convm, inter_channels=filters[0], sub_samp=att_sub_samp[0], conv_params=conv_params)
    att_conv1,att_1 = GridAttentionBlock3D(conv_tensor=conv1, gating_tensor=convm, inter_channels=filters[1], sub_samp=att_sub_samp[1], conv_params=conv_params)
    att_conv2,att_2 = GridAttentionBlock3D(conv_tensor=conv2, gating_tensor=convm, inter_channels=filters[2], sub_samp=att_sub_samp[2], conv_params=conv_params)
    att_conv3,att_3 = GridAttentionBlock3D(conv_tensor=conv3, gating_tensor=convm, inter_channels=filters[3], sub_samp=att_sub_samp[3], conv_params=conv_params)

    # Decoder: Nested U-Net - Stage 3
    deconv3     = tf.keras.layers.Conv3DTranspose(filters=filters[3], kernel_size=kernel_sizes[4], strides=strides[4], padding="same")(convm)
    deconv3_up1 = tf.keras.layers.Conv3DTranspose(filters=filters[2], kernel_size=kernel_sizes[3], strides=strides[3], padding="same")(deconv3)
    deconv3_up2 = tf.keras.layers.Conv3DTranspose(filters=filters[1], kernel_size=kernel_sizes[2], strides=strides[2], padding="same")(deconv3_up1)
    deconv3_up3 = tf.keras.layers.Conv3DTranspose(filters=filters[0], kernel_size=kernel_sizes[1], strides=strides[1], padding="same")(deconv3_up2)
    uconv3      = tf.keras.layers.concatenate([deconv3, att_conv3])    
    uconv3      = SEResNetBottleNeck(filters=filters[3], kernel_size=kernel_sizes[3], strides=(1,1,1), reduction=se_reduction[3], conv_params=conv_params)(uconv3)
    uconv3      = DropoutFunc(dropout_rate)(uconv3)
  
    # Decoder: Nested U-Net - Stage 2
    deconv2     = tf.keras.layers.Conv3DTranspose(filters=filters[2], kernel_size=kernel_sizes[3], strides=strides[3], padding="same")(uconv3)
    deconv2_up1 = tf.keras.layers.Conv3DTranspose(filters=filters[1], kernel_size=kernel_sizes[2], strides=strides[2], padding="same")(deconv2)
    deconv2_up2 = tf.keras.layers.Conv3DTranspose(filters=filters[0], kernel_size=kernel_sizes[1], strides=strides[1], padding="same")(deconv2_up1)
    uconv2      = tf.keras.layers.concatenate([deconv2, deconv3_up1, att_conv2]) 
    uconv2      = SEResNetBottleNeck(filters=filters[2], kernel_size=kernel_sizes[2], strides=(1,1,1), reduction=se_reduction[2], conv_params=conv_params)(uconv2)
    uconv2      = DropoutFunc(dropout_rate)(uconv2)

    # Decoder: Nested U-Net - Stage 1
    deconv1     = tf.keras.layers.Conv3DTranspose(filters=filters[1], kernel_size=kernel_sizes[2], strides=strides[2], padding="same")(uconv2)
    deconv1_up1 = tf.keras.layers.Conv3DTranspose(filters=filters[0], kernel_size=kernel_sizes[1], strides=strides[1], padding="same")(deconv1)
    uconv1      = tf.keras.layers.concatenate([deconv1, deconv2_up1, deconv3_up2, att_conv1])
    uconv1      = SEResNetBottleNeck(filters=filters[1], kernel_size=kernel_sizes[1], strides=(1,1,1), reduction=se_reduction[1], conv_params=conv_params)(uconv1)
    uconv1      = DropoutFunc(dropout_rate)(uconv1)

    # Decoder: Nested U-Net - Stage 0
    deconv0     = tf.keras.layers.Conv3DTranspose(filters=filters[0], kernel_size=kernel_sizes[1], strides=strides[1], padding="same")(uconv1)
    uconv0      = tf.keras.layers.concatenate([deconv0, deconv1_up1, deconv2_up2, deconv3_up3, att_conv0])   
    uconv0      = tf.keras.layers.Conv3DTranspose(filters=filters[0], kernel_size=kernel_sizes[1], strides=strides[1], padding="same")(uconv1)   
    uconv0      = SEResNetBottleNeck(filters=filters[0], kernel_size=kernel_sizes[0], strides=(1,1,1), reduction=se_reduction[0], conv_params=conv_params)(uconv0)
    uconv0      = DropoutFunc(dropout_rate/2)(uconv0)

    # Final Convolutional Layer [Logits] + Softmax/Argmax
    y__         = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(uconv0)
    y_          = tf.argmax(y__, axis=-1) \
                        if num_classes>1  \
                        else tf.cast(tf.greater_equal(y__[..., 0], 0.5), tf.int32)

    # Deep Supervision
    if deep_supervision:
        # Upsample Feature Maps to Original Resolution
        y_1     = tf.keras.layers.UpSampling3D(size=np.array(strides[1])                                          )(uconv1)
        y_2     = tf.keras.layers.UpSampling3D(size=np.array(strides[2])*np.array(strides[3])                     )(uconv2)
        y_3     = tf.keras.layers.UpSampling3D(size=np.array(strides[3])*np.array(strides[3])*np.array(strides[3]))(uconv3)
        # Generate Logits
        y_1     = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(y_1)
        y_2     = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(y_2)
        y_3     = tf.keras.layers.Conv3D(filters=num_classes, kernel_size=(1,1,1), strides=(1,1,1), **conv_params)(y_3)


    # Model Summary
    if summary:
        print('Input Volume:---------------------------', x_.shape)
        print('Initial Convolutional Layer (Stage 0):--', x.shape)
        print('Attention Gating: Stage 0:--------------', att_conv0.shape)
        print('Encoder: Stage 1; SE-Residual Block:----', conv1.shape)
        print('Attention Gating: Stage 1:--------------', att_conv1.shape)
        print('Encoder: Stage 2; SE-Residual Block:----', conv2.shape)
        print('Attention Gating: Stage 2:--------------', att_conv2.shape)    
        print('Encoder: Stage 3; SE-Residual Block:----', conv3.shape)
        print('Attention Gating: Stage 3:--------------', att_conv3.shape)    
        print('Middle: High-Dim Latent Features:-------', convm.shape)
        print('Decoder: Stage 3; Nested U-Net:---------', uconv3.shape)
        print('Decoder: Stage 2; Nested U-Net:---------', uconv2.shape)
        print('Decoder: Stage 1; Nested U-Net:---------', uconv1.shape)    
        print('Decoder: Stage 0; Nested U-Net:---------', uconv0.shape)    
        print('U-Net [Logits]:-------------------------', y__.shape)

    # Probabilistic Variant -------------------------------------------------------------------------------------------------------------------------------------------------
    if probabilistic:

        # Extract Image and Label
        image = inputs[:,:,:,:,:-(num_classes-1)]
        label = inputs[:,:,:,:,-(num_classes-1)-1:-1]

        # Prior Variational Auto Encoder
        prior_dist     = AxisAligned3DConvGaussian(filters            =  filters,            
                                                   strides            =  strides,
                                                   kernel_sizes       =  kernel_sizes,            
                                                   se_reduction       =  se_reduction,
                                                   kernel_initializer =  kernel_initializer,
                                                   bias_initializer   =  bias_initializer,  
                                                   kernel_regularizer =  kernel_regularizer,
                                                   bias_regularizer   =  bias_regularizer,
                                                   proba_event_shape  =  proba_event_shape,
                                                   latent_tensor_name = 'prior_distribution')(image,None)
        # Posterior Variational Auto Encoder
        posterior_dist = AxisAligned3DConvGaussian(filters            =  filters,            
                                                   strides            =  strides,
                                                   kernel_sizes       =  kernel_sizes,            
                                                   se_reduction       =  se_reduction,
                                                   kernel_initializer =  kernel_initializer,
                                                   bias_initializer   =  bias_initializer,  
                                                   kernel_regularizer =  kernel_regularizer,
                                                   bias_regularizer   =  bias_regularizer,
                                                   proba_event_shape  =  proba_event_shape,
                                                   latent_tensor_name = 'posterior_distribution')(image,label)
        
        # Compute Prior and Posterior Distributions and Sample One Point From Each
        _p  = tfp.distributions.MultivariateNormalDiag(loc=prior_dist[:,:proba_event_shape],     scale_diag=prior_dist[:,proba_event_shape:])
        _q  = tfp.distributions.MultivariateNormalDiag(loc=posterior_dist[:,:proba_event_shape], scale_diag=posterior_dist[:,proba_event_shape:])
        z_p = _p.sample()
        z_q = _q.sample()
        print('Prior Distribution [VAE]:---------------', _p)                
        print('Posterior Distribution [VAE]:-----------', _q)

        # Broadcast Sampled Latent Vector to Spatial Dimensions of Logits
        spatial_shape = [uconv0.shape[axis] for axis in [1,2,3]]
        multiples     = [1] + spatial_shape
        multiples.insert(4,1)
        for _ in range(3):  
            z_p = tf.expand_dims(z_p, axis=1)
            z_q = tf.expand_dims(z_q, axis=1)
        z_p     = tf.tile(z_p,multiples)
        z_q     = tf.tile(z_q,multiples)

        # Final Convolutional Layers at Train-Time/Inference
        final_conv = Conv1x1x1withLatentDist(num_classes        =  num_classes, 
                                             num_channels       =  proba_event_shape,                                     
                                             kernel_initializer =  kernel_initializer,
                                             bias_initializer   =  bias_initializer,  
                                             kernel_regularizer =  kernel_regularizer,
                                             bias_regularizer   =  bias_regularizer,
                                             logits_tensor_name = 'final_logits_tensor')

        infer_conv = final_conv(tf.concat([uconv0, z_p], axis=-1))
        train_conv = final_conv(tf.concat([uconv0, z_q], axis=-1))

        # Compute Kullback-Leibler Divergence KL(Q||P) + Softmax(Reconstructed Logits)
        kl         = tfp.distributions.kl_divergence(_q,_p)

        print('Concatenated Latent Tensor:-------------', tf.concat([uconv0, z_p], axis=-1).shape)
        print('Kullback-Leibler Divergence [KL(Q||P)]:-', kl.shape)   

        # Export Probabilistic Output Tensors
        outputs['prob_infer_conv'] = infer_conv
        outputs['prob_train_conv'] = train_conv
        outputs['prob_kl']         = kl

        if deep_supervision: outputs['prob_softmax'] = tf.concat([tf.keras.activations.softmax(t) for t in [train_conv, y_1, y_2, y_3]], axis=-1)
        else:                outputs['prob_softmax'] = tf.keras.activations.softmax(train_conv)

    # Export Common Output Tensors
    if deep_supervision:
        outputs['y_softmax'] = tf.concat([tf.keras.activations.softmax(t) for t in [y__, y_1, y_2, y_3]], axis=-1)     
        outputs['y_sigmoid'] = tf.concat([tf.keras.activations.sigmoid(t) for t in [y__, y_1, y_2, y_3]], axis=-1)
    else:
        outputs['y_softmax'] = tf.keras.activations.softmax(y__)     
        outputs['y_sigmoid'] = tf.keras.activations.sigmoid(y__)
    
    outputs['pre_logits']    = uconv0
    outputs['logits']        = y__
    outputs['y_']            = y_

    return outputs  
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

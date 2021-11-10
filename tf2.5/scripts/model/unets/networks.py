import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from .modelio import LoadableModel, store_config_args
from .network_blocks import SEResNetBottleNeck, GridAttentionBlock3D,\
                            StitchingProbDecoder, MonteCarloDropout
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


# (Hierarchical Probabilistic) 3D U-Net: Top-Level Wrapper for Inter-Model Architecture (Cascaded/Stand-Alone) ----------------------------------------------------------
class M1(LoadableModel):
    '''
    [1] Z. Zhou et al. (2019), "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", IEEE TMI.
    [2] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    [3] S. Kohl et al. (2019), "A Hierarchical Probabilistic U-Net for Modeling Multi-Scale Ambiguities", NeurIPS.
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
                 att_sub_samp       = ((1,1,1),(1,1,1),(1,1,1)),      
                 kernel_initializer =   tf.keras.initializers.Orthogonal(gain=1.0),
                 bias_initializer   =   tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
                 kernel_regularizer =   tf.keras.regularizers.l2(1e-4),
                 bias_regularizer   =   tf.keras.regularizers.l2(1e-4),         
                 cascaded           =   False,
                 dense_skip         =   False,
                 deep_supervision   =   False,
                 probabilistic      =   False,
                 prob_latent_dims   =  (3,2,1),
                 summary            =   True,
                 name               =  'UNET-TYPE-M1'):

        # Ensure Correct Dimensionality
        ndims = len(input_spatial_dims)
        assert ndims in [1,2,3], 'Variable (ndims) should be  1, 2 or 3. Found: %d.'%ndims
        
        # Standard Single-Stage Model
        if (cascaded==False):
            # Input Layer Definition
            image = tf.keras.Input(shape=(*input_spatial_dims, input_channels), name='image')
            
            # U-Net Model Definition
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
                             dense_skip         = dense_skip,
                             deep_supervision   = deep_supervision,
                             probabilistic      = probabilistic,
                             prob_latent_dims   = prob_latent_dims,
                             summary            = True)
            
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
            self.references.num_classes   = num_classes    

        # Cascaded Two-Stage Model
        else:
            # Input Layers Definition
            image_v1 = tf.keras.Input(shape=(*input_spatial_dims, input_channels), name='image_1')
            image_v2 = tf.keras.Input(shape=(*input_spatial_dims, input_channels), name='image_2')
            
            # First-Stage U-Net Model Definition
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
                             dense_skip         = dense_skip,
                             deep_supervision   = deep_supervision,
                             probabilistic      = probabilistic,
                             prob_latent_dims   = prob_latent_dims,
                             summary            = True)

            # Second-Stage U-Net Model Definition
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
                             dense_skip         = dense_skip,                             
                             deep_supervision   = deep_supervision,
                             probabilistic      = probabilistic,
                             prob_latent_dims   = prob_latent_dims,
                             summary            = True)

            # Coupled Inference via Specified Aggregation Strategy
            prior_pred_train, joint_pred_train     = self.decision_fusion(prior_softmax     = m1_stage1['prob_softmax'][:,:,:,:,num_classes-1] if probabilistic \
                                                                                              else m1_stage1['y_softmax'][:,:,:,:,num_classes-1],
                                                                          follow_up_softmax = m1_stage2['prob_softmax'][:,:,:,:,num_classes-1] if probabilistic \
                                                                                              else m1_stage2['y_softmax'][:,:,:,:,num_classes-1],
                                                                          strategy          = cascaded)
            if probabilistic:
                prior_pred_infer, joint_pred_infer = self.decision_fusion(prior_softmax     = tf.keras.activations.softmax(m1_stage1['prob_infer_conv'])\
                                                                                              [:,:,:,:,num_classes-1],
                                                                          follow_up_softmax = tf.keras.activations.softmax(m1_stage2['prob_infer_conv'])\
                                                                                              [:,:,:,:,num_classes-1],
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
            self.references.num_classes   = num_classes

    # Returns Reconfigured Model to Predict Segment Probabilities Only
    def get_detect_model(self):
        if (self.references.cascaded!=False):
            if self.references.probabilistic: 
                return tf.keras.Model(self.inputs, [self.references.infer_softmax_1,\
                                                    self.references.infer_softmax_2])
            else:                             
                return tf.keras.Model(self.inputs, [self.references.m1_stage1['y_softmax'][:,:,:,:,:self.references.num_classes],\
                                                    self.references.m1_stage2['y_softmax'][:,:,:,:,:self.references.num_classes]])
        else:
            if self.references.probabilistic: return tf.keras.Model(self.inputs,  self.references.infer_softmax)
            else:                             return tf.keras.Model(self.inputs,  self.references.m1_model['y_softmax'][:,:,:,:,:self.references.num_classes])

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






# (Hierarchical Probabilistic) 3D U-Net: Mid-Level Wrapper for Intra-Model Architecture (Deterministic/Probabilistic) ---------------------------------------------------
def m1(inputs, num_classes, 
       dropout_mode       =  'standard',
       dropout_rate       =   0.50,
       filters            =  (32,64,128,256,512), 
       strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2)),           
       kernel_sizes       = ((1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3)),
       se_reduction       =  (8,8,8,8,8),
       att_sub_samp       = ((1,1,1),(1,1,1),(1,1,1),(1,1,1)),
       kernel_initializer =   tf.keras.initializers.Orthogonal(gain=1.0),
       bias_initializer   =   tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
       kernel_regularizer =   tf.keras.regularizers.l2(1e-4),
       bias_regularizer   =   tf.keras.regularizers.l2(1e-4),
       dense_skip         =   False,
       deep_supervision   =   False,
       probabilistic      =   False,
       prob_latent_dims   =  (1,1,1,1),
       summary            =   True):
    """
    [1] Z. Zhou et al. (2019), "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", IEEE TMI.
    [2] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    [3] S. Kohl et al. (2019), "A Hierarchical Probabilistic U-Net for Modeling Multi-Scale Ambiguities", NeurIPS.
    [4] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    [5] K. He et al.(2016), "Deep Residual Learning for Image Recognition", IEEE CVPR.
    [6] K. He et al.(2016), "Identity Mappings in Deep Residual Networks", ECCV.
    """
    # Preamble
    outputs     = {}
    conv_params = {'padding':           'same',
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer':   bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer':   bias_regularizer}

    # Deterministic Variant -------------------------------------------------------------------------------------------------------------------------------------------------
    if not probabilistic:
        m1_features = M1Core(num_classes        =  num_classes,
                             dropout_mode       =  dropout_mode,
                             dropout_rate       =  dropout_rate,
                             filters            =  filters, 
                             strides            =  strides,           
                             kernel_sizes       =  kernel_sizes,
                             se_reduction       =  se_reduction,
                             att_sub_samp       =  att_sub_samp,
                             kernel_initializer =  kernel_initializer,
                             bias_initializer   =  bias_initializer,
                             kernel_regularizer =  kernel_regularizer,
                             bias_regularizer   =  bias_regularizer,
                             dense_skip         =  dense_skip,
                             deep_supervision   =  deep_supervision,
                             probabilistic      =  probabilistic)(inputs=inputs)

        # Print Brief Model Overview
        print('--------------------------------------------------------------------')
        print('Deterministic 3D U-Net (Type: M1)')
        print('--------------------------------------------------------------------')
        if summary: m1_features.summary()
        print('--------------------------------------------------------------------')        
    
        # Export Common Output Tensors
        outputs['y_softmax']  = m1_features['y_softmax']     
        outputs['y_sigmoid']  = m1_features['y_sigmoid']    
        outputs['logits']     = m1_features['logits']    
        outputs['y_']         = m1_features['y_']        

    # Hierarchical Probabilistic Variant ------------------------------------------------------------------------------------------------------------------------------------
    if probabilistic:

        # Split Image and Label
        image = inputs[:,:,:,:,:-(num_classes-1)]
        label = inputs[:,:,:,:,-(num_classes-1)-1:-1]

        # Prior Network (w/ Images as Input)
        prior_p_features = M1Core(num_classes        =  num_classes,
                                  dropout_mode       =  dropout_mode,
                                  dropout_rate       =  dropout_rate,
                                  filters            =  filters, 
                                  strides            =  strides,           
                                  kernel_sizes       =  kernel_sizes,
                                  se_reduction       =  se_reduction,
                                  att_sub_samp       =  att_sub_samp,
                                  kernel_initializer =  kernel_initializer,
                                  bias_initializer   =  bias_initializer,
                                  kernel_regularizer =  kernel_regularizer,
                                  bias_regularizer   =  bias_regularizer,
                                  dense_skip         =  dense_skip,
                                  probabilistic      =  probabilistic,
                                  prob_latent_dims   =  prob_latent_dims)

        # Posterior Network (w/ Images+Labels as Input)
        posterior_q_features = M1Core(num_classes    =  num_classes,
                                  dropout_mode       =  dropout_mode,
                                  dropout_rate       =  dropout_rate,
                                  filters            =  filters, 
                                  strides            =  strides,           
                                  kernel_sizes       =  kernel_sizes,
                                  se_reduction       =  se_reduction,
                                  att_sub_samp       =  att_sub_samp,
                                  kernel_initializer =  kernel_initializer,
                                  bias_initializer   =  bias_initializer,
                                  kernel_regularizer =  kernel_regularizer,
                                  bias_regularizer   =  bias_regularizer,
                                  dense_skip         =  dense_skip,
                                  probabilistic      =  probabilistic,
                                  prob_latent_dims   =  prob_latent_dims)

        # Final Decoder at Train-Time/Inference
        final_decoder = StitchingProbDecoder(num_classes        =  num_classes, 
                                             filters            =  filters,            
                                             strides            =  strides,                    
                                             kernel_sizes       =  kernel_sizes,
                                             kernel_initializer =  kernel_initializer,
                                             bias_initializer   =  bias_initializer,  
                                             kernel_regularizer =  kernel_regularizer,
                                             bias_regularizer   =  bias_regularizer)

        # Sample Points in Latent Distributions Z and Q
        q_sample          = posterior_q_features(inputs=tf.concat([image,label],axis=-1), prob_mean=False, prob_z_q=None)
        q_mean            = posterior_q_features(inputs=tf.concat([image,label],axis=-1), prob_mean=True,  prob_z_q=None)
        p_sample          = prior_p_features(inputs=image, prob_mean=False, prob_z_q=None)
        p_sample_z_q      = prior_p_features(inputs=image, prob_mean=False, prob_z_q=q_sample['prob_used_latents'])
        p_sample_z_q_mean = prior_p_features(inputs=image, prob_mean=False, prob_z_q=q_mean['prob_used_latents'])

        # Inject Latent Points for Optimization/Inference
        infer_conv = final_decoder(decoder_features=p_sample['prob_decoder_features'])   
        train_conv = final_decoder(decoder_features=p_sample_z_q_mean['prob_decoder_features'])
        
        # Print Brief Model Overview
        print('-------------------------------------------------------------------------------------')
        print('Hierarchical Prob. 3D U-Net (Type: M1) - Prior Network')
        print('-------------------------------------------------------------------------------------')
        if summary: prior_p_features.summary()
        print('-------------------------------------------------------------------------------------')
        print('Hierarchical Prob. 3D U-Net (Type: M1) - Posterior Network')
        print('-------------------------------------------------------------------------------------')
        if summary: posterior_q_features.summary()
        print('-------------------------------------------------------------------------------------')
        print('Hierarchical Prob. 3D U-Net (Type: M1) - P [Logits]:--------', train_conv.shape)
        print('Hierarchical Prob. 3D U-Net (Type: M1) - Q [Logits]:--------', infer_conv.shape)
        print('-------------------------------------------------------------------------------------')

        # Compute Kullback-Leibler Divergence KL(Q||P) + Softmax(Reconstructed Logits)
        kl, kl_elems = {},0
        for level,(q,p) in enumerate(zip(q_sample['prob_distributions'], p_sample_z_q['prob_distributions'])):
          kl_per_voxel    = tfp.distributions.kl_divergence(q,p)       # Shape: (B,D,H,W)
          kl_per_instance = tf.reduce_sum(kl_per_voxel, axis=[1,2,3])  # Shape: (B,)
          kl[level]       = tf.reduce_mean(kl_per_instance)            # Shape: (1,)
          kl_elems       += np.prod(kl_per_voxel.get_shape()[1:])

        print('Kullback-Leibler Divergence Elems. [KL(Q||P)]: ', kl_elems)   

        # Export Probabilistic Output Tensors
        outputs['prob_infer_conv'] = infer_conv
        outputs['prob_train_conv'] = train_conv
        outputs['prob_kl']         = tf.reduce_sum(tf.stack([k for _, k in kl.items()], axis=-1))
        
        # Deep Supervision or Otherwise
        if deep_supervision:  outputs['prob_softmax'] = tf.concat([tf.keras.activations.softmax(train_conv),\
                                                        p_sample_z_q_mean['y_softmax'][...,num_classes:]], axis=-1) 
        else:                 outputs['prob_softmax'] = tf.keras.activations.softmax(train_conv)
    
    return outputs  
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------







# (Hierarchical Probabilistic) 3D U-Net: Low-Level Wrapper for CNN Architecture (Attention Gates + Nested Decoder + SE) -------------------------------------------------
class M1Core(snt.Module):
    """
    [1] Z. Zhou et al. (2019), "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", IEEE TMI.
    [2] J. Hu et al.(2019), "Squeeze-and-Excitation Networks", IEEE TPAMI.
    [3] S. Kohl et al. (2019), "A Hierarchical Probabilistic U-Net for Modeling Multi-Scale Ambiguities", NeurIPS.
    [4] O. Oktay et al. (2018), "Attention U-Net: Learning Where to Look for the Pancreas", MIDL.
    [5] K. He et al.(2016), "Deep Residual Learning for Image Recognition", IEEE CVPR.
    [6] K. He et al.(2016), "Identity Mappings in Deep Residual Networks", ECCV.
    
    U-Net Schematic:
        Resol. 0  (x)------------->(att_conv0)-->(deconv2_up2)-->(deconv1_up1)-->(deconv0)-->(uconv0_)-->(uconv0)-->(y__)
        Resol. 1   |---->(conv1)-->(att_conv1)-->(deconv3_up2)-->(deconv2_up1)-->(deconv1)-->(uconv1_)-->(uconv1)
        Resol. 2            |----->(conv2)------>(att_conv2)---->(deconv3_up1)-->(deconv2)-->(uconv2_)-->(uconv2)
        Resol. 3                      |--------->(conv3)-------->(att_conv3)---->(deconv3)-->(uconv3_)-->(uconv3)
        Resol. 4                                    |----------->(convm)-------------|
    """
    def __init__(self, 
                 num_classes        =   2, 
                 dropout_mode       =  'standard',
                 dropout_rate       =   0.50,
                 filters            =  (32,64,128,256,512), 
                 strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2)),           
                 kernel_sizes       = ((1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3)),
                 se_reduction       =  (8,8,8,8,8),
                 att_sub_samp       = ((1,1,1),(1,1,1),(1,1,1),(1,1,1)),
                 kernel_initializer =   tf.keras.initializers.Orthogonal(gain=1.0),
                 bias_initializer   =   tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
                 kernel_regularizer =   tf.keras.regularizers.l2(1e-4),
                 bias_regularizer   =   tf.keras.regularizers.l2(1e-4),
                 dense_skip         =   False,
                 deep_supervision   =   False,
                 probabilistic      =   False,
                 prob_latent_dims   =  (1,1,1,1)):

        super(M1Core, self).__init__()

        self.num_classes        = num_classes
        self.dropout_mode       = dropout_mode      
        self.dropout_rate       = dropout_rate      
        self.filters            = filters           
        self.strides            = strides
        self.kernel_sizes       = kernel_sizes      
        self.se_reduction       = se_reduction      
        self.att_sub_samp       = att_sub_samp      
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer  
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer   = bias_regularizer  
        self.dense_skip         = dense_skip           
        self.deep_supervision   = deep_supervision  
        self.probabilistic      = probabilistic     
        self.prob_latent_dims   = prob_latent_dims          

        # Preamble
        self.conv_params = {'padding':           'same',
                            'kernel_initializer': self.kernel_initializer,
                            'bias_initializer':   self.bias_initializer,
                            'kernel_regularizer': self.kernel_regularizer,
                            'bias_regularizer':   self.bias_regularizer}
    
        if   (self.dropout_mode=='standard'):     DropoutFunc = tf.keras.layers.Dropout
        elif (self.dropout_mode=='monte-carlo'):  DropoutFunc = MonteCarloDropout
    
        assert len(self.filters)==5, "ERROR: Expected Tuple/Array with 5 Values (One Per Resolution)."
        assert len(self.se_reduction)==5, "ERROR: Expected Tuple/Array with 5 Values (One Per Resolution)."
        assert [len(a) for a in self.att_sub_samp]==[3,3,3,3], "ERROR: Expected 4x3 Tuple/Array (3D Sub-Sampling Factors for 4 Attention Gates)."
        assert [len(s) for s in self.strides]==[3,3,3,3,3], "ERROR: Expected 5x3 Tuple/Array (3D Strides for 5 Resolutions)."
        assert [len(k) for k in self.kernel_sizes]==[3,3,3,3,3], "ERROR: Expected 5x3 Tuple/Array (3D Kernels for 5 Resolutions)."
    
        # Preliminary Convolutional Layer
        self.conve0 = tf.keras.layers.Conv3D(filters=self.filters[0], kernel_size=self.kernel_sizes[0], strides=self.strides[0], **self.conv_params)
        self.norme0 = tfa.layers.InstanceNormalization()
    
        # Encoder: Backbone SE-Residual Blocks for Feature Extraction
        self.serse1 = SEResNetBottleNeck(filters=self.filters[1], kernel_size=self.kernel_sizes[1], strides=self.strides[1],\
                                         reduction=self.se_reduction[1], conv_params=self.conv_params) 
        self.drope1 = DropoutFunc(self.dropout_rate)
        self.serse2 = SEResNetBottleNeck(filters=self.filters[2], kernel_size=self.kernel_sizes[2], strides=self.strides[2],\
                                         reduction=self.se_reduction[2], conv_params=self.conv_params) 
        self.drope2 = DropoutFunc(self.dropout_rate)
        self.serse3 = SEResNetBottleNeck(filters=self.filters[3], kernel_size=self.kernel_sizes[3], strides=self.strides[3],\
                                         reduction=self.se_reduction[3], conv_params=self.conv_params)  
        self.drope3 = DropoutFunc(self.dropout_rate)
        self.serse4 = SEResNetBottleNeck(filters=self.filters[4], kernel_size=self.kernel_sizes[4], strides=self.strides[4],\
                                         reduction=self.se_reduction[4], conv_params=self.conv_params)
        self.drope4 = DropoutFunc(self.dropout_rate)
    
        # Grid-Based Attention Gating
        self.att0   = GridAttentionBlock3D(inter_channels=self.filters[0], sub_samp=self.att_sub_samp[0], conv_params=self.conv_params)
        self.att1   = GridAttentionBlock3D(inter_channels=self.filters[1], sub_samp=self.att_sub_samp[1], conv_params=self.conv_params)
        self.att2   = GridAttentionBlock3D(inter_channels=self.filters[2], sub_samp=self.att_sub_samp[2], conv_params=self.conv_params)
        self.att3   = GridAttentionBlock3D(inter_channels=self.filters[3], sub_samp=self.att_sub_samp[3], conv_params=self.conv_params)
        
        # Decoder: Nested U-Net - Stage 3
        self.convtd3     = tf.keras.layers.Conv3DTranspose(filters=self.filters[3], kernel_size=self.kernel_sizes[4], strides=self.strides[4], **self.conv_params)
        self.convtd3_up1 = tf.keras.layers.Conv3DTranspose(filters=self.filters[2], kernel_size=self.kernel_sizes[3], strides=self.strides[3], **self.conv_params)
        self.convtd3_up2 = tf.keras.layers.Conv3DTranspose(filters=self.filters[1], kernel_size=self.kernel_sizes[2], strides=self.strides[2], **self.conv_params)
        self.convtd3_up3 = tf.keras.layers.Conv3DTranspose(filters=self.filters[0], kernel_size=self.kernel_sizes[1], strides=self.strides[1], **self.conv_params)
        self.sersd3      = SEResNetBottleNeck(filters=self.filters[3], kernel_size=self.kernel_sizes[3], strides=(1,1,1),\
                                              reduction=self.se_reduction[3], conv_params=self.conv_params)
        self.dropd3      = DropoutFunc(self.dropout_rate)
      
        # Decoder: Nested U-Net - Stage 2
        self.convtd2     = tf.keras.layers.Conv3DTranspose(filters=self.filters[2], kernel_size=self.kernel_sizes[3], strides=self.strides[3], **self.conv_params)
        self.convtd2_up1 = tf.keras.layers.Conv3DTranspose(filters=self.filters[1], kernel_size=self.kernel_sizes[2], strides=self.strides[2], **self.conv_params)
        self.convtd2_up2 = tf.keras.layers.Conv3DTranspose(filters=self.filters[0], kernel_size=self.kernel_sizes[1], strides=self.strides[1], **self.conv_params)
        self.sersd2      = SEResNetBottleNeck(filters=self.filters[2], kernel_size=self.kernel_sizes[2], strides=(1,1,1),\
                                              reduction=self.se_reduction[2], conv_params=self.conv_params)
        self.dropd2      = DropoutFunc(self.dropout_rate)
    
        # Decoder: Nested U-Net - Stage 1
        self.convtd1     = tf.keras.layers.Conv3DTranspose(filters=self.filters[1], kernel_size=self.kernel_sizes[2], strides=self.strides[2], **self.conv_params)
        self.convtd1_up1 = tf.keras.layers.Conv3DTranspose(filters=self.filters[0], kernel_size=self.kernel_sizes[1], strides=self.strides[1], **self.conv_params)
        self.sersd1      = SEResNetBottleNeck(filters=self.filters[1], kernel_size=self.kernel_sizes[1], strides=(1,1,1),\
                                              reduction=self.se_reduction[1], conv_params=self.conv_params)
        self.dropd1      = DropoutFunc(self.dropout_rate)
    
        # Decoder: Nested U-Net - Stage 0
        self.convtd0     = tf.keras.layers.Conv3DTranspose(filters=self.filters[0], kernel_size=self.kernel_sizes[1], strides=self.strides[1], **self.conv_params)
        self.sersd0      = SEResNetBottleNeck(filters=self.filters[0], kernel_size=self.kernel_sizes[0], strides=(1,1,1),\
                                              reduction=self.se_reduction[0], conv_params=self.conv_params)
        self.dropd0      = DropoutFunc(self.dropout_rate/2)
    
        # Final Convolutional Layer [Logits] + Softmax/Argmax
        self.logits      = tf.keras.layers.Conv3D(filters=self.num_classes, kernel_size=(1,1,1), strides=(1,1,1), **self.conv_params)
            
        # Deep Supervision - Generate Logits
        self.dsy1_logits = tf.keras.layers.Conv3D(filters=self.num_classes, kernel_size=(1,1,1), strides=(1,1,1), **self.conv_params)
        self.dsy2_logits = tf.keras.layers.Conv3D(filters=self.num_classes, kernel_size=(1,1,1), strides=(1,1,1), **self.conv_params)
        self.dsy3_logits = tf.keras.layers.Conv3D(filters=self.num_classes, kernel_size=(1,1,1), strides=(1,1,1), **self.conv_params)
         
        # Hierarchical Probabilistic Variant - Predict Gaussian Distribution for Each Voxel in Feature Map
        self.mu_logsig3 = tf.keras.layers.Conv3D(filters=2*self.prob_latent_dims[0], kernel_size=(1,1,1), strides=(1,1,1), **self.conv_params)
        self.mu_logsig2 = tf.keras.layers.Conv3D(filters=2*self.prob_latent_dims[1], kernel_size=(1,1,1), strides=(1,1,1), **self.conv_params)
        self.mu_logsig1 = tf.keras.layers.Conv3D(filters=2*self.prob_latent_dims[2], kernel_size=(1,1,1), strides=(1,1,1), **self.conv_params)
        self.mu_logsig0 = tf.keras.layers.Conv3D(filters=2*self.prob_latent_dims[3], kernel_size=(1,1,1), strides=(1,1,1), **self.conv_params)
        
        # Clipping to Prevent Divergence and NaN Loss [Ref: https://github.com/y0ast/VAE-Torch/issues/3]
        self.prob_dist3 = tfp.layers.DistributionLambda(lambda t: tfp.distributions.MultivariateNormalDiag(loc=t[0], scale_diag=tf.exp(tf.clip_by_value(t[1],-0.1,0.1))))
        self.prob_dist2 = tfp.layers.DistributionLambda(lambda t: tfp.distributions.MultivariateNormalDiag(loc=t[0], scale_diag=tf.exp(tf.clip_by_value(t[1],-0.1,0.1))))
        self.prob_dist1 = tfp.layers.DistributionLambda(lambda t: tfp.distributions.MultivariateNormalDiag(loc=t[0], scale_diag=tf.exp(tf.clip_by_value(t[1],-0.1,0.1))))
        self.prob_dist0 = tfp.layers.DistributionLambda(lambda t: tfp.distributions.MultivariateNormalDiag(loc=t[0], scale_diag=tf.exp(tf.clip_by_value(t[1],-0.1,0.1))))

        # Hierarchical Probabilistic Variant - Predict Gaussian Distribution for Each Voxel in Feature Map
        self.dec_hi3 = tf.keras.layers.Conv3DTranspose(filters=self.filters[::-1][0+1], kernel_size=self.kernel_sizes[::-1][0],\
                                                       strides=self.strides[::-1][0], **self.conv_params)
        self.dec_hi2 = tf.keras.layers.Conv3DTranspose(filters=self.filters[::-1][1+1], kernel_size=self.kernel_sizes[::-1][1],\
                                                       strides=self.strides[::-1][1], **self.conv_params)
        self.dec_hi1 = tf.keras.layers.Conv3DTranspose(filters=self.filters[::-1][2+1], kernel_size=self.kernel_sizes[::-1][2],\
                                                       strides=self.strides[::-1][2], **self.conv_params)
        self.dec_hi0 = tf.keras.layers.Conv3DTranspose(filters=self.filters[::-1][3+1], kernel_size=self.kernel_sizes[::-1][3],\
                                                       strides=self.strides[::-1][3], **self.conv_params)
        self.sersp3  = SEResNetBottleNeck(filters=self.filters[::-1][0+1], kernel_size=self.kernel_sizes[::-1][0+1], strides=(1,1,1),\
                                          reduction=self.se_reduction[::-1][0+1], conv_params=self.conv_params)
        self.sersp2  = SEResNetBottleNeck(filters=self.filters[::-1][1+1], kernel_size=self.kernel_sizes[::-1][1+1], strides=(1,1,1),\
                                          reduction=self.se_reduction[::-1][1+1], conv_params=self.conv_params)
        self.sersp1  = SEResNetBottleNeck(filters=self.filters[::-1][2+1], kernel_size=self.kernel_sizes[::-1][2+1], strides=(1,1,1),\
                                          reduction=self.se_reduction[::-1][2+1], conv_params=self.conv_params)
        self.sersp0  = SEResNetBottleNeck(filters=self.filters[::-1][3+1], kernel_size=self.kernel_sizes[::-1][3+1], strides=(1,1,1),\
                                          reduction=self.se_reduction[::-1][3+1], conv_params=self.conv_params)
        self.dropp3  = DropoutFunc(self.dropout_rate)
        self.dropp2  = DropoutFunc(self.dropout_rate)
        self.dropp1  = DropoutFunc(self.dropout_rate)
        self.dropp0  = DropoutFunc(self.dropout_rate)


    def __call__(self, inputs, prob_mean, prob_z_q):
    
        self.inputs = inputs
        outputs     = {}
    
        # Preliminary Convolutional Layer
        x      = self.conve0(inputs)
        x      = self.norme0(x)
        self.x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)  

        # Encoder: Backbone SE-Residual Blocks for Feature Extraction
        self.conv1 = self.drope1(self.serse1(self.x))
        self.conv2 = self.drope2(self.serse2(self.conv1))
        self.conv3 = self.drope3(self.serse3(self.conv2))
        self.convm = self.drope4(self.serse4(self.conv3))
    
        # Grid-Based Attention Gating
        self.att_conv0, att_0 = self.att0(conv_tensor=self.x,     gating_tensor=self.convm)
        self.att_conv1, att_1 = self.att1(conv_tensor=self.conv1, gating_tensor=self.convm)
        self.att_conv2, att_2 = self.att2(conv_tensor=self.conv2, gating_tensor=self.convm)
        self.att_conv3, att_3 = self.att3(conv_tensor=self.conv3, gating_tensor=self.convm)
    
        # Decoder: Nested U-Net - Stage 3
        deconv3          = self.convtd3(self.convm)
        if self.dense_skip:
            deconv3_up1  = self.convtd3_up1(deconv3)
            deconv3_up2  = self.convtd3_up2(deconv3_up1)
            deconv3_up3  = self.convtd3_up3(deconv3_up2)
        self.uconv3_     = tf.concat([deconv3, self.att_conv3], axis=-1)    
        self.uconv3      = self.dropd3(self.sersd3(self.uconv3_))
      
        # Decoder: Nested U-Net - Stage 2
        deconv2          = self.convtd2(self.uconv3)
        if self.dense_skip:
            deconv2_up1  = self.convtd2_up1(deconv2)
            deconv2_up2  = self.convtd2_up2(deconv2_up1)
            self.uconv2_ = tf.concat([deconv2, deconv3_up1, self.att_conv2], axis=-1) 
        else:
            self.uconv2_ = tf.concat([deconv2, self.att_conv2], axis=-1)             
        self.uconv2      = self.dropd2(self.sersd2(self.uconv2_))
    
        # Decoder: Nested U-Net - Stage 1
        deconv1          = self.convtd1(self.uconv2)
        if self.dense_skip:
            deconv1_up1  = self.convtd1_up1(deconv1)
            self.uconv1_ = tf.concat([deconv1, deconv2_up1, deconv3_up2, self.att_conv1], axis=-1)
        else:
            self.uconv1_ = tf.concat([deconv1, self.att_conv1], axis=-1)            
        self.uconv1      = self.dropd1(self.sersd1(self.uconv1_))
    
        # Decoder: Nested U-Net - Stage 0
        deconv0          = self.convtd0(self.uconv1)
        if self.dense_skip:
            self.uconv0_ = tf.concat([deconv0, deconv1_up1, deconv2_up2, deconv3_up3, self.att_conv0], axis=-1)   
        else:
            self.uconv0_ = tf.concat([deconv0, self.att_conv0], axis=-1)   
        self.uconv0      = self.dropd0(self.sersd0(self.uconv0_))
    
        # Final Convolutional Layer [Logits] + Softmax/Argmax
        self.y__     = self.logits(self.uconv0)
        self.y_      = tf.argmax(self.y__, axis=-1)\
                        if self.num_classes>1\
                        else tf.cast(tf.greater_equal(self.y__[...,0],0.5), tf.int32)
    
        # Hierarchical Probabilistic Variant ----------------------------------------------------------------------------------------------------------------------------
        if self.probabilistic:
            mean, distributions, used_latents, ds_ops = [prob_mean]*len(self.prob_latent_dims), [], [], []

            # For Resolution 3 ------------------------------------------------------------------------------------------------------------------------------------------
            if self.prob_latent_dims[0]!=0:
                # Predict Gaussian Distribution for Each Voxel in Feature Map
                mu_logsigma3 = self.mu_logsig3(self.convm)
                mu3          = mu_logsigma3[..., :self.prob_latent_dims[0]]
                logsigma3    = mu_logsigma3[..., self.prob_latent_dims[0]:]
                distrib3     = tfp.distributions.MultivariateNormalDiag(loc=mu3, scale_diag=tf.exp(tf.clip_by_value(logsigma3,-0.1,0.1)))
              
                # Latents to Condition On
                if prob_z_q is not None: z3 = prob_z_q[0]
                elif mean[0]:            z3 = distrib3.loc
                else:                    z3 = distrib3.sample()
                distributions.append(distrib3)
                used_latents.append(z3)
    
                # Concatenate and Upsample Latents with Features From Lower Spatial Resolution
                decoder_features = self.dropp3(self.sersp3(tf.concat([\
                    self.dec_hi3(tf.concat([z3, self.convm], axis=-1)), self.uconv3_], axis=-1)))
            else:
                decoder_features = self.dropp3(self.sersp3(tf.concat([\
                    self.dec_hi3(self.convm), self.uconv3_], axis=-1)))
            ds_ops.append(decoder_features)

            # For Resolution 2 ------------------------------------------------------------------------------------------------------------------------------------------
            if self.prob_latent_dims[1]!=0:

                # Predict Gaussian Distribution for Each Voxel in Feature Map
                mu_logsigma2 = self.mu_logsig2(decoder_features)
                mu2          = mu_logsigma2[..., :self.prob_latent_dims[1]]
                logsigma2    = mu_logsigma2[..., self.prob_latent_dims[1]:]
                distrib2     = tfp.distributions.MultivariateNormalDiag(loc=mu2, scale_diag=tf.exp(tf.clip_by_value(logsigma2,-0.1,0.1)))

                # Latents to Condition On
                if prob_z_q is not None: z2 = prob_z_q[1]
                elif mean[1]:            z2 = distrib2.loc
                else:                    z2 = distrib2.sample()
                distributions.append(distrib2)
                used_latents.append(z2)
    
                # Concatenate and Upsample Latents with Features From Lower Spatial Resolution
                decoder_features = self.dropp2(self.sersp2(tf.concat([\
                    self.dec_hi2(tf.concat([z2, decoder_features], axis=-1)), self.uconv2_], axis=-1)))
            else:
                decoder_features = self.dropp2(self.sersp2(tf.concat([\
                    self.dec_hi2(decoder_features), self.uconv2_], axis=-1)))
            ds_ops.append(decoder_features)

            # For Resolution 1 ------------------------------------------------------------------------------------------------------------------------------------------
            if self.prob_latent_dims[2]!=0:
            
                # Predict Gaussian Distribution for Each Voxel in Feature Map
                mu_logsigma1 = self.mu_logsig1(decoder_features)
                mu1          = mu_logsigma1[..., :self.prob_latent_dims[2]]
                logsigma1    = mu_logsigma1[..., self.prob_latent_dims[2]:]
                distrib1     = tfp.distributions.MultivariateNormalDiag(loc=mu1, scale_diag=tf.exp(tf.clip_by_value(logsigma1,-0.1,0.1)))

                # Latents to Condition On
                if prob_z_q is not None: z1 = prob_z_q[2]
                elif mean[2]:            z1 = distrib1.loc
                else:                    z1 = distrib1.sample()
                distributions.append(distrib1)
                used_latents.append(z1)

                # Concatenate and Upsample Latents with Features From Lower Spatial Resolution
                decoder_features = self.dropp1(self.sersp1(tf.concat([\
                    self.dec_hi1(tf.concat([z1, decoder_features], axis=-1)), self.uconv1_], axis=-1)))
            else:
                decoder_features = self.dropp1(self.sersp1(tf.concat([\
                    self.dec_hi1(decoder_features), self.uconv1_], axis=-1)))
            ds_ops.append(decoder_features)

            # For Resolution 0 ------------------------------------------------------------------------------------------------------------------------------------------
            if self.prob_latent_dims[3]!=0:

                # Predict Gaussian Distribution for Each Voxel in Feature Map
                mu_logsigma0 = self.mu_logsig0(decoder_features)
                mu0          = mu_logsigma0[..., :self.prob_latent_dims[3]]
                logsigma0    = mu_logsigma0[..., self.prob_latent_dims[3]:]
                distrib0     = tfp.distributions.MultivariateNormalDiag(loc=mu0, scale_diag=tf.exp(tf.clip_by_value(logsigma0,-0.1,0.1)))

                # Latents to Condition On
                if prob_z_q is not None: z0 = prob_z_q[3]
                elif mean[3]:            z0 = distrib0.loc
                else:                    z0 = distrib0.sample()
                distributions.append(distrib0)
                used_latents.append(z0)
    
                # Concatenate and Upsample Latents with Features From Lower Spatial Resolution
                decoder_features = self.dropp0(self.sersp0(tf.concat([\
                    self.dec_hi0(tf.concat([z0, decoder_features], axis=-1)), self.uconv0_], axis=-1)))
            else:
                # Concatenate and Upsample Latents with Features From Lower Spatial Resolution
                decoder_features = self.dropp0(self.sersp0(tf.concat([\
                    self.dec_hi0(decoder_features), self.uconv0_], axis=-1)))

            # Export Probabilistic Tensors ------------------------------------------------------------------------------------------------------------------------------
            outputs['prob_distributions']    =  distributions
            outputs['prob_used_latents']     =  used_latents
            outputs['prob_decoder_features'] =  decoder_features

        # Deep Supervision ----------------------------------------------------------------------------------------------------------------------------------------------
        if self.deep_supervision and not self.probabilistic:
            # Upsample Feature Maps to Original Resolution + Generate Logits
            y_1 = self.dsy1_logits(tf.keras.layers.UpSampling3D(size=np.array(self.strides[1]))(self.uconv1))
            y_2 = self.dsy2_logits(tf.keras.layers.UpSampling3D(size=np.array(self.strides[1])*np.array(self.strides[2]))(self.uconv2))
            y_3 = self.dsy3_logits(tf.keras.layers.UpSampling3D(size=np.array(self.strides[1])*np.array(self.strides[2])*np.array(self.strides[3]))(self.uconv3))
         
        if self.deep_supervision and self.probabilistic:
            # Upsample Feature Maps to Original Resolution + Generate Logits
            y_1 = self.dsy1_logits(tf.keras.layers.UpSampling3D(size=np.array(self.strides[1]))(ds_ops[-1]))
            y_2 = self.dsy2_logits(tf.keras.layers.UpSampling3D(size=np.array(self.strides[1])*np.array(self.strides[2]))(ds_ops[-2]))
            y_3 = self.dsy3_logits(tf.keras.layers.UpSampling3D(size=np.array(self.strides[1])*np.array(self.strides[2])*np.array(self.strides[3]))(ds_ops[-3]))

        # Export Output Tensors
        if self.deep_supervision:
            outputs['y_softmax']   = tf.concat([tf.keras.activations.softmax(t) for t in [self.y__, y_1, y_2, y_3]], axis=-1)     
            outputs['y_sigmoid']   = tf.concat([tf.keras.activations.sigmoid(t) for t in [self.y__, y_1, y_2, y_3]], axis=-1)
        else:
            outputs['y_softmax']   = tf.keras.activations.softmax(self.y__)     
            outputs['y_sigmoid']   = tf.keras.activations.sigmoid(self.y__)
        outputs['logits']          = self.y__
        outputs['y_']              = self.y_
    
        return outputs

    def summary(self):
        # Model Summary
        print('Input Volume:-----------------------------------------------', self.inputs.shape)
        print('Initial Convolutional Layer (Stage 0):----------------------', self.x.shape)
        print('Attention Gating: Stage 0:----------------------------------', self.att_conv0.shape)
        print('Encoder: Stage 1; SE-Residual Block:------------------------', self.conv1.shape)
        print('Attention Gating: Stage 1:----------------------------------', self.att_conv1.shape)
        print('Encoder: Stage 2; SE-Residual Block:------------------------', self.conv2.shape)
        print('Attention Gating: Stage 2:----------------------------------', self.att_conv2.shape)    
        print('Encoder: Stage 3; SE-Residual Block:------------------------', self.conv3.shape)
        print('Attention Gating: Stage 3:----------------------------------', self.att_conv3.shape)    
        print('Middle: High-Dim Latent Features:---------------------------', self.convm.shape)
        print('Decoder: Stage 3; Nested U-Net Concat.:---------------------', self.uconv3_.shape)
        print('Decoder: Stage 3; Nested U-Net End:-------------------------', self.uconv3.shape)
        print('Decoder: Stage 2; Nested U-Net Concat.:---------------------', self.uconv2_.shape)
        print('Decoder: Stage 2; Nested U-Net End:-------------------------', self.uconv2.shape)
        print('Decoder: Stage 1; Nested U-Net Concat.:---------------------', self.uconv1_.shape)
        print('Decoder: Stage 1; Nested U-Net End:-------------------------', self.uconv1.shape)    
        print('Decoder: Stage 0; Nested U-Net Concat.:---------------------', self.uconv0_.shape)
        print('Decoder: Stage 0; Nested U-Net End:-------------------------', self.uconv0.shape)    
        if not self.probabilistic:
            print('Prob. 3D U-Net (Type: M1) [Logits]:---------------------------', self.y__.shape)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------












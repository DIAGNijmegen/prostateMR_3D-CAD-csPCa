# Clinically Significant Prostate Cancer Detection in bpMRI

**Note**: This repo will be continually updated upon future advancements and we welcome open-source contributions! Currently, it shares the open-source TensorFlow 2.4 version of the Dual Attention U-Net (*M1*), as introduced in the publication(s) listed below. Source code (and the anatomical prior) used for training this model, as per our original setup, have a large number of dependencies on internal datasets, tooling, infrastructure and hardware, and their release is currently not feasible. However, an equivalent MWE adaptation will soon be made available. We encourage users to test out *M1*, identify potential areas for significant improvement, update its architecture accordingly and propose changes for inclusion to this repo.

**Pre-Trained Model using 1950 bpMRI with [PI-RADS v2](https://www.sciencedirect.com/science/article/pii/S0302283815008489?via%3Dihub) Annotations [80:20 - Training:Validation Ratio]:** To infer lesion predictions on testing samples using the pre-trained variant of the algorithm, please visit: https://grand-challenge.org/algorithms/prostate-mri-cad-cspca/

**Related U-Net Architectures:**  
  ● UNet++: https://github.com/MrGiovanni/UNetPlusPlus  
  ● Attention U-Net: https://github.com/ozan-oktay/Attention-Gated-Networks  
  ● nnU-Net: https://github.com/MIC-DKFZ/nnUNet  

<kbd>![schematic](docs/image-1.png)</kbd>

**Model Setup Example in TensorFlow 2.4:**  
*(Reference: [Training CNNs in TF2: Walkthrough](https://www.tensorflow.org/tutorials/images/cnn) )*
```python
# U-Net Model Definition (Hyperparameters are Data-Centric and Require Adequate Tuning for Optimal Performance)
unet_model = models.networks.M1(input_spatial_dims =  (18,192,192),            
                                input_channels     =   3,
                                num_classes        =   2,                       
                                filters            =  (32,64,128,256,512),   
                                strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2)),  
                                kernel_sizes       = ((1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3)),  
                                dropout_rate       =   0.50,       
                                dropout_mode       =  'standard',
                                se_reduction       =  (8,8,8,8,8),
                                att_sub_samp       = ((1,1,1),(1,1,1),(1,1,1)),
                                kernel_initializer =   tf.keras.initializers.Orthogonal(gain=1.0), 
                                bias_initializer   =   tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
                                kernel_regularizer =   tf.keras.regularizers.l2(1e-4),
                                bias_regularizer   =   tf.keras.regularizers.l2(1e-4),     
                                cascaded           =   False)  

# Compile Model w/ Optimizer and Loss Function(s)
unet_model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-4, amsgrad=True), 
                   loss      = models.losses.Focal(alpha=0.75, gamma=2.00).loss)

# Display Model Summary
unet_model.summary()

# Train Model
unet_model.fit(...)
```

**Related Publications:**  
● [A. Saha, M. Hosseinzadeh, H. Huisman (2021), "End-to-End Prostate Cancer Detection in bpMRI via 3D CNNs: Effect of Attention Mechanisms, Clinical Priori and Decoupled False
  Positive Reduction", *Under Review at MedIA: Medical Image Analysis*.](https://arxiv.org/abs/2101.03244)

● [A. Saha, M. Hosseinzadeh, H. Huisman (2020), "Encoding Clinical Priori in 3D Convolutional Neural Networks for Prostate Cancer Detection in bpMRI", Medical Imaging Meets
  NeurIPS Workshop – 34th Conference on Neural Information Processing Systems (NeurIPS), Vancouever, Canada.](https://arxiv.org/abs/2011.00263)
  
**Contact:** anindo@ieee.org; matin.hosseinzadeh@radboudumc.nl 




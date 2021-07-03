# Clinically Significant Prostate Cancer Detection in bpMRI

**Note**: This repo will be continually updated upon future advancements and we welcome open-source contributions! Currently, it shares the open-source TensorFlow 2.4 version of the deep attention-driven 3D U-Net (Type: *M1*), as introduced in the publication(s) listed below. Source code (and the anatomical prior) used for training this model, as per our original setup, carry a large number of dependencies on internal datasets, tooling, infrastructure and hardware, and their release is currently not feasible. However, an equivalent MWE adaptation will soon be made available. We encourage users to test out *M1*, identify potential areas for significant improvement and propose PRs for inclusion to this repo.

**Pre-Trained Model using 1950 bpMRI with [PI-RADS v2](https://www.sciencedirect.com/science/article/pii/S0302283815008489?via%3Dihub) Annotations [Training:Validation Ratio - 80:20]:**  
To infer lesion predictions on testing samples using the pre-trained variant of this algorithm, please visit https://grand-challenge.org/algorithms/prostate-mri-cad-cspca/

**Related U-Net Architectures:**  
  ● UNet++: https://github.com/MrGiovanni/UNetPlusPlus  
  ● Attention U-Net: https://github.com/ozan-oktay/Attention-Gated-Networks  
  ● nnU-Net: https://github.com/MIC-DKFZ/nnUNet  

<kbd>![schematic](docs/image-1.png)</kbd>

**Related Publications:**  
● [A. Saha, M. Hosseinzadeh, H. Huisman (2021), "End-to-End Prostate Cancer Detection in bpMRI via 3D CNNs: Effect of Attention Mechanisms, Clinical Priori and Decoupled False
  Positive Reduction", Medical Image Analysis:102155.](https://doi.org/10.1016/j.media.2021.102155)

● [A. Saha, M. Hosseinzadeh, H. Huisman (2020), "Encoding Clinical Priori in 3D Convolutional Neural Networks for Prostate Cancer Detection in bpMRI", Medical Imaging Meets
  NeurIPS Workshop – 34th Conference on Neural Information Processing Systems (NeurIPS), Vancouever, Canada.](https://arxiv.org/abs/2011.00263)

**Minimal Example of Model Setup in TensorFlow 2.4:**  
*(Reference: [Training CNNs in TF2: Walkthrough](https://www.tensorflow.org/tutorials/images/segmentation); [TF2 Datasets: Best Practices](https://www.tensorflow.org/guide/data_performance))*
```python

# Define Basic Hyperparameters
NUM_EPOCHS        =   150
TRAIN_SAMPLES     =   2500
BATCH_SIZE        =   10
AUGM_PARAMS       =  [1.00, # Probability of Undergoing Augmentation
                      0.15, # ± Maximum X-Y Translation Factor
                      7.50, # ± Maximum Rotation in Degrees
                      True, # Enable Horizontal Flip
                      1.15, # Maximum Zoom-In Factor
                      1.00] # Maximum StdDev of Additive Gaussian Noise
                      
# Expected Input/Image and Label/Detection Data Type+Shape
EXPECTED_IO_TYPE  = ({"image":      tf.float32}, 
                     {"detection":  tf.float32})
EXPECTED_IO_SHAPE = ({"image":     (20,160,160)+(3,)}, 
                     {"detection": (20,160,160)+(2,)})

# Initialize TensorFlow Dataset, Cache on Remote Server (Optional), Map Parallelized Data Augmentation
train_gen = tf.data.Dataset.from_generator(lambda:'''PLACE_NUMPY_DATA_GENERATOR''', 
                                           output_types  = EXPECTED_IO_TYPE, 
                                           output_shapes = EXPECTED_IO_SHAPE)
                                           
train_gen = train_gen.cache(filename='''ENTER_PATH_TO_CACHE_FILE''')     
train_gen = train_gen.map(lambda x,y: models.augmentations.augment_tensors(x,y,AUGM_PARAMS,False,True), 
                                                       num_parallel_calls=multiprocessing.cpu_count())
                                                                               
train_gen = train_gen.repeat()                               # Repeat Samples Upon Exhaustion
train_gen = train_gen.shuffle(4*BATCH_SIZE)                  # Shuffle Samples with Buffer Size of Batch Size
train_gen = train_gen.batch(BATCH_SIZE)                      # Load Data in Batches
train_gen = train_gen.prefetch(buffer_size=tf.data.AUTOTUNE) # Prefetch Data via CPU while GPU is Training

# U-Net Model Definition (Note: Hyperparameters are Data-Centric -> Adequate Tuning for Optimal Performance)
unet_model = models.networks.M1(\
                          input_spatial_dims =  (20,160,160),            
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
                          bias_initializer   =   tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1e-3),
                          kernel_regularizer =   tf.keras.regularizers.l2(1e-4),
                          bias_regularizer   =   tf.keras.regularizers.l2(1e-4),     
                          cascaded           =   False)  

# Schedule Cosine Annealing Learning Rate with Warm Restarts
LR_SCHEDULE = tf.keras.optimizers.schedules.CosineDecayRestarts(\
                          initial_learning_rate=1e-3, t_mul=2.00, m_mul=1.00, alpha=1e-3,
                          first_decay_steps=int(np.ceil(((TRAIN_SAMPLES)/BATCH_SIZE)))*10)
                                                  
# Compile Model w/ Optimizer and Loss Function(s)
unet_model.compile(optimizer = tf.keras.optimizers.Adam(lr=LR_SCHEDULE, amsgrad=True), 
                   loss      = models.losses.Focal(alpha=0.75, gamma=2.00).loss)

# Display Model Summary
unet_model.summary()

# Train Model
unet_model.fit(x=train_gen, epochs=NUM_EPOCHS, steps_per_epoch=int(np.ceil(((TRAIN_SAMPLES)/BATCH_SIZE))))
```
  
**Contact:** anindo@ieee.org; matin.hosseinzadeh@radboudumc.nl 




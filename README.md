# Clinically Significant Prostate Cancer Detection in bpMRI

**Related Publications:**  
● [A. Saha, M. Hosseinzadeh, H. Huisman (2021), "End-to-End Prostate Cancer Detection in bpMRI via 3D CNNs: Effect of Attention Mechanisms, Clinical Priori and Decoupled False
  Positive Reduction", Medical Image Analysis:102155.](https://doi.org/10.1016/j.media.2021.102155) [(commit 58b784f)](https://github.com/DIAGNijmegen/prostateMR_3D-CAD-csPCa/tree/58b784ffbd2e8c89139c6773cb9490b2fd53d814)

● [A. Saha, M. Hosseinzadeh, H. Huisman (2020), "Encoding Clinical Priori in 3D Convolutional Neural Networks for Prostate Cancer Detection in bpMRI", Medical Imaging Meets
  NeurIPS Workshop – 34th Conference on Neural Information Processing Systems (NeurIPS), Vancouever, Canada.](https://arxiv.org/abs/2011.00263) [(commit 58b784f)](https://github.com/DIAGNijmegen/prostateMR_3D-CAD-csPCa/tree/58b784ffbd2e8c89139c6773cb9490b2fd53d814)

**Note**: This repo will be continually updated upon future advancements and we welcome open-source contributions! Currently, it shares the open-source TensorFlow 2.5 version of the deep attention-driven `3D U-Net (Type:M1)`, as introduced in the publication(s) listed above. Source code used for training this model, as per our original setup, carry a large number of dependencies on internal datasets, tooling, infrastructure and hardware, and their release is currently not feasible. However, an equivalent MWE adaptation will soon be made available. We encourage users to test out `M1`, identify potential areas for significant improvement and propose PRs for inclusion to this repo.

**Pre-Trained Model using 1950 bpMRI with [PI-RADS v2](https://www.sciencedirect.com/science/article/pii/S0302283815008489?via%3Dihub) Annotations [Training:Validation Ratio - 80:20]:**  
To infer lesion predictions on testing samples using the pre-trained variant [(commit 58b784f)](https://github.com/DIAGNijmegen/prostateMR_3D-CAD-csPCa/tree/58b784ffbd2e8c89139c6773cb9490b2fd53d814) of this algorithm, please visit https://grand-challenge.org/algorithms/prostate-mri-cad-cspca/

**Related U-Net Architectures:**  
  ● Probabilistic U-Net: https://github.com/SimonKohl/probabilistic_unet  
  ● nnU-Net: https://github.com/MIC-DKFZ/nnUNet  
  ● Attention U-Net: https://github.com/ozan-oktay/Attention-Gated-Networks  
  ● UNet++: https://github.com/MrGiovanni/UNetPlusPlus  

<kbd>![schematic](docs/image-1.png)</kbd>
Train-time schematic for the Bayesian/probabilistic configuration of `3D U-Net (Type:M1)`. `L_KL` denotes the Kullback–Leibler divergence loss between prior distribution `P` and posterior distribution `Q`. `L_S` denotes the segmentation loss between prediction `p` and ground-truth `Y`. For each execution of the model, one sample `z ∈ Q` (train-time) or `z ∈ P` (test-time) is drawn to predict one segmentation mask `p`.

<kbd>![schematic](docs/image-2.png)</kbd>
Architecture schematic of the deep attention-driven `3D U-Net (Type:M1)`.

**Minimal Example of Model Setup in TensorFlow 2.5:**  
*(Reference: [Training CNNs in TF2: Walkthrough](https://www.tensorflow.org/tutorials/images/segmentation); [TF2 Datasets: Best Practices](https://www.tensorflow.org/guide/data_performance))*
```python
# U-Net Definition (Note: Hyperparameters are Data-Centric -> Require Adequate Tuning for Optimal Performance)
unet_model = models.networks.M1(\
                        input_spatial_dims =  (20,160,160),            
                        input_channels     =   3,
                        num_classes        =   2,                       
                        filters            =  (32,64,128,256,512),   
                        strides            = ((1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2)),  
                        kernel_sizes       = ((1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3)),
                        proba_event_shape  =   512
                        dropout_rate       =   0.50,       
                        dropout_mode       =  'monte-carlo',
                        se_reduction       =  (8,8,8,8,8),
                        att_sub_samp       = ((1,1,1),(1,1,1),(1,1,1),(1,1,1)),
                        kernel_initializer =   tf.keras.initializers.Orthogonal(gain=1), 
                        bias_initializer   =   tf.keras.initializers.TruncatedNormal(mean=0, stddev=1e-3),
                        kernel_regularizer =   tf.keras.regularizers.l2(1e-4),
                        bias_regularizer   =   tf.keras.regularizers.l2(1e-4),     
                        cascaded           =   False,
                        probabilistic      =   True,
                        deep_supervision   =   True)  
                                       
# Schedule Cosine Annealing Learning Rate with Warm Restarts
LR_SCHEDULE = (tf.keras.optimizers.schedules.CosineDecayRestarts(\
                        initial_learning_rate=1e-3, t_mul=2.00, m_mul=1.00, alpha=1e-3,
                        first_decay_steps=int(np.ceil(((TRAIN_SAMPLES)/BATCH_SIZE)))*10))
                                                  
# Compile Model w/ Optimizer and Loss Function(s)
unet_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=LR_SCHEDULE, amsgrad=True), 
                   loss      = models.losses.Focal(alpha=0.75, gamma=2.00).loss)

# Train Model
unet_model.fit(...)
```
**Contact:** anindo@ieee.org; matin.hosseinzadeh@radboudumc.nl 




# Clinically Significant Prostate Cancer Detection in bpMRI

**Authors**: Anindo Saha, Matin Hosseinzadeh, Henkjan Huisman

**Note**: This repo is currently under development and shares the open-source TensorFlow 1.15 version of the Dual Attention U-Net (*M1*) and the SEResNet (*M2*) architectures, as detailed in the publications listed below. Code (and the probabilistic prior) used for training the models have a large number of dependencies on internal datasets, tooling, infrastructure and hardware, and their release is currently not feasible. However, an equivalent MWE adaptation in TensorFlow 2.4 will soon be made available. To infer lesion predictions on testing samples using the pre-trained variant of the algorithm, please visit: https://grand-challenge.org/algorithms/prostate-mri-cad-cspca/ 

**Related Models:**  
  ● UNet++: https://github.com/MrGiovanni/UNetPlusPlus  
  ● Attention U-Net: https://github.com/ozan-oktay/Attention-Gated-Networks  
  ● nnU-Net: https://github.com/MIC-DKFZ/nnUNet  


**Related Publications:**  
● [A. Saha, M. Hosseinzadeh, H. Huisman (2021), "End-to-End Prostate Cancer Detection in bpMRI via 3D CNNs: Effect of Attention Mechanisms, Clinical Priori and Decoupled False
  Positive Reduction", *Under Review at MedIA: Medical Image Analysis*.](https://arxiv.org/abs/2101.03244)

● [A. Saha, M. Hosseinzadeh, H. Huisman (2020), "Encoding Clinical Priori in 3D Convolutional Neural Networks for Prostate Cancer Detection in bpMRI", Medical Imaging Meets
  NeurIPS Workshop – 34th Conference on Neural Information Processing Systems (NeurIPS), Vancouever, Canada.](https://arxiv.org/abs/2011.00263)
  




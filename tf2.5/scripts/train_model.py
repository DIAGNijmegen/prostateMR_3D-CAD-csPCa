from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import SimpleITK as sitk
import os
import numpy as np
import scipy.ndimage
import time
import os
import cv2
import argparse
import json
import pandas as pd
from skimage.measure import regionprops
from scipy.stats import entropy
from shutil import copyfile, rmtree
import tensorflow as tf
import model.losses as losses
from model.augmentations import augment_tensors
import model.unets as unets
from callbacks import WeightsSaver, ResumeTraining
from data_generators import custom_data_generator
from misc import setup_device, print_overview
import warnings
import multiprocessing
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

'''
Prostate Cancer Detection or Zonal Segmentation in MRI
Script:         Train-Time Callbacks
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Task 1: Benign(0), Malignant(1)
                Task 2: Whole-Gland(0), Transitional Zone(1),
                        Peripheral Zone (2)
Update:         03/10/2021

'''

# Command Line Arguments for Hyperparameters and I/O Paths
prsr = argparse.ArgumentParser(description='Command Line Arguments for Training Script')
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset Definition 
prsr.add_argument('--TRAIN_OBJ',                  type=str,   default='lesion',                                                            help="Training Objective: 'zonal'/'lesion'")
prsr.add_argument('--NAME',                       type=str,                                                                                help='Path to Load/Store Model Weights and Performance Metrics')
prsr.add_argument('--NUM_EPOCHS',                 type=int,   default=100,                                                                 help="Number of Training Epochs")
prsr.add_argument('--FOLDS',                      type=int,   default=[0,1,2,3,4],                                              nargs='+', help="Folds Selected For Training")
prsr.add_argument('--TRAIN_XLSX_PREFIX',          type=str,   default='./models/2021/medneurips2021/data_feed/prostateX_200_train-fold-',  help="Path+Prefix to Training Fold Files")
prsr.add_argument('--VALID_XLSX_PREFIX',          type=str,   default='./models/2021/medneurips2021/data_feed/prostateX_200_valid-fold-',  help="Path+Prefix to Validation Fold Files")
prsr.add_argument('--WEIGHTS_DIR',                type=str,   default='./models/2021/medneurips2021/weights/',                             help="Path to Load/Store Model Weights")
prsr.add_argument('--METRICS_DIR',                type=str,   default='./models/2021/medneurips2021/weights/',                             help="Path to Load/Store Performance Metrics")
prsr.add_argument('--USE_PRETRAINED_WEIGHTS',     type=str,   default=False,                                                               help="Path to Pretrained Weights or 'False' (Optional)")
prsr.add_argument('--FREEZE_LAYERS',              type=int,   default=9999,                                                                help="Freeze First N Layers when (USE_PRETRAINED_WEIGHTS!=9999) [e.g. 184]")
prsr.add_argument('--WEIGHTS_MIN_EPOCH',          type=int,   default=50,                                                                  help="Minimum Epoch to Start Exporting Weights")
prsr.add_argument('--VALIDATE_PER_N_EPOCHS',      type=int,   default=5,                                                                   help="Validate Model Performance Every N Epochs")
prsr.add_argument('--STORE_WEIGHTS_PER_N_EPOCHS', type=int,   default=5,                                                                   help="Store Weights Every N Epochs")
prsr.add_argument('--WEIGHTS_OVERWRITE',          type=bool,  default=False,                                                               help="Store All Weights or Most Recent One")
prsr.add_argument('--VALIDATE_MIN_EPOCH',         type=int,   default=0,                                                                   help="Minimum Epoch to Start Validation")
prsr.add_argument('--SHOW_SUMMARY',               type=bool,  default=True,                                                                help="Display Overview")
prsr.add_argument('--RESUME_TRAIN',               type=bool,  default=False,                                                               help="Enable Resume Training (Experimental)")
prsr.add_argument('--CACHE_TDS_PATH',             type=str,   default=None,                                                                help="Path to TensorFlow Data Cache for Faster I/O or 'False' (Optional)")
prsr.add_argument('--GPU_DEVICE_IDs',             type=str,   default="0",                                                                 help="Number of GPUs Available for Computation")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# U-Net Hyperparameters
prsr.add_argument('--UNET_DEEP_SUPERVISION',      type=bool,  default=False,                                                      help="U-Net: Enable Deep Supervision")
prsr.add_argument('--UNET_PROBABILISTIC',         type=bool,  default=False,                                                      help="U-Net: Enable Probabilistic/Bayesian Output Computation")
prsr.add_argument('--UNET_PROBA_EVENT_SHAPE',     type=int,   default=320,                                                        help="U-Net: Probabilistic Latent Distribution Size")
prsr.add_argument('--UNET_PROBA_ITER',            type=int,   default=1,                                                          help="U-Net: Iterations of Probabilistic Inference During Validation")
prsr.add_argument('--UNET_FEATURE_CHANNELS',      type=int,   default=[32,64,128,256,512],                             nargs='+', help="U-Net: Encoder/Decoder Channels")
prsr.add_argument('--UNET_STRIDES',               type=int,   default=[(1,1,1),(1,2,2),(1,2,2),(2,2,2),(2,2,2)],       nargs='+', help="U-Net: Down/Upsampling Factor per Resolution")
prsr.add_argument('--UNET_KERNEL_SIZES',          type=int,   default=[(1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3)],       nargs='+', help="U-Net: Convolution Kernel Sizes")
prsr.add_argument('--UNET_ATT_SUBSAMP',           type=int,   default=[(1,1,1),(1,1,1),(1,1,1),(1,1,1)],               nargs='+', help="U-Net: Attention Gate Subsampling Factor")
prsr.add_argument('--UNET_SE_REDUCTION',          type=int,   default=[8,8,8,8,8],                                     nargs='+', help="U-Net: Squeeze-and-Excitation Reduction Ratio")
prsr.add_argument('--UNET_KERNEL_REGULARIZER_L2', type=float, default=1e-5,                                                       help="U-Net: L2 Kernel Regularizer (Contributes to Total Loss at Train-Time)")
prsr.add_argument('--UNET_BIAS_REGULARIZER_L2',   type=float, default=1e-5,                                                       help="U-Net: L2 Bias Regularizer (Contributes to Total Loss at Train-Time)")
prsr.add_argument('--UNET_DROPOUT_MODE',          type=str,   default="monte-carlo",                                              help="U-Net: Dropout Mode: 'standard'/'monte-carlo'")
prsr.add_argument('--UNET_DROPOUT_RATE',          type=float, default=0.33,                                                       help="U-Net: Dropout Regularization Rate")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Training Hyperparameters
prsr.add_argument('--BATCH_SIZE',          type=int,   default=1,                                                                         help="Batch Size")
prsr.add_argument('--BASE_LR',             type=float, default=1e-3,                                                                      help="Base Learning Rate")
prsr.add_argument('--LR_MODE',             type=str,   default="CALR",                                                                    help="Learning Rate Mode: 'CLR'/'CALR'")
prsr.add_argument('--CALR_PARAMS',         type=float, default=[2.00, 1.00, 1e-3],                                                        help="'CosineDecayRestarts': t_mul, m_mul, alpha")
prsr.add_argument('--CLR_PARAMS',          type=float, default=[5e-5, 1.00, 1.25],                                                        help="'CyclicLR': Max LR, Decay Factor, Step Factor")
prsr.add_argument('--OPTIMIZER',           type=str,   default="adam",                                                                    help="Optimizer: 'adam'/'momentum'")
prsr.add_argument('--LOSS_MODE',           type=str,   default="distribution_focal",                                                      help="Loss: 'distribution_focal'/'region_boundary'")
prsr.add_argument('--FOCAL_LOSS_ALPHA',    type=float, default=[1.00, 1.00],                                                   nargs='+', help="Focal Loss (alpha)")
prsr.add_argument('--FOCAL_LOSS_GAMMA',    type=float, default=2.0,                                                                       help="Focal Loss (gamma). Note: When gamma=0; FL reduces down to CE/BCE.")
prsr.add_argument('--DSC_BD_LOSS_WEIGHTS', type=float, default=[0.50, 0.50],                                                              help="Soft Dice + Boundary Loss (weights)")
prsr.add_argument('--ELBO_LOSS_PARAMS',    type=float, default=[1.0],                                                                     help="Evidence Lower Bound Loss for Prob Dist. (weight)")
prsr.add_argument('--AUGM_PARAMS',         type=float, default=[1.00, 0.25, 0.15, 10.0, True, 1.20, 0.15, 0.025, True, [0.50, 1.50]],     help="Train-Time Augmentations (M_PROB,TX_PROB,TRANS,ROT,HFLIP,SCALE,\
                                                                                                                                                                          NOISE,C_SHIFT,POOR_QUAL,GAMMA)")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
args, _ = prsr.parse_known_args()

# For Each Fold
for f in args.FOLDS:
    # Verify Whether Training Had Completed (Yes -> Jump to Next Fold; No -> Resume/Restart Training)
    if os.path.isfile(args.WEIGHTS_DIR+args.NAME+'/F'+str(f+1)+'/model_weights_'+str({args.NUM_EPOCHS})+'.h5'): continue
    else:                                                                                                       pass
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Dataset Definition 
    TRAIN_XLSX         =  args.TRAIN_XLSX_PREFIX+str(f+1)+'.xlsx'        # Paths to Training Scans/Labels
    VALID_XLSX         =  args.VALID_XLSX_PREFIX+str(f+1)+'.xlsx'        # Paths to Validation Scans/Labels
    TRAIN_DATA_SAMPLES =  len(pd.read_excel(TRAIN_XLSX)['image_path'])   # Number of Training Samples
    VALID_DATA_SAMPLES =  len(pd.read_excel(VALID_XLSX)['image_path'])   # Number of Validation Samples
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Cosine Annealing Learning Rate (Cosine Decay w/ Warm Restarts)
    if (args.LR_MODE=='CALR'): 
      BASE_LR = (tf.keras.optimizers.schedules.CosineDecayRestarts(\
                initial_learning_rate=args.BASE_LR, first_decay_steps=int(np.ceil(((TRAIN_DATA_SAMPLES)/args.BATCH_SIZE)))*args.NUM_EPOCHS,
                t_mul=args.CALR_PARAMS[0], m_mul=args.CALR_PARAMS[1], alpha=args.CALR_PARAMS[2]))
    else: BASE_LR = args.BASE_LR

    # Optimizer Setup
    if   (args.OPTIMIZER=='adam'):     OPTIMIZER_SET = tf.keras.optimizers.Adam(learning_rate=args.BASE_LR, amsgrad=True)
    elif (args.OPTIMIZER=='momentum'): OPTIMIZER_SET = tf.keras.optimizers.SGD(learning_rate=args.BASE_LR,  nesterov=True, momentum=0.90)
    
    # Segmentation/Detection Loss Function Setup
    if   (args.LOSS_MODE=='distribution_focal'): LOSSES = [losses.Focal(alpha=args.FOCAL_LOSS_ALPHA, gamma=args.FOCAL_LOSS_GAMMA).loss]   
    elif (args.LOSS_MODE=='region_boundary'):    LOSSES = [losses.SoftDicePlusBoundarySurface(loss_weights=args.DSC_BD_LOSS_WEIGHTS).loss]   
    LOSS_WEIGHTS    = [1.00]

    # Loss Function for Probabilistic Setup
    if args.UNET_PROBABILISTIC:
      LOSSES       +=  [losses.EvidenceLowerBound().loss]
      LOSS_WEIGHTS +=  [args.ELBO_LOSS_PARAMS[0]]
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Display Overview of Training Configuration
    print_overview(args)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Load Python Data Generators
    print("Loading Training + Validation Data into RAM...")
    train_data_gen = custom_data_generator(data_xlsx=TRAIN_XLSX, train_obj=args.TRAIN_OBJ, probabilistic=args.UNET_PROBABILISTIC)
    train_metrics  = custom_data_generator(data_xlsx=TRAIN_XLSX, train_obj=args.TRAIN_OBJ, probabilistic=args.UNET_PROBABILISTIC)
    valid_data_gen = custom_data_generator(data_xlsx=VALID_XLSX, train_obj=args.TRAIN_OBJ, probabilistic=args.UNET_PROBABILISTIC)
    print("Complete.")
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Assert Input Dimensions and Data Types via TensorFlow Datasets
    IMAGE_SPATIAL_DIMS =  np.load(pd.read_excel(TRAIN_XLSX)['image_path'][0])[...,0].shape   #  Spatial Dimensions of Input MRI (D,H,W)
    IMAGE_NUM_CHANNELS = (3 if args.TRAIN_OBJ=='lesion' else 1)                              # 'lesion':{T2W,DWI,ADC},'zonal':{T2W}
    NUM_CLASSES        = (2 if args.TRAIN_OBJ=='lesion' else 3)                              # 'lesion':{BG,csPCa},'zonal':{WG,TZ,PZ}

    if ((args.LOSS_MODE)=='distribution_focal')&(len(args.FOCAL_LOSS_ALPHA)!=NUM_CLASSES):
        raise Exception("Number of Class Weights Declared in Loss Function != Number of Classes in Labels/Loss Objective")

    if args.UNET_PROBABILISTIC: IMAGE_NUM_CHANNELS += NUM_CLASSES-1
    
    if args.UNET_PROBABILISTIC:
        EXPECTED_IO_TYPE  = ({"image":     tf.float32}, 
                             {"detection": tf.float32,
                              "KL":        tf.float32})
        EXPECTED_IO_SHAPE = ({"image":     IMAGE_SPATIAL_DIMS+(IMAGE_NUM_CHANNELS,)}, 
                             {"detection": IMAGE_SPATIAL_DIMS+(NUM_CLASSES,),
                              "KL":        IMAGE_SPATIAL_DIMS+(NUM_CLASSES,)})
    else:
        EXPECTED_IO_TYPE  = ({"image":     tf.float32}, 
                             {"detection": tf.float32})
        EXPECTED_IO_SHAPE = ({"image":     IMAGE_SPATIAL_DIMS+(IMAGE_NUM_CHANNELS,)}, 
                             {"detection": IMAGE_SPATIAL_DIMS+(NUM_CLASSES,)})
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # TensorFlow GPU Handling + Datasets
    devices, num_devices         = setup_device(args.GPU_DEVICE_IDs)
    if (num_devices>1): strategy = tf.distribute.MirroredStrategy(devices).scope()
    else:               strategy = tf.device(devices)
    assert np.mod(args.BATCH_SIZE, num_devices)==0, 'Batch size (%d) should be a multiple of the number of GPUs (%d).'%(BATCH_SIZE,num_devices)
    print("GPU Device(s):", devices)
    
    # Switch I/O to TensorFlow Datasets
    print("Switching I/O to TensorFlow Datasets...")
    train_gen     = tf.data.Dataset.from_generator(lambda:train_data_gen, output_types  = EXPECTED_IO_TYPE, 
                                                                          output_shapes = EXPECTED_IO_SHAPE)       # Initialize TensorFlow Dataset
    if str(args.CACHE_TDS_PATH)!='None': 
        train_gen = train_gen.cache(filename=(None if str(args.CACHE_TDS_PATH)=='None' else args.CACHE_TDS_PATH))  # Cache Dataset on Remote Server
    train_gen     = train_gen.shuffle(int(TRAIN_DATA_SAMPLES*0.50))                                                           # Shuffle Samples
    train_gen     = train_gen.map(lambda x,y: augment_tensors(x,y,args.AUGM_PARAMS,args.TRAIN_OBJ), 
                                                              num_parallel_calls=multiprocessing.cpu_count())
    train_gen     = train_gen.batch(args.BATCH_SIZE)                                                               # Load Data in Batches
    train_gen     = train_gen.prefetch(buffer_size=tf.data.AUTOTUNE)                                               # Prefetch Data via CPU while GPU is Training
    print("Complete.")
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Model Training/Validation
    with strategy:
        # U-Net Definition
        unet_model = unets.networks.M1(input_spatial_dims = IMAGE_SPATIAL_DIMS,          input_channels    = IMAGE_NUM_CHANNELS,
                                       num_classes        = NUM_CLASSES,                 filters           = args.UNET_FEATURE_CHANNELS,                            
                                       dropout_rate       = args.UNET_DROPOUT_RATE,      strides           = args.UNET_STRIDES,
                                       kernel_sizes       = args.UNET_KERNEL_SIZES,      dropout_mode      = args.UNET_DROPOUT_MODE,
                                       se_reduction       = args.UNET_SE_REDUCTION,      att_sub_samp      = args.UNET_ATT_SUBSAMP,
                                       probabilistic      = args.UNET_PROBABILISTIC,     proba_event_shape = args.UNET_PROBA_EVENT_SHAPE,
                                       deep_supervision   = args.UNET_DEEP_SUPERVISION,  summary           = args.SHOW_SUMMARY,
                                       bias_initializer   = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=8),   
                                       bias_regularizer   = tf.keras.regularizers.l2(args.UNET_BIAS_REGULARIZER_L2),                                               
                                       kernel_initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=8), 
                                       kernel_regularizer = tf.keras.regularizers.l2(args.UNET_KERNEL_REGULARIZER_L2))  
        
        # Display Number of Layers and Definition of Frozen Layers (If Any)
        print("Number of Model Layers: ", len(unet_model.layers))        
        if args.FREEZE_LAYERS!=9999:
            for layer in unet_model.layers[:args.FREEZE_LAYERS]: layer.trainable = False
            print("Trainable Layers: ", len(unet_model.layers)-args.FREEZE_LAYERS)
            for layer in unet_model.layers: 
                if layer.trainable==True: print(layer, layer.trainable)
    
        # Load Pre-Trained Weights
        if str(args.USE_PRETRAINED_WEIGHTS)!='False': 
          unet_model = unets.networks.M1.load(path=args.USE_PRETRAINED_WEIGHTS)

        # Restart/Resume Training
        if args.RESUME_TRAIN:           
          unet_model, init_epoch = ResumeTraining(model=unet_model, weights_dir=args.WEIGHTS_DIR+args.NAME+'/F'+str(f+1))
        else:        
          init_epoch             = 0
          if os.path.exists(args.WEIGHTS_DIR+args.NAME+'/F'+str(f+1)): 
            raise Exception("Target Folder Already Exists! Either Remove It or Enable 'RESUME_TRAIN'.")
          else: os.makedirs(args.WEIGHTS_DIR+args.NAME+'/F'+str(f+1))

        # Compile Model w/ Hyperparameters, Optimizer, Loss Functions
        unet_model.compile(optimizer=OPTIMIZER_SET, loss=LOSSES, loss_weights=LOSS_WEIGHTS)
                                
        # Callbacks: Export Weights, Validate Model, Learning Rate Schedule
        callbacks  = [WeightsSaver(unet_model, 
                                weights_overwrite    =  args.WEIGHTS_OVERWRITE, 
                                weights_dir          =  args.WEIGHTS_DIR+args.NAME+'/F'+str(f+1), 
                                min_epoch            =  args.WEIGHTS_MIN_EPOCH,  
                                weights_num_epochs   =  args.STORE_WEIGHTS_PER_N_EPOCHS,
                                init_epoch           =  init_epoch)]

        if (args.LR_MODE=='CLR'):
          callbacks += [CyclicLR(mode                = 'exp_range', 
                                 max_lr              =  args.CLR_PARAMS[1], 
                                 gamma               =  args.CLR_PARAMS[2], 
                                 base_lr             =  BASE_LR, 
                                 step_size           = (round(TRAIN_SAMPLES)//args.BATCH_SIZE)*args.CLR_PARAMS[3])]
        
        if (args.TRAIN_OBJ=='zonal'):
          callbacks += [# TBA - CALLBACKS FOR TRAIN-TIME VALIDATION OF PERFORMANCE
                       ]

        if (args.TRAIN_OBJ=='lesion'):
          callbacks += [# TBA - CALLBACKS FOR TRAIN-TIME VALIDATION OF PERFORMANCE
                       ]
        # Train Model
        history = unet_model.fit(x                   =  train_gen, 
                                 epochs              =  args.NUM_EPOCHS,  
                                 steps_per_epoch     =  int(np.ceil(((TRAIN_DATA_SAMPLES)/args.BATCH_SIZE))),         
                                 initial_epoch       =  init_epoch,
                                 verbose             =  2,  
                                 callbacks           =  callbacks,
                                 use_multiprocessing =  True)
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

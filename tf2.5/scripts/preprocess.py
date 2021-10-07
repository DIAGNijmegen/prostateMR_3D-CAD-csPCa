import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import nibabel as nib
from dipy.align.imaffine import AffineMap



'''
Prostate Cancer Detection or Zonal Segmentation in MRI
Script:         Preprocessing Functions
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Task 1: Benign(0), Malignant(1)
                Task 2: Whole-Gland(0), Transitional Zone(1),
                        Peripheral Zone (2)
Update:         07/10/2021

'''




# Image Whitening (Mean=0; Standard Deviation=1) [Ref:DLTK]
def whitening(image, percentile=None):
    image = image.astype(np.float32)

    if (percentile!=None):
        image = np.clip(image, np.percentile(image,100-percentile), 
                               np.percentile(image,percentile))
    mean  = np.mean(image)
    std   = np.std(image)
    if std > 0: ret = (image - mean) / std
    else:       ret = image * 0.
    return ret

# Center Crop NumPy Volumes
def center_crop(img,cropz,cropx,cropy,center_2d_coords=None,multi_channel=False):
    if center_2d_coords: x,y = center_2d_coords
    else:                x,y = img.shape[1]//2,img.shape[2]//2
    startz = img.shape[0]//2    - (cropz//2)
    startx = int(x) - (cropx//2)
    starty = int(y) - (cropy//2)
    if (multi_channel==True): return img[startz:startz+cropz,startx:startx+cropx,starty:starty+cropy,:]
    else:                     return img[startz:startz+cropz,startx:startx+cropx,starty:starty+cropy]

# Resample Images to Target Resolution Spacing [Ref:SimpleITK]
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size    = itk_image.GetSize()
    
    out_size = [ int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                 int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                 int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    
    if is_label: resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:        resample.SetInterpolator(sitk.sitkBSpline)
    
    return resample.Execute(itk_image)

# Resize Image with Crop/Pad [Ref:DLTK]
def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    rank = len(img_size)  # Image Dimensions

    # Placeholders for New Shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding   = [[0, 0] for dim in range(rank)]
    slicer       = [slice(None)] * rank

    # For Each Dimension Determine Process (Cropping/Padding)
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]
        # Create Slicer Object to Crop/Leave Each Dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad Cropped Image to Extend Missing Dimension
    return np.pad(image[tuple(slicer)], to_padding, **kwargs)

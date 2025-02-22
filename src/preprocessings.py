"""This file contains all the functions required for the preprocessing of an .dcm imgage"""
import pandas as pd
import numpy as np
import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
import random
import torch
import scipy
import pydicom
import warnings
import monai
from monai.transforms import (
    LoadImage, Spacing, Resize, EnsureChannelFirst, Compose, ScaleIntensity
)

loader = LoadImage(image_only=False)  # Load image with metadata using monai
warnings.filterwarnings('ignore')

## function to convert values to HU
def get_HU(dicom_slices,images):
    stacked_images = images.copy()
    for slice in range(len(dicom_slices)):
        intercept = dicom_slices[slice].RescaleIntercept
        slope = dicom_slices[slice].RescaleIntercept

        ## if slope is less than 1 multiply it to the eac pixel value
        if slope != 1:
            #dicom_slices[slice] = np.int16(slope) * dicom_slices[slice]
            stacked_images[slice] = slope * stacked_images[slice].astype(np.float64)
            stacked_images[slice] = stacked_images[slice].astype(np.int16)
        
        ## add intercept to the slices
        #stacked_images[slice] += np.int16(intercept)
        stacked_images[slice] = stacked_images[slice].astype(np.int16) + np.int16(intercept)
    return stacked_images

## Function to adust pixel spacing
def adjust_pixel_space(dicom_slices,stacked_image,desired_spacing = [1,1,1]):
    image = stacked_image.copy()
    slice_thickness = dicom_slices[0].SliceThickness
    pixel_spacing = [float(dicom_slices[0].PixelSpacing[0]), float(dicom_slices[0].PixelSpacing[1])]
    resize_factor = np.array([slice_thickness] + pixel_spacing,dtype=np.float32)/desired_spacing

    new_shape = np.round(image.shape *resize_factor)

    new_resize_factor = new_shape / image.shape

    new_spacing = desired_spacing / new_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, new_spacing, mode='nearest')
    return image

def resize_slices(image_array, target_size=(512, 512)):
    ## Resize each slice of a 3D image to the given target size while preserving depth.
    depth = image_array.shape[0]
    resized_slices = np.zeros((depth, target_size[0], target_size[1]), dtype=image_array.dtype)

    for i in range(depth):
        resized_slices[i] = cv2.resize(image_array[i], target_size, interpolation=cv2.INTER_LINEAR)

    return resized_slices

def crop_and_resize_img(img, output_size=(512, 512)):
    """
    Crops an image to its non-zero region and resizes it to a fixed output size.
    
    Args:
        img (numpy.ndarray): Input 2D image array.
        output_size (tuple): Desired output size (width, height), default (512, 512).
    
    Returns:
        numpy.ndarray: Cropped and resized image.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    c1, c2 = False, False

    try:
        rmin, rmax = np.where(rows)[0][[0, -1]]
    except:
        rmin, rmax = 0, img.shape[0]
        c1 = True

    try:
        cmin, cmax = np.where(cols)[0][[0, -1]]
    except:
        cmin, cmax = 0, img.shape[1]
        c2 = True

    if c1 and c2:
        return np.zeros(output_size, dtype=img.dtype)  # Return blank image if empty

    cropped_img = img[rmin:rmax+1, cmin:cmax+1]  # Crop to non-zero region
    resized_img = cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_AREA)  # Resize to fixed size
    
    return resized_img


def preprocess(folder_having_dcm):
    """
    This function returns the fully preprocessed imge
    folder_having_dcm: should be the path to the folder where all the images in .dcm format are stored
    """
    if folder_having_dcm is None:
        raise ValueError("Path to the folder havinf .dcm images cannot be None")
    paths = f'{folder_having_dcm}/*.dcm'
    dcm_imgs = [pydicom.dcmread(sli) for sli in glob(paths)]
    #print(len(dcm_imgs))
    if len(dcm_imgs)==0:
        raise ValueError("Please provide valid path to the images in the .dcm format")
    
    ## Sorting the Images according to the Z-axis
    dcm_imgs.sort(key= lambda X: X.ImagePositionPatient[2])
    ## stacking the images
    stacked_img = np.stack([im.pixel_array for im in dcm_imgs])
    #print("Shape of the stacked Image:\t",stacked_img.shape)
    # ## HU ajusted image
    # new_stacked_img = get_HU(dcm_imgs,stacked_img)
    ## get the pixel adjusted image
    pixel_adjusted_img = adjust_pixel_space(dcm_imgs,stacked_img)
    #print("shape of pixel Adjusted image:\t",pixel_adjusted_img.shape)
    # Resize to (512,512)
    resized_croped_img = [crop_and_resize_img(i) for i in pixel_adjusted_img]
    # resized_image = resize_slices(pixel_adjusted_img)
    ## removing the slices with no image inside it
    filtered_array = []
    for indx in range(len(resized_croped_img)):
        if np.mean(resized_croped_img[indx]) >= 50.0:
            filtered_array.append(np.array(resized_croped_img[indx],dtype=np.float32))
    filtered_array = np.array(filtered_array)
    # return resized_image
    return filtered_array

def preprocess_monai(path_to_DICOM:str, fixed_slices:int = 64,fixed_size:tuple = (512, 512)):
    transforms = Compose([
    EnsureChannelFirst(),  # Converts slices into channels (C, H, W)
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),  # Resample to uniform voxel spacing
    ScaleIntensity(),  # Normalize intensity values
    Resize(spatial_size=(fixed_slices, *fixed_size), mode="trilinear"),  # Standardize shape
    ]) ## compose all the transforms in a transforms function
    image, meta = loader(path_to_DICOM)  # Loading the image inside the Provided path to DICOM folder
    return transforms(image).squeeze(0)  # Apply the transforms and remove the channel dimension

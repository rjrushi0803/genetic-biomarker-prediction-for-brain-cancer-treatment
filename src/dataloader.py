"""This is script contains the dataloader for the dataset"""
import pandas as pd
import numpy as np
import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
import random
import torch

import pydicom
#os.chdir('/mnt/nvme_disk2/User_data/nc36192d/rushi/Learnings/RSNA-MCCAI')
from preprocessings import preprocess,preprocess_monai

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from glob import glob

class MRIDataset(Dataset):
    def __init__(self,data_path:str,labels:list=None,scan_type:str = "FLAIR"):
        self.mode = "train"
        self.scan_type = scan_type
        self.path = data_path
        self.data_dirs = os.listdir(data_path)
        if labels != None:
            self.labels = np.array(labels,dtype=np.float32())

        self.x_train , self.x_val , self.y_train, self.y_val = None, None, None, None

    def split_data(self):
        if self.mode != "all":
            self.x_train , self.x_val , self.y_train, self.y_val = train_test_split(self.data_dirs , self.labels,test_size=0.20,random_state=42)
        else:
            pass
    
    def __len__(self):
        if self.mode == "all":
            return len(self.data_dirs)
        if self.mode == "train":
            return len(self.x_train)
        if self.mode == "val":
            return len(self.x_val)
        

    ## get processed data using normal preprocessing
    # def __getitem__(self, index):
    #     if self.mode == "all":
    #         path = f"{self.path}/{self.data_dirs[index]}/{self.scan_type}"
    #         sample ={"image":preprocess(path)}
    #     if self.mode == "train":
    #         #path = f"{self.path}/{self.x_train[index]}/{self.scan_type}"
    #         path = f"{self.path}/{self.x_train[index]}/{self.scan_type}"
    #         sample = {"image":preprocess(path),"label":self.y_train[index]}
    #     if self.mode == "val":
    #         path = f"{self.path}/{self.self.x_val[index]}/{self.scan_type}"
    #         sample = {"image":preprocess(path),"label":self.y_val[index]}
    #     return sample
    
    ## get preprocessed data monai preprocessing
    def __getitem__(self, index):
        if self.mode == "all":
            path = f"{self.path}/{self.data_dirs[index]}/{self.scan_type}"
            sample ={"ID":self.data_dirs[index],"image":preprocess_monai(path,fixed_size=(256,256))}
        if self.mode == "train":
            #path = f"{self.path}/{self.x_train[index]}/{self.scan_type}"
            path = f"{self.path}/{self.x_train[index]}/{self.scan_type}"
            sample = {"image":preprocess_monai(path,fixed_size=(256,256)),"label":self.y_train[index]}
        if self.mode == "val":
            path = f"{self.path}/{self.x_val[index]}/{self.scan_type}"
            sample = {"image":preprocess_monai(path,fixed_size=(256,256)),"label":self.y_val[index]}
        return sample
"""Training pipeline"""
import pandas as pd 
import numpy as np
import os
import torch
from tqdm import tqdm
import pickle

#os.chdir('/mnt/nvme_disk2/User_data/nc36192d/rushi/Learnings/RSNA-MCCAI')
from dataloader import MRIDataset
from model import CNN
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = ("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
print(f"Model loaded to {torch.cuda.get_device_name(0)}")

csv = pd.read_csv('/mnt/nvme_disk2/User_data/nc36192d/rushi/Learnings/kaggle_datasets/rsna_dataset/train_labels.csv')
labels = csv["MGMT_value"].to_list()

EPOCH = 50
LEARNING_RATE = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
criterion = nn.BCELoss()

mri_data = MRIDataset(data_path='/mnt/nvme_disk2/User_data/nc36192d/rushi/Learnings/kaggle_datasets/rsna_dataset/train',
                      labels=labels,scan_type="FLAIR")
mri_data.split_data()
train_dataloader = DataLoader(mri_data,batch_size=48,shuffle=True)
mri_data.mode = "val"
val_dataloader = DataLoader(mri_data,batch_size=48,shuffle=True)
print("Data loaded...")
print("Training started...")

## training the model with the train and validation split
epoch_train_loss = []
epoch_val_loss = []
for epoch in range(EPOCH):
    model.train()
    ## training part
    train_losses = []
    for D in tqdm(train_dataloader,desc=f"Training epoch: {epoch+1}"):
        optimizer.zero_grad()
        img = D['image'].to(device)
        label = D['label'].to(device)

        #predictions
        pred = model(img)
        error = nn.BCELoss()
        loss = torch.sum(error(pred.squeeze(),label.squeeze()))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        #break
    epoch_train_loss.append(np.mean(train_losses))

    ## validation part
    model.eval()
    val_losses = []

    with torch.no_grad():
        for D in tqdm(val_dataloader,desc=f"Validation epoch: {epoch+1}"):
            img = D['image'].to(device)
            label = D['label'].to(device)

            ##prdictions
            pred = model(img)

            error = nn.BCELoss()
            loss = torch.sum(error(pred.squeeze(),label.squeeze()))
            val_losses.append(loss.item())
            #break
    ## loss for each epoch
    epoch_val_loss.append(np.mean(val_losses))

    ## printing the losses at each epoch
    if (epoch+1) % 10 == 0:
        print(f"Training epoch: {epoch+1}\tTraining loss: {np.mean(train_losses)}\tValidation loss: {np.mean(val_losses)}")
    #break
print("Training completed...")

torch.save(model.state_dict(),'saved_models/FLAIR/dropout_model_50i_B_48.pth')
print("Model saved...")
with open('saved_models/FLAIR/dropout_model_50i_B_48_losses.pkl','wb') as f:
    pickle.dump({"train_loss":epoch_train_loss,"val_loss":epoch_val_loss},f)
print("Losses saved...")
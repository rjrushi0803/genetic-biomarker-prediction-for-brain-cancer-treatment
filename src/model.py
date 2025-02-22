"""This function contains the CNN model"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    ## initializing the class
    def __init__(self):
        ## initializing the parent class
        super(CNN,self).__init__()

        ## Defining the convolutional model
        self.cnn_model = nn.Sequential(
            # First convolutionla layer
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=5),
            #Here we stacked images of 64 channels, so the in_channels is 64
            #we are making 128 feature maps, so the out_channels is 128
            #we will use 5x5 filter size for the convolution
            nn.Tanh(),
            #nn.BatchNorm2d(128),  # Batch Normalization
            nn.AvgPool2d(kernel_size=5,stride=4),
            # Second convolutionla layer
            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=5),
            #Here output previous conv layer is of 128 channels, so the in_channels is 128
            #we are making 128 feature maps, so the out_channels is 128
            #we will use 5x5 filter size for the convolution
            nn.Tanh(),
            #nn.BatchNorm2d(256),  # Batch Normalization
            nn.Dropout2d(p=0.3),  # Spatial Dropout after deep conv layers
            nn.AvgPool2d(kernel_size=5,stride=5)
            )
        ## Defining the fully connected layer for image shape 64,128,128, 256*256 = 65536
        self.fc_model = nn.Sequential(
            #first linear layer
            nn.Linear(in_features=30976,out_features=12800),
            nn.Tanh(),
            nn.Dropout(p=0.5), ## drops 50% of the neurons
            #second linear layer
            nn.Linear(in_features=12800,out_features=1280),
            nn.Tanh(),
            nn.Dropout(p=0.4), ## drops 40% of the neurons
            ##final output layer
            nn.Linear(in_features=1280,out_features=128),
            nn.Tanh(),
            nn.Dropout(p=0.2), ## drops 20% of the neurons
            ##Fourth Linear layer
            nn.Linear(in_features=128,out_features=1)
        )
        # ## Defining the fully connected layer 20*20*256 = 102400
        # self.fc_model = nn.Sequential(
        #     #first linear layer
        #     nn.Linear(in_features=102400,out_features=51200),
        #     nn.Tanh(),
        #     #second linear layer
        #     nn.Linear(in_features=51200,out_features=25600),
        #     nn.Tanh(),
        #     ##Third Liner layer
        #     nn.Linear(in_features=25600,out_features=12800),
        #     nn.Tanh(),
        #     ##Fourth Linear layer
        #     nn.Linear(in_features=12800,out_features=6400),
        #     nn.Tanh(),
        #     ##Fifth Linear layer
        #     nn.Linear(in_features=6400,out_features=640),
        #     nn.Tanh(),
        #     ##final output layer
        #     nn.Linear(in_features=640,out_features=1),
        #     nn.Sigmoid()
        # )
        
    ## Defining the forward pass
    def forward(self,x):
        ##passing the image in the cnn_model
        x = self.cnn_model(x) # x will store the output from the cnn_model
        x = x.view(x.size(0),-1) # flatteneing the output which we will be getting from the cnn_model
        #print("Shape after flattening the outputs from the convolutional layer:",x.shape)
        
        ## passing the flattened output to the fully connected network
        x = self.fc_model(x)
        x = F.sigmoid(x) ## converting the output from fully connected in between 0 to 1
        return x


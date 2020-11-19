import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch

############################# Train by appending descriptors ##############################
class Get_model(nn.Module):
    def __init__(self):
        super(Get_model, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_drop = nn.Dropout2d()
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv4_drop = nn.Dropout2d()
        self.conv5 = nn.Conv2d(512, 512, 3, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv5_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.sig = nn.Sigmoid()


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.max_pool2d(self.bn2(self.conv2(x)),2)
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(self.bn3(self.conv3(x)),2))
        x = self.conv3_drop(x)
        x = F.relu(F.max_pool2d(self.bn4(self.conv4(x)),2))
        x = self.conv4_drop(x)
        x = F.relu(F.max_pool2d(self.bn5(self.conv5(x)),2))
        x = self.conv5_drop(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = self.sig(x)
        return x

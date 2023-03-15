#letnet的实现
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(1,6,kernel_size=5)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,kernel_size=5)
        self.pool2=nn.MaxPool2d(2,2)
        self.conv3=nn.Conv2d(16,120,kernel_size=5)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1,120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x











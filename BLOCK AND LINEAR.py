import os
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
#自定义块
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
    def forward(self,x):
        y = self.hidden(x)
        y = F.relu(y)
        return y

net=MLP()





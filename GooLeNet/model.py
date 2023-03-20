#pytorch实现GooleNet
import torch
import torch.nn as nn
import torch.nn.functional as F


















'''卷积块'''
class BasicConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,**kwargs):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,**kwargs)
        self.relu = nn.ReLU(inplace=True)
    def forwward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        return x

'''Inception块'''
class Inception(nn.Module):
    def __init__(self,inputchannels,ch1x1,ch3x3red,ch3x3,ch5x5red,ch5x5,pool_proj):
        super(Inception,self).__init__()
        self.branch1 = BasicConv2d(inputchannels,ch1x1,kernel_size=1),

        self.branch2 = nn.Sequential(
        BasicConv2d(inputchannels,ch3x3red,kernel_size=1),
        BasicConv2d(ch3x3red,ch3x3,kernel_size=3,padding=1)
        )

        self.branch3 = nn.Sequential(
        BasicConv2d(inputchannels,ch5x5red,kernel_size=1),
        BasicConv2d(ch5x5red,ch5x5,kernel_size=5,padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            BasicConv2d(inputchannels,pool_proj,kernel_size=1)
        )

    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outouts = [branch1,branch2,branch3,branch4]
        return torch.cat(outouts,dim=1)

'''构建辅助分类器 InceptionAUX'''
class InceptionAUX(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(InceptionAUX,self).__init__()
        self.avpool = nn.AvgPool2d(kernel_size=5,stride=3)
        self.conv   = BasicConv2d(in_channels,128,kernel_size=1)\
        self.lin1










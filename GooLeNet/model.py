#pytorch实现GooleNet
import torch
import torch.nn as nn
import torch.nn.functional as F
'搭建GooLENet模型'
class GooLeNet(nn.Module):
    def __init__(self,num_class,aux_turn=True,init_weights=False):
        super(GooLeNet,self).__init__()
        self.aux_logits = aux_turn
        self.conv1 = BasicConv2d(3,64,kernel_sieze=7,stride=2,padding=3)
        self.maxpool1 = nn.MaxPool2d(3,stride=2,ceil_mode=True)
        self.conv2 = BasicConv2d(64,64,kernel_size=1)
        self.conv3 = BasicConv2d(64,192,kernel_size=3,padding=1)
        self.maxpool2 = nn.MaxPool2d(3,stride=2,ceil_mode=True)
        self.inception3a = Inception(192,64,96,128,16,32,32)
        self.inception3b = Inception(256,128,128,192,32,96,64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)
        self.inception4a = Inception(480,192,96,208,16,48,64)
        self.inception4b = Inception(512,160,112,224,24,64,64)
        self.inception4c = Inception(512,128,128,256,24,64,64)
        self.inception4d = Inception(512,112,114,288,32,64,64)
        self.inception4e = Inception(528,256,160,320,32,128,128)
        self.maxpool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.inception5a = Inception(832,256,160,320,32,128,128)
        self.inception5b = Inception(832,384,192,384,48,128,128)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(1024,num_class)

        if self.aux_logits == True:
            self.aux1 = InceptionAUX(512,num_class)
            self.aux2 = InceptionAUX(528,num_class)

        if init_weights == True:
            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
           x = self.conv1(x),
           x = self.maxpool1(x),
           x = self.conv2(x),
           x = self.conv3(x),
           x = self.maxpool2(x),
           x = self.inception3a(x),
           x = self.inception4a(x),
           if self.training == True and self.aux_logits ==True:
                aux1 = self.aux1(x)
           x = self.inception4b(x),
           x = self.inception4c(x),
           x = self.inception4d(x),
           if self.training == True and self.aux_logits == True:
                aux2 = self.aux2(x)
           x = self.inception4e(x),
           x = self.maxpool4(x),
           x = self.inception5a(x),
           x = self.inception5b(x),
           x = self.avgpool(x),
           x = self.flatten(x),
           x = self.dropout(x)
           x = self.fc(x)
           if self.training == True and self.aux_logits == True:
               return x ,aux1,aux2
           else:
               return x

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
        self.conv = BasicConv2d(in_channels,128,kernel_size=1)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.7,training=self.training)#当实例化一个模型model后，可以通过model.train()和model.eval（）控制模型的训练和评估模式
        self.lin1 = nn.Linear(2048,1024)
        self.lin2 = nn.Linear(1024,num_classes)

    def forward(self,x):
        x = self.avpool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = F.relu(self.lin1(x),inplace=True)
        x = self.drop(x)
        x = self.lin2(x)
        return x









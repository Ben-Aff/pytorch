#感知机,全连接神经网络(MLP)
import os
import torch
from torch import nn
from d2l import torch as d2l

'''设置超参数'''
batch_size = 256
num_epochs = 10
lr = 0.1





#导入训练和测试模型
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
#构建网络
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,256),
                    nn.ReLU(),
                    nn.Linear(256,10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)









'''构建损失函数'''
loss = nn.CrossEntropyLoss(reduction='none')
'''构建优化器'''
optim = torch.optim.SGD(net.parameters(),lr)
'''进行训练'''
d2l.train_ch3(net, train_iter,test_iter,loss,num_epochs,optim)
'''结果展示'''
d2l.plt.show()
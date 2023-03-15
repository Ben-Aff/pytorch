#房价预测
'''
1.影响房价的关键因素：卧室个数、卫生间个数、居住面积，记为x1x2x3
2.成交价是关键因素的加权和：y=w1x1+w2x2+w3x3+b，权重和偏差的实际值在后边决定
训练数据：过去6个月卖的房子
'''
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
#在torch.nn中定义了大量的模型层
from torch import nn
#生成数据集
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)#调用d2l包中的生成数据集包来生成数据集，其中num_sample=1000
#调用框架中现有的API来读取数据
def load_array(data_arrays,batch_size,is_train=True):
    '构造一个pytorch数据迭代器'
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batchi_size = 10
data_iter = load_array((features,labels),batchi_size)
next(iter(data_iter))

#搭建神经网络
net = nn.Sequential(nn.Linear(2,1))

#初始化模型参数
net[0].weight.data.normal_(0,0.01)#nomal_将权重初始值使用正态分布，mean为0，均值为0.01
net[0].bias.data.fill_(0)#偏差初始值为0

#定义损失loss
loss = nn.MSELoss()

#构造优化器
optim = torch.optim.SGD(net.parameters(),lr=0.03)

#设置训练
num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        optim.zero_grad()
        l.backward()
        optim.step()
        l = loss(net(features),labels)
    print(f'epoch {epoch + 1}, loss {l:f}')





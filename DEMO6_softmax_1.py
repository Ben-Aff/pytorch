#SOFTMAX-PYTORCH实现
import torch
from torch import nn
from d2l import torch as d2l

'''加载数据集和测试集'''
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)#train_iter和test_iter分别返回训练集和测试集的迭代器

'''搭建神经网路'''
net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))#定义展平层，在线性层前调整网络输入的形状
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights);

'''定义损失函数loss'''
loss = nn.CrossEntropyLoss(reduction='none')

'''构造优化器'''
optim = torch.optim.SGD(net.parameters(),lr=0.1)

'''设置训练循环'''
num_epochs = 10
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,optim)
d2l.plt.show()







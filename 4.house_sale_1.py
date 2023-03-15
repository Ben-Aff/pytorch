#房价预测
'''
1.影响房价的关键因素：卧室个数、卫生间个数、居住面积，记为x1x2x3
2.成交价是关键因素的加权和：y=w1x1+w2x2+w3x3+b，权重和偏差的实际值在后边决定
训练数据：过去6个月卖的房子
'''
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils import data
from d2l import torch as d2l
#构造人造数据集
def synthetic_data(w,b,num_examples):
    '''生成噪声数据集：y=xw+b+随机噪声'''
    x = torch.normal(0,1,(num_examples,len(w)))#生成服从正态分布的数据x，行尺寸为num_example(样本大小)，列尺寸为w的长度
    y = torch.matmul(x,w) + b
    y += torch.normal(0,0.01,y.shape)#加入服从正态分布的随机噪音，行列尺寸和y的行列尺寸一致
    return x,y.reshape((-1,1))#返回x和y(y以列向量的形式返回)
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)#创造噪声数据集,x(1000,2)，y(1000,1)

#定义一个迭代器datat_iter函数，该函数接受批量大小、特征矩阵和向量标签作为输入，生成大小为batch_size的小批量数据集
def data_iter(batch_size,features,labels):
    num_examples = len(features)#获取行的大小
    indices = list(range(num_examples))#生成索引下标
    #这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)#将下标索引完全打乱，用随机的索引取访问随机的样本
    for i in range(0,num_examples,batch_size):#从0到所有样本，每次取batchsize个大小
        batch_indices=torch.tensor(
            indices[i:min(i+batch_size,num_examples)])#找出batch的索引，min和nun_examples是如果不能正好去满，就取最后一个样本
        yield features[batch_indices],labels[batch_indices]

batch_size = 10

for x,y in data_iter(batch_size,features,labels):
    print(x, '\n',y)
    break

#定义初始化模型参数
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

#定义模型
def LinearModel(x,w,b):
    '线性回归模型'
    return torch.matmul(x,w)+b

#定义损失函数
def mes_loss(y_hat,y):
    'mse均方损失'
    return (y_hat-y.reshape(y_hat.shape))**2/2# 这里为什么要加y_hat_shape:torch.Size([10, 1])   y_shape: torch.Size([10])

#定义优化算法
def SGD(params, batch_size, lr):
    with torch.no_grad():  # with torch.no_grad() 则主要是用于停止autograd模块的工作，
        for param in params:
            param -= lr * param.grad / batch_size  ##  这里用param = param - lr * param.grad / batch_size会导致导数丢失，zero_()函数报错
            param.grad.zero_()  ##导数如果丢失了,会报错‘NoneType’ object has no attribute ‘zero_’

#设置训练超参数
lr = 0.03
num_epochs = 3

#训练模型
for epoch in range(0, num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = LinearModel(X,w,b)
        f = mes_loss(y_hat, y)
        # 因为`f`形状是(`batch_size`, 1)，而不是一个标量。`f`中的所有元素被加到一起，
        # 并以此计算关于[`w`, `b`]的梯度
        f.sum().backward()
        SGD([w, b], batch_size, lr)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = mes_loss(LinearModel(features, w, b), labels)
        print("w {0} \nb {1} \nloss {2:f}".format(w, b, float(train_l.mean())))

#比较真实参数和通过训练学到的参数来评价训练的成功程度
print("w误差 ", true_w - w, "\nb误差 ", true_b - b)





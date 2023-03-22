#参数初始化
import torch
import torch.nn as nn
'''pytorch中线性层参数内置初始化'''
def init_normal(m):                 #m是model
    if type(m) == nn.Linear:        #判断model是否是线性层
        nn.init.normal_(m.weight,mean=0,std=0.01)#将线性层的weight正态分布初始化
        nn.init.zeros_(m.bias)#将线性层的bias全部置0
net.apply(init_normal)#对net网络中的所有线性层都遍历应用一遍



'''将线性层中所有参数初始化为给定的常数，比如初始化为1(并不会用)'''
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)



'''对某些块应用不同的初始化方法'''
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(init_xavier)#对第一个神经网络层应用Xavier初始化方法
net[2].apply(init_42)#对第三个神经网络层初始化为常量42


'''自定义初始化(并不常用)'''
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5
net.apply(my_init)

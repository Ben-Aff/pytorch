#Linear model y=wx
#method of gradient Decent
import torch
import numpy as np
import matplotlib.pyplot as plt
#prepar the train set
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
#initlize the weight w=1.0 and learning rate n=0.01
w=1.0
n=0.01
#define the model y=wx
def forward(x):
    return x*w
#define the loss function loss=∑(y-y^)2/N
def loss(x,y_label):
    loss=0
    for x1,y_label1 in zip(x,y_label):#use for statement to relize the accumlation function
        y=forward(x1)
        loss+=(y_label1-y)**2
    return loss/len(x)
#define the gradient grad=∑2(y-y^)/N*X
def gradient(x,y_label):
    grad=0
    for x1,y_label1 in zip(x,y_label):#use for statement to relize the accumlation funtion
        y=forward(x1)
        grad+=2*(y-y_label1)*x1
    return grad/len(x)
#optimize：update weight-n*grad
print('predict  before training',4,forward(4))
for epoch in range(100):
    loss_val=loss(x_data,y_data)
    grad_val=gradient(x_data,y_data)
    w-=n*grad_val
    print('epoch：',epoch)
    print('w=',w,'loss=',loss_val)
print('predict after training',4,forward(4))

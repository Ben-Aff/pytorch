#Linear model y=wx
#method of exhaustion
import torch
import numpy as np
import matplotlib.pyplot as plt
#prepar the train set
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
#define the linear model y=wx+b
def forward(x):
    return x*w
#define error
def error(x,y_label):
    y=forward(x)
    return (y-y_label)*(y-y_label)
#create two empty lists to save all weight and weight
w_list=[]
loss_list=[]
#method of exhaustion
for w in np.arange(0.0,4.1,0.1):
    print('now the weight is',w)#initialize the weight from 0.0-4.0,step=0.1
    error_sum_val=0
    for x_val,y_label_val in zip(x_data,y_data):
        y_val=forward(x_val)
        error_val=error(x_val,y_label_val)
        error_sum_val+=error_val
        print('\t',x_val,y_val,y_label_val,error_sum_val)
    loss=error_sum_val/3
    print('loss=',loss)
    w_list.append(w)
    loss_list.append(loss)
print(w_list)
print(loss_list)
#show the result by matplotlib
plt.plot(w_list,loss_list)
plt.ylabel('loss')
plt.xlabel('w')
plt.show()

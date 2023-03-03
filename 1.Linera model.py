#Linear model y=wx
#method of exhaustion
import torch
import numpy as np
import matplotlib.pyplot as plt
#prepar the train set
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
#define the forward function
def forward(x):
    return x*w
def error(x,y_label):
    y=forward(x)
    return (y-y_label)*(y-y_label)
#create two empty lists to save all weight and bias and loss
w_list=[]
loss_list=[]
#method of exhaustion
for w in np.arange(0.0,4.1,0.1):
    error_sum_val=0
    print('w=',w)
    for x_val,y_label_val in zip(x_data,y_data):
        y_val=forward(x_val)
        error_val=error(x_val,y_label_val)
        error_sum_val+=error_val
        print('\t',x_val ,y_val ,y_label_val ,'\t',error_sum_val  )
    loss=error_sum_val/3
    print('loss=',loss)
    loss_list.append(loss)
    w_list.append(w)
#show the result by matplotlib
plt.plot(w_list,loss_list,color='red')
plt.ylabel('loss')
plt.xlabel('w')
plt.grid(alpha=0.4)
plt.title('method of exhaustion')
plt.show()

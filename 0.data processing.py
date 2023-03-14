import numpy as np
import pandas as pd
import torch
import pandas
import os
os.makedirs(os.path.join('..','data',),exist_ok=True)
data_file=os.path.join('..','data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')#列名
    f.write('NA,Pave,127500\n')#每行表示一个数据样本
    f.write('2,NA,10600\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data=pd.read_csv(data_file)
print(data)
#为了处理缺失的数据，典型的方法包括插值和删除，这里我们考虑插值
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]
inputs=inputs.fillna(inputs.mean())
print(inputs)
#独热编码
inputs=pd.get_dummies(inputs,dummy_na=True)
print(inputs)
x,y=torch.tensor(inputs.values),torch.tensor(outputs.values)

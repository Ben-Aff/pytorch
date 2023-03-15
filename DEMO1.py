#线性回归
#构造一组输入数据x和其对应的标签
import torch
import numpy as np
import torch.nn as nn
#构造一组输入数据和其对应的标签
x_values=[i for i in range(11)]
x_train=np.array(x_values,dtype=np.float32)
x_train=x_train.reshape(-1,1)
y_values=[2*i+1 for i in x_values]
y_train=np.array(y_values,dtype=np.float32)
y_trainy=y_train.reshape(-1,1)
#构造线性模型
class LinearRegressionModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegressionModel,self).__init__()
        self.linear=nn.Linear(input_dim,output_dim)
    def forward(self,x):
        out=self.linear(x)
        return out
model=LinearRegressionModel(1,1)
#指定好参数和损失函数
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
criterion=nn.MSELoss()
#指定使用GPU运算
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#指定使用GPU运算
model.to(device)
#将训练数据转换为tensor形式
inputs = torch.from_numpy(x_train)
labels = torch.from_numpy(y_train)
#训练模型
for epoch in range(1000):
    epoch += 1
    #将输入和标签传入到GPU中
    inputs=torch.from_numpy(x_train).to(device)
    labels=torch.from_numpy(x_train).to(device)
    #梯度要清零每一次迭代
    optimizer.zero_grad()
    #前向传播
    outputs=model(inputs)
    #计算损失
    loss=criterion(outputs,labels)
    #反向传播
    loss.backward()
    #更新权重参数
    optimizer.step()
    if epoch%50==0:
        print('epoch {},loss {}'.format(epoch,loss.item()))
#预测模型预测结果
predicted=model(torch.from_numpy(x_train).requires_grad_())
#模型的保存与读取
torch.save(model.state_dict(),'model.pkl')#torch.save,model.state_dict保存的是模型的权重和偏差
model.load_state_dict(torch.loda(model.pkl))#









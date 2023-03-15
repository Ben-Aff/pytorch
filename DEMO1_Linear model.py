#使用pytorch搭建线性模型
import torch
import numpy as np
#加载数据集
x_data=torch.tensor([[1.0],[2.0],[3.0]])
y_data=torch.tensor([[2.0],[4.0],[6.0]])
#定义模型
class LinearModel(torch.nn.Module):#需要继承自nn.Module类
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred
#实例化模型
model=LinearModel()
#构造损失函数和优化器
criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#进行训练(前馈、反馈、更新)
for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#打印权重和偏差
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())
#测试模型
x_test=torch.Tensor([[4.0]])
y_test=model(x_test)
print('y_pred=',y_test.data)
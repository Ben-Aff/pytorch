#使用pytorch进行逻辑回归
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[0],[0],[1]])
#搭建网络模型
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred
model=LogisticRegression()
#构建损失函数和优化器
creation=torch.nn.BCELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#训练过程(前馈、反馈、更新)
for epoch in range(1000):
    y_pred=model(x_data)
    loss=creation(y_pred,y_data)
    print(epoch,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#测试模型
x=np.linspace(0,10,200)
x_t=torch.Tensor(x).view((200,1))
y_t=model(x_t)
y=y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('probability of Pass')
plt.grid()
plt.show()

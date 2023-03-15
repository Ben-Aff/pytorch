#用pytorch进行MNIST分类
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) #归一化,均值和方差
#数据集处理
train_dataset = datasets.MNIST(root = 'F:/7.pytorch/pytorch_demo1/dataset/mnist',
                               train = True,
                               transform = transforms,
                               download = True)
test_dataset = datasets.MNIST(root = 'F:/7.pytorch/pytorch_demo1/dataset/mnist',
                              train = False,
                              transform = transforms,
                              download = True)#加载数据集
train_Loader = DataLoader(dataset = train_dataset,
                          batch_size = 32,
                          shuffle = True)
test_Loader = DataLoader(dataset = test_dataset,
                         batch_size = 32,
                         shuffle = False)
#构建网络模型
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
        self.fc    = torch.nn.Linear(320,10)
        self.pooling=torch.nn.MaxPool2d(2)
    def forward(self, x):
        bitch_size =x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = x.view(bitch_size,-1)
        x = self.fc(x)
        return x               #最后一层不做激活，不进行非线性变换
model=MyModel()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
#构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
#设置训练循环
def train(epoch):
    total_loss = 0.0
    for batch_idx,data in enumerate(train_Loader,1):
        inputs,target = data
        inputs,target = inputs.to(device),target.to(device)
        optimizer.zero_grad()
        #forward
        outputs = model(inputs)
        #loss
        loss = criterion(outputs, target)
        #backward
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 300 ==299: #每300个batch打印一次
            print('[%d, %5d] loss: %.3f' %(epoch +1 ,batch_idx , total_loss / 300))
        total_loss =0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():#用此语句声明后，语句下的代码不会计算梯度
        for data in test_Loader:
            input, labels = data
            input, labels = input.to(device), labels.to(device)
            output = model(input)
            _, predicted = torch.max(output.data, dim=1)  # dim=1 列是第0个维度，行是第1个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(100):#设定训练100轮
        train(epoch)
        test()





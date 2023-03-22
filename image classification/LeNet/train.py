#训练文件
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
#图像预处理
    transform = transforms.Compose(  [transforms.ToTensor(),
                                     transforms.Normalize((0.1307), (0.3081))])

#设置训练集
    trainset = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=True,
                                          download=True, transform= transform )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,
                                           shuffle=True, num_workers=0)

#设置测试集
    valset = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=False,
                                        download=True, transform=transform)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=10000,
                                             shuffle=False, num_workers=0)

    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

#调用模型
    net = LeNet()

#构造损失函数
    loss_function = nn.CrossEntropyLoss()

#构造优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)

#设置训练
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for step, data in enumerate(trainloader, start=0):
#get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad()
# 前馈 + 反馈 + 优化
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

# print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y,val_label).sum().item() / val_label.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                        (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

#保存权重
    save_path = 'logs/CIFAR10LeNet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()

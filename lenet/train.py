#训练文件
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(  [transforms.ToTensor(),
                                     transforms.Resize((32, 32)),
                                     transforms.Normalize((0.1307), (0.3081))])

# 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform= transform )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                           shuffle=True, num_workers=0)
#第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                         shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

#调用模型
    net = LeNet()

#构造损失函数
    loss_function = nn.CrossEntropyLoss()

#构造优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)

#设置训练
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
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
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                        (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

#保存权重
    save_path = './logs/MNistLeNet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()

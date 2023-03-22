#训练文件
import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet

def main():

    '''设置设备'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    '''数据集预处理'''
    data_transform = {  "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                        "val":   transforms.Compose([transforms.Resize((224, 224)),  #cannot 224, must (224, 224)
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    '''设置训练集'''
    train_dataset = datasets.ImageFolder(root="../dataset/flowers/train",
                                         transform=data_transform["train"])

    '''查看训练集大小---验证训练集是否加载成功'''
    train_num = len(train_dataset)

    # 设置类别{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))




    '''加载训练集'''
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root="../dataset/flowers/train",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    '''加载验证集'''
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4,
                                                  shuffle=False,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = AlexNet(numclass=5, init_weights=True)
    net.to(device)
    '''定义损失函数'''
    loss_function = nn.CrossEntropyLoss()
    '''构造优化器'''
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    epochs = 100
    save_path = 'logs/AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    #训练
    for epoch in range(epochs):
        #训练
        net.train()#训练时开启丢弃法
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            #打印结果
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        #验证
        net.eval()#验证时关闭丢弃法
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')



if __name__ == '__main__':
    main()

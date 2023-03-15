#读取图片数据集
import torch
import torchvision
import torchvision.transforms as transforms
from d2l import torch as d2l
from torch.utils import data
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import extract_archive

#将图像高清显示
d2l.use_svg_display()
#将图像数据从PIL类型转换为32位浮点数格式
trans = transforms.ToTensor()

'''设置数据集'''
mnist_train = torchvision.datasets.FashionMNIST(root = "../data",
                                                train = True,
                                                transform = trans,
                                                download = True)
mnist_test = torchvision.datasets.FashionMNIST(root = "../data",
                                               train = False,
                                               transform = trans,
                                               download = True)

#len(mnist_train)=60000,len(mnist_test)=10000
#mnist_train[0][0].shape=torch.siez([1,28,28]),第一张图片是hwc为1*28*28

'''可视化数据集函数'''
def get_fashion_mnist_labels(labels):  #@save
    '''返回Fashion-MNIST数据集的文本标签。'''
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']        #将图片中的类别进行标注
    return [text_labels[int(i)] for i in labels]
def show_images(imgs,num_rows,num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            #图片张量
            ax.imshow(img.numpy())
        else:
            #PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

'''展示数据集'''
x,y = next(iter(data.DataLoader(mnist_train,batch_size=18)))
show_images(x.reshape(18, 28, 28),2,9,titles=get_fashion_mnist_labels(y))
d2l.plt.show()

'''加载数据集'''
train_data = data.DataLoader(mnist_train,
                             batch_size=256,
                             shuffle=True,
                             num_workers=4)

'''测试加载数据集速度'''
timer = d2l.Timer()
for X, y in train_data:
    continue
print(f'{timer.stop():.2f} sec')











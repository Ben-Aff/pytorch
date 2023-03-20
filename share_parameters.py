#参数共享
shared = nn.Linear(8, 8)#指定共享层名称，以便可以引用它的参数
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0]=100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
'''这个例子表明第三个和第五个神经网络层的参数是绑定的它们不仅值相等，而且由相同的张量表示。 
   因此，如果我们改变其中一个参数，另一个参数也会改变。 
这里有一个问题：当参数绑定时，梯度会发生什么情况？ 
答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层（即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。
'''

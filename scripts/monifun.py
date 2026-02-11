import random
import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt

scatter = True


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(0)     # 固定随机种子， 让我们每次运行都得到相同的结果。

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(in_features=1,out_features=100),    #  Linear model
            # nn.Sigmoid(),
            # nn.Linear(100,100),
            # nn.Sigmoid(),
            # nn.Linear(100, 100),
            nn.ReLU(),
            # nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            # nn.Linear(100, 100),
            # nn.ReLU(),
            nn.Linear(100, 1)
        )
        #




    def forward(self, input):
        return self.net(input)

class mydataset(Dataset):
    def __init__(self,x ,y):
        super(mydataset, self).__init__()
        self.x = torch.tensor(x,dtype=torch.float)
        self.y = torch.tensor(y,dtype=torch.float)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


data_num = 100
# 准备数据
train_s = -5
train_e = 5
x=np.linspace(train_s,train_e, data_num)
# y=np.sin(x)

# y = 20*x+4
#创建数据

y = x**3 + 2*x**2


# 在输入上加入噪声。
mu = 0
sigma = 1
y += np.random.normal(mu, sigma, y.shape)
# for i in range(data_num):
#     y[i] += random.gauss(mu, sigma)

# 将数据做成数据集的模样
X=np.expand_dims(x,axis=1)
Y=y.reshape(data_num,-1)


# 使用批训练方式
# dataset=TensorDataset(torch.tensor(X,dtype=torch.float),torch.tensor(Y,dtype=torch.float))
dataset = mydataset(X, Y)
dataloader=DataLoader(dataset,batch_size=100,shuffle=True)


# 神经网络主要结构，这里就是一个简单的线性结构


net = Net()
# 定义优化器和损失函数
optim=torch.optim.Adam(net.parameters(),lr=0.001)
# optim = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.09)
Loss = nn.MSELoss()



# 下面开始训练：
# 一共训练1000次
for epoch in range(1000):
    loss = None
    for batch_x,batch_y in dataloader:
        y_predict=net(batch_x)
        loss=Loss(y_predict,batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    # 每100次 的时候打印一次日志
    if (epoch+1)%100==0:
        print("step: {0} , loss: {1}".format(epoch+1,loss.item()))
        # 使用训练好的模型进行预测
        predict=net(torch.tensor(X,dtype=torch.float))
        # 绘图展示预测的和真实数据之间的差异
        if scatter:
            plt.scatter(x, y, 1)
        else:
            plt.plot(x,y,color="coral",label="fact")
        plt.plot(x,predict.detach().numpy(), color="coral",label="predict")
        plt.title("function")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        # plt.savefig(fname="result.png",figsize=[10,10])
        plt.show()



# # 做一个测试集，解除注释可查看测试集结果
# if scatter:
#     plt.scatter(x,y,1)
# else:
#     plt.plot(x, y, label="fact")
#
# test_s = train_e
# test_e = test_s+5
# test_x = np.linspace(test_s, test_e, data_num)
# # y=np.sin(x)
# # y = 20*x+4
# test_y = test_x ** 3 + 2 * test_x ** 2

# # 将数据做成数据集的模样
# test_X = np.expand_dims(test_x, axis=1)
# test_Y = test_y.reshape(data_num, -1)
# # 使用批训练方式
# # test_dataset = TensorDataset(torch.tensor(test_X, dtype=torch.float), torch.tensor(test_Y, dtype=torch.float))
# # test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)
#
# pre = net(torch.tensor(test_X,dtype=torch.float))
#
#
#
# x = np.concatenate((x, test_x), axis=0)
# y = np.concatenate((y, test_y), axis=0)
#
# pre_y = np.concatenate((predict.detach().numpy(), pre.detach().numpy()), axis=0)
# plt.plot(x, pre_y, color="coral", label="predict")
#
# if scatter:
#     plt.scatter(x, y, 1,color="blue", label="fact")
# else:
#     plt.plot(x, y, label="fact",color="blue")
#
#
#
#
# # plt.title("sin function")
# plt.xlabel("x")
# plt.legend()
# # plt.savefig(fname="result.png",figsize=[10,10])
# plt.show()
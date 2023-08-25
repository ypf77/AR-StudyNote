# -------------------------------------使用指南---------------------------------
# 1、数据集使用.npy格式
# 2、模型要选择
# 3、模型保存路径要确认

# -------------------------------------导入相关包---------------------------------
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tools import *
# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from models import MLP, FCN

# -------------------------------------数据准备---------------------------------
# 读入滑窗操作完成后的数据集
X = np.load("data/UCR/NATOPS/X.npy")
y = np.load("data/UCR/NATOPS/y.npy")
y = y.astype(np.float32) - 1
y = torch.tensor(y, dtype=torch.long)
# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.5, shuffle=True)


# -------------------------------------构造Dataset和DataLoader---------------------------------
class MyDataSet(Dataset):
    def __init__(self, features, label):
        self.X = features
        self.y = label

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


TrainSet = MyDataSet(X_train, y_train)
TestSet = MyDataSet(X_test, y_test)

TrainLoader = DataLoader(TrainSet, batch_size=9, shuffle=False)
TestLoader = DataLoader(TestSet, batch_size=9, shuffle=False)
# -------------------------------------数据可视化---------------------------------
# 绘制一个batch的图像
# plot_figures(TestLoader, 3, 3, suptitle='a batch samples')

# -------------------------------------模型引入，超参数设置---------------------------------
# 定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCN()
model = model.to(device)

# 超参数
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epoch = 100

# 日志
# writer = SummaryWriter("logs_train")
# -------------------------------------训练过程---------------------------------
best_accuracy = 0.0
plot_list = [[], [], []]
for i in range(epoch):
    train_accuracy, train_loss = train(model, device, TrainLoader, criterion, optimizer, epoch)
    test_accuracy = test(model, device, TestLoader, criterion, epoch)
    plot_list[0].append(train_accuracy)
    plot_list[1].append(train_loss)
    plot_list[2].append(test_accuracy)
    if test_accuracy > best_accuracy:
        savemodel(model, 'modelpth/FCN_v1.pth')
        print("epoch-{} save".format(i + 1))
        best_accuracy = test_accuracy

# writer.close()
# 绘制训练曲线
plt_curve(plot_list)

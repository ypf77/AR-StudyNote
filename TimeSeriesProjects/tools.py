import matplotlib.pyplot as plt
import torch

# ---------------------------------------绘图工具----------------------------------------------------------------
plt.rcParams['figure.figsize'] = (13, 6)
# plt.rcParams['axes.facecolor'] = 'snow'
plt.rcParams['axes.facecolor'] = 'azure'


def plot_single_figure(index, row, column, X, y, channels):
    ax1 = plt.subplot(row, column, index + 1)
    for i in range(channels):
        plt.plot(X[index][i])
    plt.title("label:{}".format(y[index]), fontsize=10, color='black', pad=5)
    plt.axis([0, 50, -3.5, 3.5])
    # plt.xticks([])
    plt.grid(linestyle='-.', linewidth=0.5)


def plot_figures(DataLoader, row, column, suptitle):
    for X, y in DataLoader:
        for index in range(X.shape[0]):
            plot_single_figure(index, row, column, X, y, X.shape[1])
        plt.suptitle(suptitle, fontsize=18)
        plt.show()
        break


# ---------------------------------------模型保存----------------------------------------------------------------
def savemodel(model, path):
    torch.save(model.state_dict(), path)


# ---------------------------------------训练函数----------------------------------------------------------------
def train(model, device, trainloader, criterion, optimizer, epoch, writer=None):
    model.train()
    # 总的样本数
    total_num = 0.0
    correct = 0.0
    total_loss = 0.0  # 总损失值初始化为0
    # 循环读取训练数据集，更新模型参数
    for batch_id, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度初始化为0
        output = model(data)  # 训练后的输出
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        # 获取预测结果中每行数据概率最大的下标
        _, preds = torch.max(output.data, dim=1)
        total_num += target.size(0)
        # 累计预测正确的个数
        correct += (preds == target).sum().item()
        total_loss += loss.item()  # 累计训练损失
    # writer.add_scaler("Train Loss", total_loss / len(trainloader), epoch)
    # writer.flush()  # 刷新
    # 平均损失
    total_loss /= total_num
    # 正确率
    accuracy = correct / total_num
    return accuracy, total_loss  # 返回平均损失值


# ---------------------------------------测试函数----------------------------------------------------------------
def test(model, device, testloader, criterion, epoch, writer=None):
    model.eval()
    # 损失和正确
    total_loss = 0.0
    correct = 0.0
    # 总的样本数
    total_num = 0.0
    # 循环读取数据
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            # 预测输出
            output = model(data)
            # 计算损失
            total_loss += criterion(output, target).item()
            # 获取预测结果中每行数据概率最大的下标
            _, preds = torch.max(output.data, dim=1)
            total_num += target.size(0)
            # 累计预测正确的个数
            correct += (preds == target).sum().item()
        # 平均损失
        total_loss /= total_num
        # 正确率
        accuracy = correct / total_num
        # # 写入日志
        # writer.add_scaler('Test Loss', total_loss, epoch)
        # writer.add_scaler('Accuracy', accuracy, epoch)
        # # 刷新
        # writer.flush()
        print("Test Loss : {:.4f}, Accuracy : {:.2f}%".format(total_loss, 100*accuracy))
        return accuracy


# ---------------------------------------训练过程可视化----------------------------------------------------------------
def plt_curve(plot_list):
    legend_list = ["train_acc", "train_loss", "test_acc"]
    plt.grid(linestyle='-.', linewidth=0.5)
    for i in range(3):
        plt.plot(plot_list[i],label=legend_list[i])
    plt.legend()
    plt.title("Training Curve", fontsize=20, color='black', pad=10)
    plt.show()

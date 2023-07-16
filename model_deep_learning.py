import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

from untils import read_events_heat, all_columns


# 定义CNN模型
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, stride=1, padding=7)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=15, stride=1, padding=7)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 11, 32 * 2)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 11)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x[:, 0]

    # def __init__(self):
    #     super(CNNRegressor, self).__init__()
    #     self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
    #     self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
    #     self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
    #     self.pool = nn.MaxPool1d(kernel_size=2)
    #     self.fc1 = nn.Linear(128 * 3, 128)
    #     self.fc2 = nn.Linear(128, 16)
    #     self.fc3 = nn.Linear(16, 1)
    #     self.relu = nn.ReLU()
    #
    # def forward(self, x):
    #     x = self.pool(self.relu(self.conv1(x)))
    #     x = self.pool(self.relu(self.conv2(x)))
    #     x = self.pool(self.relu(self.conv3(x)))
    #     x = x.view(-1, 128 * 3)
    #     x = self.relu(self.fc1(x))
    #     x = self.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x[:, 0]


# 模型训练函数
def train():
    # 读取数据
    all_events_heat = read_events_heat(filename="heat_events.xlsx")
    event_heat = all_events_heat['热度']

    events_extra = read_events_heat(filename="combine_data_v1.xlsx")
    df_cluster = events_extra[all_columns]
    # 将缺失值替换为 零值
    df_cluster.fillna(0, inplace=True)

    # 标准化处理
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    data = np.array(X_scaled)
    labels = np.array(event_heat.values)

    # 将数据转换为3维张量
    data = data.reshape((data.shape[0], 1, data.shape[1]))

    # 划分训练集、验证集和测试集
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2)

    # 将数据转换为PyTorch张量，并将其转移到GPU上
    train_data = torch.tensor(train_data, dtype=torch.float32).cuda()
    val_data = torch.tensor(val_data, dtype=torch.float32).cuda()
    test_data = torch.tensor(test_data, dtype=torch.float32).cuda()
    train_labels = torch.tensor(train_labels, dtype=torch.float32).cuda()
    val_labels = torch.tensor(val_labels, dtype=torch.float32).cuda()
    test_labels = torch.tensor(test_labels, dtype=torch.float32).cuda()

    # 模型初始化到GPU上
    model = CNNRegressor().cuda()

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # MSE 每个epoch的损失 => 绘图
    train_losses = []
    val_losses = []

    # 定义验证集损失阈值
    val_loss_threshold = 0.1
    best_val_loss = float('inf')
    early_stop = False

    # 训练模型，并将数据和模型转移到GPU上
    num_epochs = 200
    batch_size = 64

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, train_data.shape[0], batch_size):
            batch_data = train_data[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        # 记录每个epoch的平均损失值
        epoch_loss = epoch_loss / len(train_labels) * 100
        train_losses.append(epoch_loss)
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))

        # 验证模型
        model.eval()
        with torch.no_grad():
            val_output = model(val_data)
            val_loss = criterion(val_output, val_labels)
            val_loss = val_loss.item() / len(val_labels) * 100
            val_losses.append(val_loss)
            print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, val_loss))

        # 判断验证集损失是否已经收敛
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        elif val_loss - best_val_loss > val_loss_threshold:
            print('Validation loss did not improve by more than threshold, early stopping...')
            early_stop = True
            break

        if not early_stop:
            print('Training completed without early stopping.')

    # 保存模型
    if not os.path.exists('./models_deeplearning'):
        os.makedirs('./models_deeplearning')
    torch.save(model.state_dict(), './models_deeplearning/CNNRegressor_k3.pth')

    # 绘制训练集和测试集的损失曲线
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curve')
    plt.legend()
    plt.savefig('./models_deeplearning/loss_curve_cnn_k3.png')
    plt.show()

    # 测试模型
    model.eval()

    with torch.no_grad():
        train_output = model(train_data)
        train_loss = criterion(train_output, train_labels)

        val_output = model(val_data)
        val_loss = criterion(val_output, val_labels)

        test_output = model(test_data)
        test_loss = criterion(test_output, test_labels)

        print('Train Loss: {:.4f}'.format(train_loss.item()))
        print('Val Loss: {:.4f}'.format(val_loss.item()))
        print('Test Loss: {:.4f}'.format(test_loss.item()))

        # 计算R2分数
        r2_train = r2_score(train_labels.cpu().numpy(), train_output.cpu().numpy())
        r2_val = r2_score(val_labels.cpu().numpy(), val_output.cpu().numpy())
        r2_test = r2_score(test_labels.cpu().numpy(), test_output.cpu().numpy())
        print('R2 score train data:', r2_train)
        print('R2 score val data:', r2_val)
        print('R2 score test data:', r2_test)


# 查看特征和最终热度的相关系数
def corr():
    # 中文显示
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 读取数据
    all_events_heat = read_events_heat(filename="heat_events.xlsx")
    event_heat = all_events_heat['热度']

    events_extra = read_events_heat(filename="combine_data_v1.xlsx")
    df_cluster = events_extra[all_columns]
    # 将缺失值替换为 零值
    df_cluster.fillna(0, inplace=True)

    # 标准化处理
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # 将训练数据转为数据表
    corr_pd = pd.DataFrame(data=X_scaled, columns=all_columns)
    corr_pd['热度'] = event_heat

    # 计算相关系数矩阵
    corr_matrix = corr_pd.corr()

    # 绘制相关系数热力图
    plt.figure(figsize=(40, 40))
    ax = sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
    plt.show()


if __name__ == '__main__':
    train()
    # corr()

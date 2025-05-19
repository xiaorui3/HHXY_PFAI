import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExcelDataset(Dataset):
    def __init__(self, file_path):
        """
        初始化数据集
        :param file_path: Excel文件路径
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            self.data = pd.read_excel(file_path, dtype={'计算类型': str})
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")

        if self.data.isnull().values.any():
            raise ValueError("Data contains missing values")

        # 将计算类型转换为数值，fp16为1，fp32为0
        self.data['计算类型'] = self.data['计算类型'].map({'fp16': 1, 'fp32': 0}).fillna(0).astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据集中的单个样本
        :param idx: 样本索引
        :return: 特征和标签
        """
        try:
            features = torch.tensor(self.data.iloc[idx, 0:-1].astype('float32'))  # 所有行除最后一列外都是特征
            label = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float32)  # 最后一列是标签
        except Exception as e:
            raise ValueError(f"Error processing data at index {idx}: {e}")
        return features, label

class PredictModel(nn.Module):
    def __init__(self):
        """
        初始化预测模型
        """
        super(PredictModel, self).__init__()
        self.fc1 = nn.Linear(6, 256)  # 增加隐藏层大小
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)  # 输出一个值，即准确率

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征
        :return: 输出预测值
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(model, dataset, device, epochs=10000, batch_size=128, learning_rate=0.0001, log_interval=100):
    """
    训练模型
    :param model: 模型
    :param dataset: 数据集
    :param device: 设备（CPU或GPU）
    :param epochs: 迭代次数
    :param batch_size: 批次大小
    :param learning_rate: 学习率
    :param log_interval: 日志间隔
    """
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ii=0
    for epoch in range(epochs):
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.view(-1, 1).to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ii+=1
            if (ii + 1) % log_interval == 0:
                logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

def predict(model, features, device):
    """
    预测
    :param model: 模型
    :param features: 输入特征
    :param device: 设备（CPU或GPU）
    :return: 预测值
    """
    model.eval()
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        prediction = model(features_tensor)
        return prediction.item()

def save_full_model(model, file_path):
    torch.save(model.cpu(), file_path)  # 保存模型时确保在CPU上

def load_full_model(model_path, device):
    return torch.load(model_path).to(device)  # 加载模型时移动到指定设备

if __name__ == '__main__':
    print("是否加载模型？")
    print("1.是")
    print("2.否")
    choice = input("请输入：")
    device = torch.device("cuda")
    save_full_model_path = "full_model.pth"
    if choice == "1":
        model = load_full_model(save_full_model_path, device)
        print("模型加载成功")
    elif choice == "2":
        print("开始训练模型")
        file_path = '工作簿2_models_test.xlsx'
        dataset = ExcelDataset(file_path)
        model = PredictModel().to(device)
        train_model(model, dataset, device)
        save_full_model_path = 'full_model.pth'
        save_full_model(model, save_full_model_path)
        print("模型训练完成，模型已保存")
    features = [16, 128, 0.00007, 3, 1, 1]  # 示例特征，其中1是'fp16'的编码
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)  # 将特征移动到指定设备
    predicted_accuracy = predict(model, features_tensor, device)  # 确保传入的是张量而不是列表
    print(f'Predicted Accuracy: {predicted_accuracy}')
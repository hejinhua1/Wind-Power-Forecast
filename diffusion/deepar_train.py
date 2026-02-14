import random
import torch.nn.functional as F
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from scipy import integrate
import sys

sys.path.append("..")
from data_provider.data_loader import Dataset_WindPower, Dataset_STGraph, Dataset_Typhoon, Dataset_KGraph
from torch.utils.data import DataLoader
from models.SpatioTemporalGraph import Model
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mse_loss, mape_loss, mase_loss, smape_loss, WeightedMSELoss, DiffLoss
from utils.metrics import metric, R2_score, CRPS, ES, VS
import torch
import torch.nn as nn
from torch import optim
import os
from datetime import datetime
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


class DeepAR(nn.Module):
    def __init__(self, input_dim=9 * 96, hidden_dim=400, num_layers=2, dropout=0.1):
        super(DeepAR, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层 - 预测高斯分布的参数 (均值和标准差)
        self.mu_layer = nn.Linear(hidden_dim, input_dim)
        self.sigma_layer = nn.Linear(hidden_dim, input_dim)

        # 激活函数确保标准差为正
        self.softplus = nn.Softplus()

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)

        # 初始化隐藏状态
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)

        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 预测分布参数
        mu = self.mu_layer(last_output)
        sigma = self.softplus(self.sigma_layer(last_output)) + 1e-6  # 避免除零

        return mu, sigma, hidden

    def sample(self, x, num_samples=1, hidden=None):
        """从学到的分布中采样"""
        mu, sigma, hidden = self.forward(x, hidden)

        # 重复均值和标准差以进行多次采样
        mu_expanded = mu.unsqueeze(1).expand(-1, num_samples, -1)
        sigma_expanded = sigma.unsqueeze(1).expand(-1, num_samples, -1)

        # 从高斯分布中采样
        eps = torch.randn_like(mu_expanded)
        samples = mu_expanded + sigma_expanded * eps

        return samples, mu, sigma


# DeepAR专用损失函数 (负对数似然)
def deepar_loss(mu, sigma, target):
    """计算高斯分布的负对数似然损失"""
    distribution = torch.distributions.Normal(mu, sigma)
    log_prob = distribution.log_prob(target)
    return -log_prob.mean()


if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    class Config:
        def __init__(self):
            self.train_epochs = 50
            self.in_channels = 26
            self.hidden_channels = 16
            self.out_channels = 1
            self.timestep_max = 96
            self.nb_blocks = 2
            self.channels_last = False
            self.show_scores = False
            self.task_name = 'KGformer'
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
            self.num_nodes = 9
            self.learning_rate = 0.0001
            self.batch_size = 64
            self.cond_channels = 26 * 9


    args_withKG = Config()
    # 设置检查点路径和文件名前缀
    checkpoint_path_withKG = "/home/hjh/WindPowerForecast/checkpoints/KGformer_normal_epoch_1.pt"
    checkpoint_withKG = torch.load(checkpoint_path_withKG)
    # 保存模型的路径
    deepar_model_dir = '/home/hjh/WindPowerForecast/deepar_checkpoints'  # 保存模型的目录
    if not os.path.exists(deepar_model_dir):
        os.makedirs(deepar_model_dir)
    # 设置GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 模型定义和训练
    model_withKG = Model(args_withKG).to(device)
    model_withKG.load_state_dict(checkpoint_withKG['model_state_dict'])

    valiset = Dataset_KGraph(flag='test')
    vali_loader = DataLoader(valiset, batch_size=args_withKG.batch_size, shuffle=False, drop_last=True)

    model_withKG.eval()
    errors = []
    conditions = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_adj, batch_em_x, batch_em_y) in enumerate(vali_loader):
            batch_x[:, :-1, :, :] = batch_y[:, :-1, :, -args_withKG.pred_len:]

            batch_x_withKG = torch.cat([batch_x, batch_em_y[:, :, :, -args_withKG.pred_len:]], dim=1)
            batch_x_withKG = batch_x_withKG.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_adj = batch_adj.float().to(device)
            batch_adj_hat = torch.zeros_like(batch_adj).float().to(device)

            outputs_withKG = model_withKG(batch_x_withKG, batch_adj, batch_adj_hat)
            batch_y = batch_y[:, -1, :, -args_withKG.pred_len:].to(device)
            outputs_withKG = outputs_withKG.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            if valiset.scale:
                outputs_withKG = valiset.inverse_transform(outputs_withKG)
                batch_y = valiset.inverse_transform(batch_y)

            pred_withKG = np.clip(outputs_withKG, 0, 1)  # 对点预测结果进行裁剪，确保在0-1之间
            true = batch_y
            # 收集预测误差、条件
            # 计算预测误差
            error = true - pred_withKG
            condition_ = batch_x_withKG.detach().cpu().numpy()
            condition = np.concatenate(
                [condition_[:, :5, :, :], np.expand_dims(pred_withKG, axis=1), condition_[:, 6:, :, :]], axis=1)

            errors.append(error)
            conditions.append(condition)

    # 将预测误差和条件转换为张量
    errors = torch.tensor(np.concatenate(errors, axis=0))
    conditions = torch.tensor(np.concatenate(conditions, axis=0))

    # 准备DeepAR训练数据 - 使用条件作为输入序列
    # 将条件数据重塑为序列格式 (batch_size, seq_len, features)
    condition_reshaped = conditions.view(conditions.shape[0], -1, 9 * 96)

    # 创建序列数据集
    dataset = TensorDataset(condition_reshaped, errors.view(errors.shape[0], -1))
    data_loader = DataLoader(dataset, batch_size=args_withKG.batch_size, shuffle=True)

    # 初始化DeepAR模型
    input_dim = 9 * 96
    deepar = DeepAR(input_dim=input_dim).to(device)
    optimizer = Adam(deepar.parameters(), lr=args_withKG.learning_rate)
    tqdm_epoch = tqdm(range(args_withKG.train_epochs))

    # DeepAR训练循环
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for batch_idx, (condition_seq, target) in enumerate(data_loader):
            condition_seq = condition_seq.float().to(device)
            target = target.float().to(device)

            optimizer.zero_grad()

            # DeepAR前向传播
            mu, sigma, _ = deepar(condition_seq)

            # 计算负对数似然损失
            loss = deepar_loss(mu, sigma, target)

            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * target.shape[0]
            num_items += target.shape[0]

        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

        # 保存检查点
        models_path = os.path.join(deepar_model_dir, f'deepar_{epoch}.pth')
        torch.save(deepar.state_dict(), models_path)

    # # 生成示例
    # with torch.no_grad():
    #     # 从数据加载器中获取一个批次的条件序列
    #     condition_seq_sample, _ = next(iter(data_loader))
    #     condition_seq_sample = condition_seq_sample.float().to(device)

    #     # 从DeepAR中采样
    #     samples, mu, sigma = deepar.sample(condition_seq_sample, num_samples=10)
    #     # samples shape: (batch_size, num_samples, input_dim)
    #     generated = samples.view(-1, 10, 9, 96)  # 重塑为合适的形状
    #     a = 1
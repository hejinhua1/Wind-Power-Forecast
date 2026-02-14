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


class CVAE(nn.Module):
    def __init__(self, input_dim=9*96, label_dim=9*96, hidden_dim=400, latent_dim=20):
        super(CVAE, self).__init__()

        # 编码器（输入+标签共同编码）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 潜在空间参数
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # 解码器（潜在向量+标签共同解码）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # 输出范围[-1,1]
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        # 拼接输入和标签
        x_cond = torch.cat([x, y], dim=1)

        # 编码
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 重参数化
        z = self.reparameterize(mu, logvar)

        # 拼接潜在向量和标签
        z_cond = torch.cat([z, y], dim=1)

        # 解码
        x_recon = self.decoder(z_cond)
        return x_recon, mu, logvar


# 条件VAE专用损失函数
def cvae_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

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
    cvae_model_dir = '/home/hjh/WindPowerForecast/cvae_checkpoints'  # 保存模型的目录
    if not os.path.exists(cvae_model_dir):
        os.makedirs(cvae_model_dir)
    # 设置GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 模型定义和训练
    model_withKG = Model(args_withKG).to(device)

    model_withKG.load_state_dict(checkpoint_withKG['model_state_dict'])

    valiset = Dataset_KGraph(flag='test')
    vali_loader = DataLoader(valiset, batch_size=args_withKG.batch_size, shuffle=False, drop_last=True)

    model_withKG.eval()
    erros = []
    conditions = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_adj, batch_em_x, batch_em_y) in enumerate(vali_loader):
            batch_x[:, :-1, :, :] = batch_y[:, :-1, :, -args_withKG.pred_len:]

            batch_x_withKG = torch.cat([batch_x, batch_em_y[:, :, :, -args_withKG.pred_len:]], dim=1)
            batch_x_withKG = batch_x_withKG.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_adj = batch_adj.float().to(device)
            batch_adj_hat = torch.zeros_like(batch_adj).float().to(device)
            ########################################################
            outputs_withKG = model_withKG(batch_x_withKG, batch_adj, batch_adj_hat)
            batch_y = batch_y[:, -1, :, -args_withKG.pred_len:].to(device)
            outputs_withKG = outputs_withKG.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            if valiset.scale:
                outputs_withKG = valiset.inverse_transform(outputs_withKG)
                batch_y = valiset.inverse_transform(batch_y)

            pred_withKG = np.clip(outputs_withKG, 0, 1) # 对点预测结果进行裁剪，确保在0-1之间
            true = batch_y
            # 收集预测误差、条件
            # 计算预测误差
            erro = true - pred_withKG
            condition_ = batch_x_withKG.detach().cpu().numpy()
            condition = np.concatenate([condition_[:, :5, :, :], np.expand_dims(pred_withKG, axis=1), condition_[:, 6:, :, :]], axis=1)

            erros.append(erro)
            conditions.append(condition)
    # 将预测误差和条件转换为张量
    erros = torch.tensor(np.concatenate(erros, axis=0))
    conditions = torch.tensor(np.concatenate(conditions, axis=0))

    dataset = TensorDataset(erros, conditions)
    data_loader = DataLoader(dataset, batch_size=args_withKG.batch_size, shuffle=True)

    input_dim = 9*96
    label_dim = 9*96
    cvae = CVAE(input_dim=input_dim, label_dim=label_dim).to(device)
    optimizer = Adam(cvae.parameters(), lr=args_withKG.learning_rate)
    tqdm_epoch = tqdm(range(args_withKG.train_epochs))

    # 训练循环（保持不变）
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for batch_idx, (data, condition) in enumerate(data_loader):
            data = data.view(-1, input_dim).to(device)
            y = condition[:, 5, :, :].view(-1, label_dim).to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = cvae(data, y)
            loss = cvae_loss(recon_batch, data, mu, logvar)

            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * data.shape[0]
            num_items += data.shape[0]
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        models_path = os.path.join(cvae_model_dir, f'cvae_{epoch}.pth')
        torch.save(cvae.state_dict(), models_path)

    # # 条件生成示例
    # with torch.no_grad():
    #     y_gen = y.float()
    #     z = torch.randn(64, 20).to(device)
    #     z_cond = torch.cat([z, y_gen], dim=1)
    #     generated = cvae.decoder(z_cond).view(64, 9, 96)
    #     a = 1


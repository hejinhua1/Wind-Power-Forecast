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


class Generator(nn.Module):
    """生成器：从潜在向量和条件生成数据"""

    def __init__(self, input_dim=9 * 96, label_dim=9 * 96, hidden_dim=400, latent_dim=20):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.label_dim = label_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # 输出范围[-1,1]
        )

    def forward(self, z, y):
        # 拼接潜在向量和条件
        z_cond = torch.cat([z, y], dim=1)
        return self.net(z_cond)


class Discriminator(nn.Module):
    """判别器：判断数据是否真实，同时考虑条件"""

    def __init__(self, input_dim=9 * 96, label_dim=9 * 96, hidden_dim=400):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1),
            # 移除了Sigmoid，因为WGAN使用线性输出
        )

    def forward(self, x, y):
        # 拼接数据和条件
        x_cond = torch.cat([x, y], dim=1)
        return self.net(x_cond)


class WGAN:
    def __init__(self, input_dim=9 * 96, label_dim=9 * 96, hidden_dim=400, latent_dim=20,
                 device='cuda', n_critic=5, lambda_gp=10):
        self.device = device
        self.n_critic = n_critic  # 判别器训练次数
        self.lambda_gp = lambda_gp  # 梯度惩罚系数

        self.generator = Generator(input_dim, label_dim, hidden_dim, latent_dim).to(device)
        self.discriminator = Discriminator(input_dim, label_dim, hidden_dim).to(device)

        # 使用RMSprop优化器，这是WGAN论文推荐的
        self.optimizer_G = optim.RMSprop(self.generator.parameters(), lr=5e-5)
        self.optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=5e-5)

    def compute_gradient_penalty(self, real_samples, fake_samples, conditions):
        """计算梯度惩罚"""
        batch_size = real_samples.size(0)

        # 在真实数据和生成数据之间随机插值
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_samples)

        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates, conditions)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def train_step(self, real_data, conditions):
        batch_size = real_data.size(0)

        # 训练判别器
        d_losses = []
        for _ in range(self.n_critic):
            # 随机噪声
            z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)

            # 生成假数据
            fake_data = self.generator(z, conditions)

            # 判别器对真实和假数据的评分
            real_validity = self.discriminator(real_data, conditions)
            fake_validity = self.discriminator(fake_data, conditions)

            # 梯度惩罚
            gradient_penalty = self.compute_gradient_penalty(real_data, fake_data, conditions)

            # 判别器损失
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()

            d_losses.append(d_loss.item())

        # 训练生成器
        z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        fake_data = self.generator(z, conditions)
        fake_validity = self.discriminator(fake_data, conditions)
        g_loss = -torch.mean(fake_validity)

        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        return np.mean(d_losses), g_loss.item()

    def generate(self, conditions, num_samples=None):
        """生成样本"""
        self.generator.eval()
        with torch.no_grad():
            if num_samples is None:
                num_samples = conditions.size(0)

            z = torch.randn(num_samples, self.generator.latent_dim).to(self.device)
            if num_samples != conditions.size(0):
                # 如果条件数量不匹配，重复条件
                conditions = conditions[:1].repeat(num_samples, 1)

            generated = self.generator(z, conditions)
        return generated


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
    wgan_model_dir = '/home/hjh/WindPowerForecast/wgan_checkpoints'  # 保存模型的目录
    if not os.path.exists(wgan_model_dir):
        os.makedirs(wgan_model_dir)
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
            erro = true - pred_withKG
            condition_ = batch_x_withKG.detach().cpu().numpy()
            condition = np.concatenate(
                [condition_[:, :5, :, :], np.expand_dims(pred_withKG, axis=1), condition_[:, 6:, :, :]], axis=1)

            erros.append(erro)
            conditions.append(condition)

    # 将预测误差和条件转换为张量
    erros = torch.tensor(np.concatenate(erros, axis=0))
    conditions = torch.tensor(np.concatenate(conditions, axis=0))

    dataset = TensorDataset(erros, conditions)
    data_loader = DataLoader(dataset, batch_size=args_withKG.batch_size, shuffle=True)

    input_dim = 9 * 96
    label_dim = 9 * 96

    # 创建WGAN模型
    wgan = WGAN(input_dim=input_dim, label_dim=label_dim, device=device)

    tqdm_epoch = tqdm(range(args_withKG.train_epochs))

    # WGAN训练循环
    for epoch in tqdm_epoch:
        d_losses = []
        g_losses = []

        for batch_idx, (data, condition) in enumerate(data_loader):
            data = data.view(-1, input_dim).to(device)
            y = condition[:, 5, :, :].view(-1, label_dim).to(device)

            d_loss, g_loss = wgan.train_step(data, y)

            d_losses.append(d_loss)
            g_losses.append(g_loss)

        avg_d_loss = np.mean(d_losses)
        avg_g_loss = np.mean(g_losses)

        tqdm_epoch.set_description(f'D Loss: {avg_d_loss:.5f}, G Loss: {avg_g_loss:.5f}')

        # 保存模型
        if epoch % 10 == 0:
            models_path = os.path.join(wgan_model_dir, f'wgan_generator_{epoch}.pth')
            torch.save(wgan.generator.state_dict(), models_path)

            models_path = os.path.join(wgan_model_dir, f'wgan_discriminator_{epoch}.pth')
            torch.save(wgan.discriminator.state_dict(), models_path)

    # 最终保存模型
    torch.save(wgan.generator.state_dict(), os.path.join(wgan_model_dir, 'wgan_generator_final.pth'))
    torch.save(wgan.discriminator.state_dict(), os.path.join(wgan_model_dir, 'wgan_discriminator_final.pth'))

    # # 生成示例
    # with torch.no_grad():
    #     # 使用一些条件生成样本
    #     sample_conditions = y[:64].float()  # 取前64个条件
    #     generated = wgan.generate(sample_conditions).view(64, 9, 96)
    #     a = 1
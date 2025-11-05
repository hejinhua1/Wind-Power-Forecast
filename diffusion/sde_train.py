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


# 实现一个简单的sde

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None]




class CrossAttention(nn.Module):
    def __init__(self, query_dim, cond_dim, embed_dim=256):
        super().__init__()
        self.query_proj = nn.Conv1d(query_dim, embed_dim, 1)
        self.cond_proj = nn.Conv1d(cond_dim, embed_dim, 1)
        self.key = nn.Conv1d(embed_dim, embed_dim, 1)
        self.value = nn.Conv1d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv1d(embed_dim, query_dim, 1)

    def forward(self, x, cond):
        # x: [B, D, T], cond: [B, C, T]
        query = self.query_proj(x).transpose(1, 2)  # [B, T, E]
        cond_proj = self.cond_proj(cond)
        key = self.key(cond_proj).transpose(1, 2)  # [B, T, E]
        value = self.value(cond_proj).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, value).transpose(1, 2)  # [B, E, T]

        return self.out_proj(output) + x  # 残差连接


class ResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.attn = CrossAttention(dim, cond_dim)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x, cond):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.attn(x, cond)
        return self.norm(x + residual)


class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, in_channels=9, cond_channels=26*9, base_dim=16*9, embed_dim=64):
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        self.dense1 = Dense(embed_dim, base_dim)
        # 初始卷积层
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_dim, 3, padding=1),
            nn.BatchNorm1d(base_dim),
            nn.ReLU()
        )

        # 下采样部分
        self.down = nn.Sequential(
            nn.Conv1d(base_dim, base_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm1d(base_dim * 2),
            nn.ReLU()
        )

        # 条件处理
        self.cond_proj = nn.Conv1d(cond_channels, base_dim * 2, 1) # TODO: 这里可能是一个瓶颈
        self.res_block = ResidualBlock(base_dim * 2, base_dim * 2)

        # 上采样部分
        self.up = nn.Sequential(
            nn.ConvTranspose1d(base_dim * 2, base_dim, 3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(base_dim),
            nn.ReLU()
        )

        # 输出层
        self.final_conv = nn.Conv1d(base_dim, in_channels, 3, padding=1)
        self.marginal_prob_std = marginal_prob_std
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, condition, t):
        embed = self.act(self.embed(t))  # [B, embed_dim]
        embed = self.dense1(embed)  # [B, base_dim, 1]

        # 输入处理
        B, N, T = x.shape
        x = self.init_conv(x) + embed  # [B, 64, T]

        # 下采样
        x_down = self.down(x)  # [B, 128, T//2]

        # 条件处理
        # condition shape [B,9*26,T]
        cond = self.cond_proj(F.avg_pool1d(condition, 2))  # [B, 128, T//2]

        # 残差块+交叉注意力
        x_down = self.res_block(x_down, cond)

        # 上采样
        x_up = self.up(x_down)  # [B, 64, T]

        # 跳过连接
        x = x + x_up  # 融合浅层和深层特征 # [B, 64, T]

        # 最终输出 shape [B,9,T]
        return self.final_conv(x) / self.marginal_prob_std(t)[:, None, None]

def marginal_prob_mean(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """

    return torch.tensor(0)





def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    t = torch.tensor(t)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(0), torch.tensor(sigma ** t)




#@title Define the loss function (double click to expand or collapse)

def loss_fn(model, x, condition, prob_mean, prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    # x [B,9,96]
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x) # z [B,9,96]
    mean_ = prob_mean(random_t)
    std = prob_std(random_t)
    perturbed_x = x + z * std[:, None, None] # perturbed_x [B,9,96]
    score = model(perturbed_x, condition, random_t) # score [B,9,96]
    loss = torch.mean(torch.sum((score * std[:, None, None] + z)**2, dim=(1, 2)))
    return loss


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
    # 设置SDE参数
    sigma = 25.0
    marginal_prob_mean_fn = functools.partial(marginal_prob_mean, sigma=sigma)
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    # 设置检查点路径和文件名前缀
    checkpoint_path_withKG = "/home/hjh/WindPowerForecast/checkpoints/KGformer_normal_epoch_3.pt"
    checkpoint_withKG = torch.load(checkpoint_path_withKG)
    # 保存模型的路径
    sde_model_dir = '/home/hjh/WindPowerForecast/sde_checkpoints'  # 保存模型的目录
    if not os.path.exists(sde_model_dir):
        os.makedirs(sde_model_dir)
    # 设置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

            batch_x[:, :-1, :, :] = batch_y[:, :-1, :, -args_withKG.pred_len:] # 舍弃历史的天气预测信息，只用未来的天气预测信息与历史的功率信息
            # batch_x shape [B,6,9,pre_len]
            # batch_em_y shape [B,20,9,label_len+pre_len]

            batch_x_withKG = torch.cat([batch_x, batch_em_y[:, :, :, -args_withKG.pred_len:]], dim=1) # 补上未来的台风预报的语义嵌入，增加20维
            batch_x_withKG = batch_x_withKG.float().to(device)  # batch_x_withKG shape [B,26,9,pre_len]

            batch_y = batch_y.float().to(device)
            batch_adj = batch_adj.float().to(device)
            batch_adj_hat = torch.zeros_like(batch_adj).float().to(device)
            ########################################################
            outputs_withKG = model_withKG(batch_x_withKG, batch_adj, batch_adj_hat)
            batch_y = batch_y[:, -1, :, -args_withKG.pred_len:].to(device)
            outputs_withKG = outputs_withKG.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy() # batch_y shape [B,9,pre_len]

            if valiset.scale:
                outputs_withKG = valiset.inverse_transform(outputs_withKG)
                batch_y = valiset.inverse_transform(batch_y)

            pred_withKG = outputs_withKG #[B,9,pre_len]
            true = batch_y
            # 收集预测误差、条件
            # 计算预测误差
            erro = true - pred_withKG
            condition_ = batch_x_withKG.detach().cpu().numpy() #[B,26,9,pre_len]
            # 把条件中的历史功率替换成预测功率
            condition = np.concatenate([condition_[:, :5, :, :], np.expand_dims(pred_withKG, axis=1), condition_[:, 6:, :, :]], axis=1)
            # [B,26,9,pre_len]
            erros.append(erro)
            conditions.append(condition)
    # 将预测误差和条件转换为张量
    erros = torch.tensor(np.concatenate(erros, axis=0)) #[B,9,pre_len]
    conditions = torch.tensor(np.concatenate(conditions, axis=0)) #[B,26,9,pre_len]

    dataset = TensorDataset(erros, conditions)
    data_loader = DataLoader(dataset, batch_size=args_withKG.batch_size, shuffle=True)

    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, cond_channels=args_withKG.cond_channels).to(device)
    optimizer = Adam(score_model.parameters(), lr=args_withKG.learning_rate)
    tqdm_epoch = tqdm(range(args_withKG.train_epochs))


    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for i, (x, condition_) in enumerate(data_loader):
            x = x.to(device) # x shape [B,9,pred_len]
            # condition_ shape [B, 9, 26, pred_len]
            condition = condition_.reshape(condition_.shape[0], -1, condition_.shape[3]).to(device) # condition shape [B, 9*26, pred_len]
            loss = loss_fn(model=score_model, x=x, condition=condition, prob_mean=marginal_prob_mean_fn, prob_std=marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        models_path = os.path.join(sde_model_dir, f'sde_{epoch}.pth')
        torch.save(score_model.state_dict(), models_path)





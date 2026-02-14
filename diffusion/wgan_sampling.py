import random
from torchvision.utils import make_grid
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from scipy import integrate
import os
from diffusion_sde_test import marginal_prob_mean, marginal_prob_std, diffusion_coeff
from diffusion_sde_test import ScoreNet
from wgan_train import Generator  # 导入WGAN的生成器
import matplotlib.pyplot as plt

import sys

sys.path.append("..")
from data_provider.data_loader import Dataset_WindPower, Dataset_STGraph, Dataset_Typhoon, Dataset_KGraph, \
    Dataset_Typhoon_KGraph
from torch.utils.data import DataLoader
from models.SpatioTemporalGraph import Model
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mse_loss, mape_loss, mase_loss, smape_loss, WeightedMSELoss, DiffLoss
from utils.metrics import metric, R2_score, CRPS, ES, VS
from torch import optim
import os
from datetime import datetime
import time
import warnings
import pandas as pd
import math

plt.switch_backend('agg')
warnings.filterwarnings('ignore')


class WGANGenerator:
    """WGAN生成器封装类，用于采样"""

    def __init__(self, input_dim=9 * 96, label_dim=9 * 96, hidden_dim=400, latent_dim=20, device='cuda'):
        self.device = device
        self.latent_dim = latent_dim
        self.generator = Generator(input_dim, label_dim, hidden_dim, latent_dim).to(device)

    def load_model(self, model_path):
        """加载生成器模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint)
        self.generator.eval()

    def generate_samples(self, conditions, num_samples=None):
        """生成样本

        Args:
            conditions: 条件张量 [B, label_dim]
            num_samples: 要生成的样本数量，如果为None则使用conditions的batch size

        Returns:
            生成的样本 [num_samples, 9, 96]
        """
        self.generator.eval()
        with torch.no_grad():
            if num_samples is None:
                num_samples = conditions.size(0)

            # 生成随机噪声
            z = torch.randn(num_samples, self.latent_dim).to(self.device)

            # 如果条件数量不匹配，处理条件
            if num_samples != conditions.size(0):
                conditions = conditions[:1].repeat(num_samples, 1)

            # 生成样本
            generated = self.generator(z, conditions)
            return generated.view(num_samples, 9, 96)


if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    class Config:
        def __init__(self):
            self.train_epochs = 100
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
            self.batch_size = 95
            self.cond_channels = 26 * 9


    args_withKG = Config()
    num_steps = 500
    sample_batch_size = 50
    input_dim = 9 * 96
    label_dim = 9 * 96

    # 设置检查点路径和文件名前缀
    wgan_result_dir = '/home/hjh/WindPowerForecast/test_results/wgan'  # 修改为WGAN结果目录
    if not os.path.exists(wgan_result_dir):
        os.makedirs(wgan_result_dir)

    checkpoint_path_withKG = "/home/hjh/WindPowerForecast/checkpoints/KGformer_normal_epoch_1.pt"
    wgan_model_path = '/home/hjh/WindPowerForecast/wgan_checkpoints/wgan_generator_final.pth'  # WGAN生成器路径

    checkpoint_withKG = torch.load(checkpoint_path_withKG)

    # 设置GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 模型定义和训练
    model_withKG = Model(args_withKG).to(device)
    model_withKG.load_state_dict(checkpoint_withKG['model_state_dict'])

    # 初始化WGAN生成器
    wgan_generator = WGANGenerator(
        input_dim=input_dim,
        label_dim=label_dim,
        device=device
    )
    wgan_generator.load_model(wgan_model_path)

    valiset = Dataset_KGraph(flag='test')
    vali_loader = DataLoader(valiset, batch_size=args_withKG.batch_size, shuffle=False, drop_last=True)

    model_withKG.eval()

    for i, (batch_x, batch_y, batch_adj, batch_em_x, batch_em_y) in enumerate(vali_loader):
        print('i:', i, 'total:', len(vali_loader))
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

        pred_withKG = np.clip(outputs_withKG, 0, 1)  # [B, 9, 96]
        true = batch_y  # [B, 9, 96]

        condition_ = batch_x_withKG.detach().cpu().numpy()  # condition_ shape [B,26,9,pre_len]

        # 构建条件张量
        if args_withKG.batch_size == 1:
            condition = np.concatenate([
                condition_[:, :5, :, :],
                np.expand_dims(np.expand_dims(pred_withKG, axis=0), axis=0),
                condition_[:, 6:, :, :]
            ], axis=1)
        else:
            condition = np.concatenate([
                condition_[:, :5, :, :],
                np.expand_dims(pred_withKG, axis=1),
                condition_[:, 6:, :, :]
            ], axis=1)

        condition = torch.from_numpy(condition).float().to(device)

        # 使用WGAN进行条件采样
        final_pred_samples = []

        # 对每个batch进行多次采样
        for _ in range(sample_batch_size):
            # 获取条件（第5个通道的预测功率）
            batch_conditioning = condition[:, 5, :, :].view(-1, label_dim).to(device)

            # 使用WGAN生成误差样本
            error_samples = wgan_generator.generate_samples(batch_conditioning)

            # 将误差限制在合理范围内（根据你的数据特性调整）
            errors = error_samples.clamp(-0.5, 0.5)  # [B, 9, 96]

            # 计算最终预测
            final_pred = pred_withKG + errors.detach().cpu().numpy()  # [B, 9, 96]
            final_pred = np.clip(final_pred, 0, 1)  # 对结果进行裁剪，输出在0-1之间
            final_pred_samples.append(final_pred)

        final_pred_samples = np.array(final_pred_samples)  # [sample_batch_size, B, 9, 96]
        final_pred = final_pred_samples.mean(axis=0)  # [B, 9, 96]
        final_true = true  # [B, 9, 96]

        # 可视化结果
        for windfarm in range(9):
            plt.figure()
            plt.plot(final_true[0, windfarm, :], label='GroundTruth', linewidth=2)
            plt.plot(final_pred[0, windfarm, :], label='Prediction', linewidth=2)

            # 绘制所有样本
            for j in range(sample_batch_size):
                plt.plot(final_pred_samples[j, 0, windfarm, :], linewidth=0.2, color='gray')

            plt.legend()

            wind_farm_dir = os.path.join(wgan_result_dir, f'windfarm_{windfarm}')
            if not os.path.exists(wind_farm_dir):
                os.makedirs(wind_farm_dir)

            ima_path = os.path.join(wind_farm_dir, f'sample_{i}.svg')
            plt.savefig(ima_path, format='svg', bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()

        print(f'Completed batch {i}, saved results to {wgan_result_dir}')

    print("WGAN sampling completed!")
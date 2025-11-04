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
from diffusion_sde import marginal_prob_mean, marginal_prob_std, diffusion_coeff, ScoreNet
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from data_provider.data_loader import Dataset_WindPower, Dataset_STGraph, Dataset_Typhoon, Dataset_KGraph, Dataset_Typhoon_KGraph
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


# 简单sde的采样

def Euler_Maruyama_sampler(score_model,
                           marginal_prob_mean,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_condition,
                           batch_size=64,
                           num_steps=500,
                           device='cuda',
                           eps=1e-3):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 9, 96, device=device) \
             * marginal_prob_std(t)[:, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            f, g = diffusion_coeff(batch_time_step)
            mean_x = x + ((g ** 2)[:, None, None] * score_model(x, batch_condition, batch_time_step)) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)
            # Do not include any noise in the last sampling step.
    return mean_x


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
            self.batch_size = 1

            self.cond_channels = 26 * 9

    args_withKG = Config()
    num_steps = 500
    sample_batch_size = 100
    # 设置检查点路径和文件名前缀
    sde_result_dir = '/home/hjh/WindPowerForecast/test_results/sde'
    if not os.path.exists(sde_result_dir):
        os.makedirs(sde_result_dir)
    checkpoint_path_withKG = "/home/hjh/WindPowerForecast/checkpoints/KGformer_normal_epoch_3.pt"
    sde_model_path = '/home/hjh/WindPowerForecast/sde_checkpoints/sde_1.pth'
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

    sde_checkpoint = torch.load(sde_model_path, map_location=device)
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std, cond_channels=args_withKG.cond_channels).to(device)
    score_model.load_state_dict(sde_checkpoint)
    sampler = Euler_Maruyama_sampler

    valiset = Dataset_Typhoon_KGraph(flag='test')
    vali_loader = DataLoader(valiset, batch_size=args_withKG.batch_size, shuffle=False, drop_last=True)

    model_withKG.eval()
    erros = []
    conditions = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_adj, batch_em_x, batch_em_y) in enumerate(vali_loader):
            print('i:', i, 'total:', len(vali_loader))
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

            pred_withKG = outputs_withKG  #[1, 9, 96]
            true = batch_y  #[1, 9, 96]
            # 收集预测误差、条件
            # 计算预测误差
            erro = true - pred_withKG
            condition_ = batch_x_withKG.detach().cpu().numpy()
            if args_withKG.batch_size == 1:
                condition = np.concatenate([condition_[:, :5, :, :], np.expand_dims(np.expand_dims(pred_withKG, axis=0), axis=0), condition_[:, 6:, :, :]], axis=1)
            else:
                condition = np.concatenate([condition_[:, :5, :, :], np.expand_dims(pred_withKG, axis=1), condition_[:, 6:, :, :]], axis=1)
            condition = torch.from_numpy(condition).float().to(device)
            batch_condition = condition.repeat(sample_batch_size, 1, 1, 1)
            batch_condition = batch_condition.reshape(batch_condition.shape[0], -1, batch_condition.shape[3])

            # 开始进行条件采样
            samples = sampler(score_model,
                              marginal_prob_mean,
                              marginal_prob_std,
                              diffusion_coeff,
                              batch_condition=batch_condition,
                              batch_size=sample_batch_size,
                              num_steps=num_steps,
                              device='cuda')

            errors = samples.clamp(-0.5, 0.5)  #[sample_batch_size, 9, 96]

            final_pred_samples = pred_withKG + errors.detach().cpu().numpy() #[sample_batch_size, 9, 96]
            final_pred = final_pred_samples.mean(axis=0)  #[9, 96]
            final_true = true[0, :, :]  #[9, 96]

            for windfarm in range(9):
                plt.figure()
                plt.plot(final_true[windfarm, :], label='GroundTruth', linewidth=2)
                plt.plot(final_pred[windfarm, :], label='Prediction', linewidth=2)
                for j in range(sample_batch_size):
                    plt.plot(final_pred_samples[j, windfarm, :], linewidth=0.2, color='gray', alpha=0.5)
                plt.legend()
                wind_farm_dir = os.path.join(sde_result_dir, f'windfarm_{windfarm}')
                if not os.path.exists(wind_farm_dir):
                    os.makedirs(wind_farm_dir)
                ima_path = os.path.join(wind_farm_dir, f'sample_{i}.svg')
                plt.savefig(ima_path, format='svg', bbox_inches='tight', dpi=300, facecolor='white')





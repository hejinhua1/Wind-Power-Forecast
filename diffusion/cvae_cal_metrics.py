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
from cvae_train import CVAE
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
            self.batch_size = 96

            self.cond_channels = 26 * 9

    args_withKG = Config()
    num_steps = 500
    sample_batch_size = 50
    input_dim = 9*96
    label_dim = 9*96
    # 设置检查点路径和文件名前缀
    cvae_result_dir = '/home/hjh/WindPowerForecast/test_results/cvae'
    if not os.path.exists(cvae_result_dir):
        os.makedirs(cvae_result_dir)
    checkpoint_path_withKG = "/home/hjh/WindPowerForecast/checkpoints/KGformer_normal_epoch_1.pt"
    cvae_model_path = '/home/hjh/WindPowerForecast/cvae_checkpoints/cvae_5.pth'
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

    cvae_checkpoint = torch.load(cvae_model_path, map_location=device)
    cvae_model = CVAE(input_dim=input_dim, label_dim=label_dim).to(device)
    cvae_model.load_state_dict(cvae_checkpoint)

    valiset = Dataset_Typhoon_KGraph(flag='test')
    # valiset = Dataset_KGraph(flag='test')
    vali_loader = DataLoader(valiset, batch_size=args_withKG.batch_size, shuffle=False, drop_last=False)

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


            # 开始进行条件采样
            final_pred_samples = []
            for _ in range(sample_batch_size):
                batch_conditioning = condition[:, 5, :, :].view(-1, label_dim).to(device)
                z = torch.randn(args_withKG.batch_size, 20).to(device)
                z_cond = torch.cat([z, batch_conditioning], dim=1)
                samples = cvae_model.decoder(z_cond).view(args_withKG.batch_size, 9, 96)

                errors = samples.clamp(-0.5, 0.5)  #[batch_size, 9, 96]

                final_pred = pred_withKG + errors.detach().cpu().numpy() #[batch_size, 9, 96]
                final_pred_samples.append(final_pred)

            final_pred_samples = np.array(final_pred_samples)  #[50, batch_size, 9, 96]
            final_pred = final_pred_samples.mean(axis=0)  #[batch_size, 9, 96]
            final_true = true  #[batch_size, 9, 96]

            # 计算所有场站的平均
            forecasts = np.mean(final_pred_samples, axis=2)  #[50, batch_size, 96]
            observations = np.mean(final_true, axis=1)  #[batch_size, 96]

            # CRPS
            crps = CRPS(observations[:,:48], forecasts[:,:,:48], m=sample_batch_size)
            es = ES(observations[:,:48], forecasts[:,:,:48], m=sample_batch_size)
            vs = VS(observations[:,:48], forecasts[:,:,:48], m=sample_batch_size)

            print('crps:', crps, 'es:', es, 'vs:', vs)








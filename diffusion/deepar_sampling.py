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
from deepar_train import DeepAR  # 修改为导入DeepAR
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
    deepar_result_dir = '/home/hjh/WindPowerForecast/test_results/deepar'  # 修改结果目录
    if not os.path.exists(deepar_result_dir):
        os.makedirs(deepar_result_dir)
    checkpoint_path_withKG = "/home/hjh/WindPowerForecast/checkpoints/KGformer_normal_epoch_1.pt"
    deepar_model_path = '/home/hjh/WindPowerForecast/deepar_checkpoints/deepar_20.pth'  # 修改模型路径
    checkpoint_withKG = torch.load(checkpoint_path_withKG)

    # 设置GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 模型定义和训练
    model_withKG = Model(args_withKG).to(device)
    model_withKG.load_state_dict(checkpoint_withKG['model_state_dict'])

    # 加载DeepAR模型
    deepar_checkpoint = torch.load(deepar_model_path, map_location=device)
    deepar_model = DeepAR(input_dim=input_dim).to(device)
    deepar_model.load_state_dict(deepar_checkpoint)
    deepar_model.eval()

    # valiset = Dataset_Typhoon_KGraph(flag='test')
    valiset = Dataset_KGraph(flag='test')
    vali_loader = DataLoader(valiset, batch_size=args_withKG.batch_size, shuffle=False, drop_last=True)

    model_withKG.eval()

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_adj, batch_em_x, batch_em_y) in enumerate(vali_loader):
            print('i:', i, 'total:', len(vali_loader))
            batch_x[:, :-1, :, :] = batch_y[:, :-1, :, -args_withKG.pred_len:]
            # batch_x shape [B,6,9,pre_len]
            # batch_em_y shape [B,20,9,label_len+pre_len]

            batch_x_withKG = torch.cat([batch_x, batch_em_y[:, :, :, -args_withKG.pred_len:]], dim=1)
            batch_x_withKG = batch_x_withKG.float().to(device)
            # batch_x_withKG shape [B,26,9,pre_len]

            batch_y = batch_y.float().to(device)  # batch_y shape [B,6,9,label_len+pre_len]
            batch_adj = batch_adj.float().to(device)
            batch_adj_hat = torch.zeros_like(batch_adj).float().to(device)

            ########################################################
            outputs_withKG = model_withKG(batch_x_withKG, batch_adj,
                                          batch_adj_hat)  # outputs_withKG shape [B,9,pre_len]
            batch_y = batch_y[:, -1, :, -args_withKG.pred_len:].to(device)  # batch_y shape [B,9,pre_len]
            outputs_withKG = outputs_withKG.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            if valiset.scale:
                outputs_withKG = valiset.inverse_transform(outputs_withKG)
                batch_y = valiset.inverse_transform(batch_y)

            pred_withKG = np.clip(outputs_withKG, 0, 1)  # [B, 9, 96],# 对点预测结果进行裁剪，确保在0-1之间
            true = batch_y  # [B, 9, 96]

            # 准备条件数据
            condition_ = batch_x_withKG.detach().cpu().numpy()  # condition_ shape [B,26,9,pre_len]
            # 0-4维度表示天气预测，第5维度表示功率历史值，第6-25维度表示台风路径嵌入
            if args_withKG.batch_size == 1:
                condition = np.concatenate(
                    [condition_[:, :5, :, :], np.expand_dims(np.expand_dims(pred_withKG, axis=0), axis=0),
                     condition_[:, 6:, :, :]], axis=1)
            else:
                condition = np.concatenate(
                    [condition_[:, :5, :, :], np.expand_dims(pred_withKG, axis=1), condition_[:, 6:, :, :]], axis=1)
                # condition shape [B,26,9,pre_len]

            # 将条件数据重塑为DeepAR需要的序列格式 [batch_size, seq_len, features]
            condition_reshaped = condition.reshape(condition.shape[0], -1, 9 * 96)
            condition_tensor = torch.from_numpy(condition_reshaped).float().to(device)

            # 使用DeepAR进行采样
            final_pred_samples = []
            for _ in range(sample_batch_size):
                # DeepAR采样 - 直接从学到的分布中生成样本
                samples, mu, sigma = deepar_model.sample(condition_tensor, num_samples=1)

                # samples shape: [batch_size, 1, input_dim] -> [batch_size, input_dim]
                samples = samples.squeeze(1)

                # 将采样结果重塑为 [batch_size, 9, 96]
                errors = samples.view(args_withKG.batch_size, 9, 96)

                # 对误差进行裁剪，确保在合理范围内
                errors = errors.clamp(-0.5, 0.5)  # [B, 9, 96]

                # 生成最终预测
                final_pred = pred_withKG + errors.detach().cpu().numpy()  # [B, 9, 96]
                final_pred = np.clip(final_pred, 0, 1)  # 对结果进行裁剪，输出在0-1之间
                final_pred_samples.append(final_pred)

            final_pred_samples = np.array(final_pred_samples)  # [sample_batch_size, B, 9, 96]
            final_pred = final_pred_samples.mean(axis=0)  # [B, 9, 96]
            final_true = true  # [B, 9, 96]

            # 可视化结果
            for windfarm in range(9):
                plt.plot(final_true[0, windfarm, :], label='GroundTruth', linewidth=2)
                plt.plot(final_pred[0, windfarm, :], label='Prediction', linewidth=2)

                # 绘制所有样本
                for j in range(sample_batch_size):
                    plt.plot(final_pred_samples[j, 0, windfarm, :], linewidth=0.2, color='gray')


                plt.legend()

                wind_farm_dir = os.path.join(deepar_result_dir, f'windfarm_{windfarm}')
                if not os.path.exists(wind_farm_dir):
                    os.makedirs(wind_farm_dir)
                ima_path = os.path.join(wind_farm_dir, f'sample_{i}.svg')
                plt.savefig(ima_path, format='svg', bbox_inches='tight', dpi=300, facecolor='white')
                plt.close()

            print(f"Completed batch {i + 1}/{len(vali_loader)}")
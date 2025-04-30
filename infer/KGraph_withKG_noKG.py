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

plt.switch_backend('agg')
warnings.filterwarnings('ignore')



def visual_withKG_noKG(true, preds_withKG, preds_noKG, name):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.plot(preds_withKG, label='Predict_with', linewidth=2)
    plt.plot(preds_noKG, label='Predict_no', linewidth=2)

    plt.legend()
    # 保存矢量图
    plt.savefig(name, format='svg', bbox_inches='tight', dpi=300, facecolor='white')




if __name__ == '__main__':
    class Config_withKG:
        def __init__(self):
            self.train_epochs = 10
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

    class Config_noKG:
        def __init__(self):
            self.train_epochs = 20
            self.in_channels = 6
            self.hidden_channels = 8
            self.out_channels = 1
            self.timestep_max = 96
            self.nb_blocks = 1
            self.channels_last = False
            self.show_scores = False
            self.task_name = 'KGformer'
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
            self.num_nodes = 9
            self.learning_rate = 0.01
            self.batch_size = 16

    args_withKG = Config_withKG()
    args_noKG = Config_noKG()

    result_dir = '/home/hjh/WindPowerForecast/test_results/'
    # 设置检查点路径和文件名前缀
    checkpoint_path_withKG = "/home/hjh/WindPowerForecast/checkpoints/KGformer_normal_epoch_3.pt"
    checkpoint_path_noKG = "/home/hjh/WindPowerForecast/checkpoints/KGformer_noKG_epoch_6.pt"
    checkpoint_withKG = torch.load(checkpoint_path_withKG)
    checkpoint_noKG = torch.load(checkpoint_path_noKG)
    # 设置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型定义和训练
    model_withKG = Model(args_withKG).to(device)
    model_noKG = Model(args_noKG).to(device)
    model_withKG.load_state_dict(checkpoint_withKG['model_state_dict'])
    model_noKG.load_state_dict(checkpoint_noKG['model_state_dict'])


    valiset = Dataset_KGraph(flag='test')
    vali_loader = DataLoader(valiset, batch_size=args_withKG.batch_size, shuffle=False, drop_last=True)



    model_withKG.eval()
    model_noKG.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_adj, batch_em_x, batch_em_y) in enumerate(vali_loader):
            batch_x[:, :-1, :, :] = batch_y[:, :-1, :, -args_withKG.pred_len:]

            batch_x_noKG = batch_x.clone()
            batch_x_withKG = torch.cat([batch_x, batch_em_y[:, :, :, -args_withKG.pred_len:]], dim=1)

            batch_x_noKG = batch_x_noKG.float().to(device)
            batch_x_withKG = batch_x_withKG.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_adj = batch_adj.float().to(device)
            batch_adj_hat = torch.zeros_like(batch_adj).float().to(device)
            ########################################################
            outputs_withKG = model_withKG(batch_x_withKG, batch_adj, batch_adj_hat)
            outputs_noKG = model_noKG(batch_x_noKG, batch_adj, batch_adj_hat)
            batch_y = batch_y[:, -1, :, -args_withKG.pred_len:].to(device)

            outputs_withKG = outputs_withKG.detach().cpu().numpy()
            outputs_noKG = outputs_noKG.detach().cpu().numpy()

            batch_y = batch_y.detach().cpu().numpy()

            if valiset.scale:
                outputs_withKG = valiset.inverse_transform(outputs_withKG)
                outputs_noKG = valiset.inverse_transform(outputs_noKG)
                batch_y = valiset.inverse_transform(batch_y)

            pred_withKG = outputs_withKG
            pred_noKG = outputs_noKG
            true = batch_y

            for windfarm in range(10):
                print('predicting windfarm:', windfarm, 'batch:', i, 'batch_size:', args_withKG.batch_size)
                for b in range(args_withKG.batch_size):
                    if windfarm == 9:
                        windfarm_dir = os.path.join(result_dir, f"windfarm_{windfarm}")
                        if not os.path.exists(windfarm_dir):
                            os.makedirs(windfarm_dir)
                        if (i * args_withKG.batch_size + b) % 12 == 0:

                            input = batch_x.detach().cpu().numpy()
                            if valiset.scale:
                                input = valiset.inverse_transform(input)
                            input_total = np.sum(input, axis=2)
                            true_total = np.sum(true, axis=1)
                            pd_withKG_total = np.sum(pred_withKG, axis=1)
                            pd_noKG_total = np.sum(pred_noKG, axis=1)

                            gt = np.concatenate((input_total[b, 5, :], true_total[b, :]), axis=0)
                            pd_withKG = np.concatenate((input_total[b, 5, :], pd_withKG_total[b, :]), axis=0)
                            pd_noKG = np.concatenate((input_total[b, 5, :], pd_noKG_total[b, :]), axis=0)
                            visual_withKG_noKG(gt, pd_withKG, pd_noKG,
                                               os.path.join(windfarm_dir, str(i * args_withKG.batch_size + b) + '.svg'))

                    else:
                        windfarm_dir = os.path.join(result_dir, f"windfarm_{windfarm}")
                        if not os.path.exists(windfarm_dir):
                            os.makedirs(windfarm_dir)
                        if (i*args_withKG.batch_size + b) % 12 == 0:

                            input = batch_x.detach().cpu().numpy()
                            if valiset.scale:
                                input = valiset.inverse_transform(input)
                            gt = np.concatenate((input[b, 5, windfarm, :], true[b, windfarm, :]), axis=0)
                            pd_withKG = np.concatenate((input[b, 5, windfarm, :], pred_withKG[b, windfarm, :]), axis=0)
                            pd_noKG = np.concatenate((input[b, 5, windfarm, :], pred_noKG[b, windfarm, :]), axis=0)
                            visual_withKG_noKG(gt, pd_withKG, pd_noKG, os.path.join(windfarm_dir, str(i*args_withKG.batch_size + b) + '.svg'))







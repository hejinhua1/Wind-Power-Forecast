from data_provider.data_loader import Dataset_Typhoon_KGraph
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mse_loss, mape_loss, mase_loss, smape_loss, WeightedMSELoss, DiffLoss
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single


# 加载数据
dataset = Dataset_Typhoon_KGraph(args, root_path, flag='train')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# 加载模型

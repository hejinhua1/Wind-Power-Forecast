import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np


class Config:
    def __init__(self):
        # 基本配置
        self.task_name = 'long_term_forecast' # 'imputation', 'short_term_forecast', 'long_term_forecast', 'anomaly_detection', 'classification'
        self.is_training = 1
        self.model_id = 'ElcPrice_96_96'
        self.model = 'Informer'   # 'Autoformer', 'Informer', 'Nonstationary_Transformer', 'TimesNet', 'TimeXer'
        self.des = 'Exp'

        # 数据加载
        self.data = 'ElcPrice'
        self.root_path = './data/'
        self.data_path = 'full_data.feather'
        self.features = 'MS'
        self.target = 'dayahead_clearing_price'
        self.freq = 't'
        self.checkpoints = './checkpoints/'

        # 预测任务
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.seasonal_patterns = 'daily'
        self.inverse = True

        # 填充任务
        self.mask_rate = 0.25

        # 异常检测任务
        self.anomaly_ratio = 0.25

        # 模型定义
        self.expand = 2
        self.d_conv = 4
        self.top_k = 5
        self.num_kernels = 6
        self.enc_in = 8
        self.dec_in = 8
        self.c_out = 8
        self.d_model = 16
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 32
        self.moving_avg = 25
        self.factor = 3
        self.distil = True
        self.dropout = 0.1
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.channel_independence = 1
        self.decomp_method = 'moving_avg'
        self.use_norm = 1
        self.down_sampling_layers = 0
        self.down_sampling_window = 1
        self.down_sampling_method = None
        self.seg_len = 48

        # 优化
        self.num_workers = 10
        self.itr = 1
        self.train_epochs = 1
        self.batch_size = 32
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = 'test'
        self.loss = 'MAPE'
        self.lradj = 'type1'
        self.use_amp = False

        # GPU
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'

        # 去平稳投影器参数
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2

        # 度量（dtw）
        self.use_dtw = False

        # 数据增强
        self.augmentation_ratio = 0
        self.seed = 2
        self.jitter = False
        self.scaling = False
        self.permutation = False
        self.randompermutation = False
        self.magwarp = False
        self.timewarp = False
        self.windowslice = False
        self.windowwarp = False
        self.rotation = False
        self.spawner = False
        self.dtwwarp = False
        self.shapedtwwarp = False
        self.wdba = False
        self.discdtw = False
        self.discsdtw = False
        self.extra_tag = ""

        # TimeXer
        self.patch_len = 16

    def __str__(self):
        return f"Config({vars(self)})"


# 示例用法
if __name__ == '__main__':

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = Config()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False


    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


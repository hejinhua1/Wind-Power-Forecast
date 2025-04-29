import sys
sys.path.append("..")
from data_provider.data_loader import Dataset_WindPower, Dataset_STGraph, Dataset_Typhoon, Dataset_KGraph
from torch.utils.data import DataLoader
from models.Informer import Model
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

warnings.filterwarnings('ignore')


def _set_mask():
    # 保留 5, 11, 17, ..., 53 的位置
    keep_indices = np.arange(5, 54, 6)
    # 生成所有索引的布尔数组，初始化为 True
    mask = np.ones(54, dtype=bool)
    # 将 5, 11, 17, ..., 53 的索引位置设置为 False
    mask[keep_indices] = False
    return mask


if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.task_name = 'Informer'
            self.label_len = 48
            self.pred_len = 96
            self.enc_in = 54
            self.dec_in = 54
            self.d_model = 512
            self.embed = 'timeF'
            self.freq = 't'
            self.dropout = 0.1
            self.activation = 'gelu'
            self.factor = 3
            self.n_heads = 8
            self.d_ff = 2048
            self.e_layers = 2
            self.d_layers = 1
            self.distil = True
            self.c_out = 54

            self.train_epochs = 10
            self.learning_rate = 0.0001
            self.patience = 3
            self.batch_size = 32
            self.in_channels = 26
            self.hidden_channels = 96
            self.out_channels = 1
            self.timestep_max = 96
            self.nb_blocks = 2
            self.channels_last = False
            self.show_scores = False

            self.seq_len = 96

            self.num_nodes = 9

    args = Config()

    # 迭代次数和检查点保存间隔
    checkpoint_interval = 1
    datatype = 'normal'
    checkpoint_prefix = 'Informer_'
    log_path = "/home/hjh/WindPowerForecast/logs/Informer_log_"
    result_dir = '/home/hjh/WindPowerForecast/test_results/'
    # 设置检查点路径和文件名前缀
    checkpoint_path = "/home/hjh/WindPowerForecast/checkpoints/"
    # Get the current date and time
    current_datetime = datetime.now()
    # Format the current date and time to display only hours and minutes
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H')

    # 设置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型定义和训练
    model = Model(args).to(device)
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.train_epochs, verbose=True)
    criterion = nn.MSELoss()

    time_now = time.time()
    trainset = Dataset_WindPower(flag='train')
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valiset = Dataset_WindPower(flag='test')
    vali_loader = DataLoader(valiset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 训练循环
    f = open(log_path + formatted_datetime + '_.txt', 'a+')  # 打开文件
    f.write('type:' + datatype + '\n')
    f.close()
    time_now = time.time()
    train_steps = len(train_loader)
    mask = _set_mask()
    for epoch in range(args.train_epochs):
        epoch_time = time.time()
        f = open(log_path + formatted_datetime + '_.txt', 'a+')  # 打开文件
        train_loss = []
        train_l_sum, test_l_sum, iter_count = 0.0, 0.0, 0
        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            opt.zero_grad()
            batch_x[:, :, mask] = batch_y[:, -args.pred_len:, mask]
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            ########################################################
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -args.pred_len:, ~mask]
            batch_y = batch_y[:, -args.pred_len:, ~mask].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            loss.backward()
            opt.step()
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)

        # 测试循环
        preds = []
        trues = []
        result_path = result_dir + checkpoint_prefix + datatype + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x[:, :, mask] = batch_y[:, -args.pred_len:, mask]
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                ########################################################
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -args.pred_len:, ~mask]
                batch_y = batch_y[:, -args.pred_len:, ~mask].to(device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if valiset.scale:
                    outputs = valiset.inverse_transform(outputs)
                    batch_y = valiset.inverse_transform(batch_y)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if valiset.scale:
                        input = valiset.inverse_transform(input)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0) # 取最后一个风电场的风电功率
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0) # 取最后一个风电场的风电功率
                    visual(gt, pd, os.path.join(result_path, str(i) + '.pdf'))
            f.write('Iter: ' + str(epoch) + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        lr_scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, opt.param_groups[0]['lr']))
        if epoch % checkpoint_interval == 0:
            # 保存模型检查点
            checkpoint_name = checkpoint_prefix + datatype + '_epoch_' + str(epoch) + '.pt'
            model_path = os.path.join(checkpoint_path, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': train_loss
            }, model_path)
            print('Checkpoint saved:', model_path)


        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


        mae, mse, rmse, mape, mspe = metric(preds, trues)
        rscore = R2_score(preds, trues)
        print('rscore:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(rscore, mae, rmse, mape, mspe))
        f.write(f'epoch: {epoch}' + "  \n")
        f.write(
            'rscore:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(rscore, mae, rmse, mape, mspe)
        )
        f.write('\n')
        f.write('\n')
        f.close()


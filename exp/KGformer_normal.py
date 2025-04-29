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

warnings.filterwarnings('ignore')




if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.train_epochs = 10
            self.in_channels = 26
            self.hidden_channels = 96
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
            self.batch_size = 32

    args = Config()

    # 迭代次数和检查点保存间隔
    checkpoint_interval = 1
    datatype = 'normal'
    checkpoint_prefix = 'KGformer_'
    log_path = "/home/hjh/WindPowerForecast/logs/KGformer_log_"
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
    trainset = Dataset_KGraph(flag='train')
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valiset = Dataset_KGraph(flag='test')
    vali_loader = DataLoader(valiset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 训练循环
    f = open(log_path + formatted_datetime + '_.txt', 'a+')  # 打开文件
    f.write('type:' + datatype + '\n')
    f.close()
    time_now = time.time()
    train_steps = len(train_loader)
    for epoch in range(args.train_epochs):
        epoch_time = time.time()
        f = open(log_path + formatted_datetime + '_.txt', 'a+')  # 打开文件
        train_loss = []
        train_l_sum, test_l_sum, iter_count = 0.0, 0.0, 0
        model.train()
        for i, (batch_x, batch_y, batch_adj, batch_em_x, batch_em_y) in enumerate(train_loader):
            iter_count += 1
            opt.zero_grad()
            batch_x[:, :-1, :, :] = batch_y[:, :-1, :, -args.pred_len:]
            batch_x = torch.cat([batch_x, batch_em_y[:, :, :, -args.pred_len:]], dim=1)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_adj = batch_adj.float().to(device)
            batch_adj_hat = torch.zeros_like(batch_adj).float().to(device)
            ########################################################
            outputs = model(batch_x, batch_adj, batch_adj_hat)
            batch_y = batch_y[:, -1, :, -args.pred_len:].to(device)
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
            for i, (batch_x, batch_y, batch_adj, batch_em_x, batch_em_y) in enumerate(vali_loader):
                batch_x[:, :-1, :, :] = batch_y[:, :-1, :, -args.pred_len:]
                batch_x = torch.cat([batch_x, batch_em_y[:, :, :, -args.pred_len:]], dim=1)
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_adj = batch_adj.float().to(device)
                batch_adj_hat = torch.zeros_like(batch_adj).float().to(device)
                ########################################################
                outputs = model(batch_x, batch_adj, batch_adj_hat)
                batch_y = batch_y[:, -1, :, -args.pred_len:].to(device)
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
                    gt = np.concatenate((input[0, 5, -1, :], true[0, -1, :]), axis=0) # 取最后一个风电场的风电功率
                    pd = np.concatenate((input[0, 5, -1, :], pred[0, -1, :]), axis=0) # 取最后一个风电场的风电功率
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


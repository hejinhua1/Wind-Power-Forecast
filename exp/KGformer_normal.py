from data_provider.data_loader import Dataset_WindPower, Dataset_STGraph, Dataset_Typhoon, Dataset_KGraph
from models.SpatioTemporalGraph import Model
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mse_loss, mape_loss, mase_loss, smape_loss, WeightedMSELoss, DiffLoss
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
from datetime import datetime
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')



def train(setting):


    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    model_optim = self._select_optimizer()
    criterion = self._select_criterion(self.args.loss)
    mse = nn.MSELoss()


    for epoch in range(self.args.train_epochs):
        iter_count = 0
        train_loss = []

        self.model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_adj, batch_em_x, batch_em_y) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            batch_x[:, :-1, :, :] = batch_y[:, :-1, :, -self.args.pred_len:]
            batch_x = torch.cat([batch_x, batch_em_y[:, :, :, -self.args.pred_len:]], dim=1)
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_adj = batch_adj.float().to(self.device)
            batch_adj_hat = torch.zeros_like(batch_adj).float().to(self.device)


            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_adj, batch_adj_hat)
                    batch_y = batch_y[:, -1, :, -self.args.pred_len:].to(self.device)
                    loss = criterion(0, 0, outputs, batch_y, torch.ones_like(batch_y))
                    train_loss.append(loss.item())
            else:
                outputs = self.model(batch_x, batch_adj, batch_adj_hat)
                batch_y = batch_y[:, -1, :, -self.args.pred_len:].to(self.device)
                loss = criterion(0, 0, outputs, batch_y, torch.ones_like(batch_y))
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if self.args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = self.vali(vali_data, vali_loader, criterion)
        test_loss = self.vali(test_data, test_loader, criterion)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        early_stopping(vali_loss, self.model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    best_model_path = path + '/' + 'checkpoint.pth'
    self.model.load_state_dict(torch.load(best_model_path))

    return model

def test(self, setting, test=0):
    test_data, test_loader = self._get_data(flag='test')
    if test:
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    preds = []
    trues = []
    folder_path = './test_results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    self.model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_adj, batch_em_x, batch_em_y) in enumerate(test_loader):
            batch_x[:, :-1, :, :] = batch_y[:, :-1, :, -self.args.pred_len:]
            batch_x = torch.cat([batch_x, batch_em_y[:, :, :, -self.args.pred_len:]], dim=1)
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_adj = batch_adj.float().to(self.device)
            batch_adj_hat = torch.zeros_like(batch_adj).float().to(self.device)


            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_adj, batch_adj_hat)
            else:
                outputs = self.model(batch_x, batch_adj, batch_adj_hat)

            batch_y = batch_y[:, -1, :, -self.args.pred_len:].to(self.device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            if test_data.scale and self.args.inverse:
                outputs = test_data.inverse_transform(outputs)
                batch_y = test_data.inverse_transform(batch_y)

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    input = test_data.inverse_transform(input)
                gt = np.concatenate((input[0, -1, -1, :], true[0, -1, :]), axis=0)
                pd = np.concatenate((input[0, -1, -1, :], pred[0, -1, :]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, dtw:{}'.format(mse, mae, rmse, mape, mspe,
                                                                                         dtw))
    f = open("result_long_term_forecast.txt", 'a')
    f.write(setting + "  \n")
    f.write(
        'mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, dtw:{}'.format(mse, mae, rmse, mape, mspe,
                                                                                       dtw))
    f.write('\n')
    f.write('\n')
    f.close()

    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)

    return


if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.in_channels = 6
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

    args = Config()

    # 迭代次数和检查点保存间隔
    epoch_size, batch_size = 50, 50
    checkpoint_interval = 1
    checkpoint_prefix = 'KGformer_'
    log_path = "/home/hjh/Tyformer/logs/Pangu_h{}_log"
    # 设置检查点路径和文件名前缀
    checkpoint_path = "/home/hjh/Tyformer/checkpoints/"
    # Get the current date and time
    current_datetime = datetime.now()
    # Format the current date and time to display only hours and minutes
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H')

    # 设置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型定义和训练
    model = Model(args).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-4)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch_size, verbose=True)
    criterion = nn.MSELoss()


    trainset = Dataset_KGraph(args, flag='train', Norm_type='minmax', M=9, N=9)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valiset = WindDataset(flag='vali', Norm_type=Norm_type, M=M, N=N)
    valiloader = DataLoader(valiset, batch_size=batch_size, shuffle=False)

    # 训练循环
    f = open(log_path + formatted_datetime + '.txt', 'a+')  # 打开文件
    f.write('Norm_type:' + Norm_type + '\n')
    f.close()
    for epoch in range(resume_epoch+1, epoch_size + 1):
        f = open(log_path + formatted_datetime + '.txt', 'a+')  # 打开文件
        train_l_sum, test_l_sum, n = 0.0, 0.0, 0
        train_ul_sum, train_vl_sum, test_ul_sum, test_vl_sum = 0.0, 0.0, 0.0, 0.0
        model.train()
        loop = tqdm((trainloader), total=len(trainloader))
        for (x, y) in loop: # x,y torch.Size([B, M, 6, 13, 41, 61])
            ########################################################
            x = x[:, :, :, :, :32, :32] # torch.Size([B, M, 6, 13, 32, 32])
            x = x.permute(0, 1, 3, 4, 5, 2).contiguous() # torch.Size([B, M, 13, 32, 32, 6])
            x = rearrange(x, 'b M v h w c -> b (M v) h w c')
            # print('Input:', get_memory_diff())
            y = y[:, N-24:N, 3:5, -1, :32, :32] # torch.Size([B, N, 2, 32, 32])
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            # print('Output and intermediate:', get_memory_diff())
            ########################################################
            u_loss = criterion(y_hat[:, :, 0, :, :], y[:, :, 0, :, :])
            v_loss = criterion(y_hat[:, :, 1, :, :], y[:, :, 1, :, :])
            loss = weighted_loss(u_loss, v_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # y_raw, y_hat_raw = y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()
            # u_raw, u_hat_raw = normalizer.inverse_target(y_raw[:, :24, :, :], y_hat_raw[:, :24, :, :], target='u', Norm_type=Norm_type)
            # v_raw, v_hat_raw = normalizer.inverse_target(y_raw[:, 24:, :, :], y_hat_raw[:, 24:, :, :], target='v', Norm_type=Norm_type)
            # u_loss = criterion(torch.from_numpy(u_raw), torch.from_numpy(u_hat_raw))
            # v_loss = criterion(torch.from_numpy(v_raw), torch.from_numpy(v_hat_raw))
            # loss = weighted_loss(u_loss, v_loss)
            # loss_num = loss.numpy()
            # loop.set_description(f'Train Epoch: [{epoch}/{epoch_size}] loss: [{loss_num}]')
            # train_l_sum += loss_num*x.shape[0]
            # train_ul_sum += u_loss*x.shape[0]
            # train_vl_sum += v_loss*x.shape[0]
            loss_num = loss.detach().cpu().numpy()
            loop.set_description(f'Train Epoch: [{epoch}/{epoch_size}] loss: [{loss_num}]')
            train_l_sum += loss_num*x.shape[0]
            train_ul_sum += u_loss.detach().cpu().numpy()*x.shape[0]
            train_vl_sum += v_loss.detach().cpu().numpy()*x.shape[0]
            n += x.shape[0]
        train_loss = train_l_sum / n
        train_u_loss = train_ul_sum / n
        train_v_loss = train_vl_sum / n


        n = 0
        model.eval()
        with torch.no_grad():
            loop = tqdm((valiloader), total=len(valiloader))
            for (x, y) in loop:
                ########################################################
                x = x[:, :, :, :, :32, :32]  # torch.Size([B, M, 6, 13, 32, 32])
                x = x.permute(0, 1, 3, 4, 5, 2).contiguous()  # torch.Size([B, M, 13, 32, 32, 6])
                x = rearrange(x, 'b M v h w c -> b (M v) h w c')
                # print('Input:', get_memory_diff())
                y = y[:, N-24:N, 3:5, -1, :32, :32]  # torch.Size([B, N, 2, 32, 32])
                x = x.to(device)
                y_hat = model(x)
                # print('Output and intermediate:', get_memory_diff())
                ########################################################
                y_raw, y_hat_raw = y.numpy(), y_hat.detach().cpu().numpy()
                u_raw, u_hat_raw = normalizer.inverse_target(y_raw[:, :, 0, :, :], y_hat_raw[:, :, 0, :, :], target='u',
                                                             Norm_type=Norm_type)
                v_raw, v_hat_raw = normalizer.inverse_target(y_raw[:, :, 1, :, :], y_hat_raw[:, :, 1, :, :], target='v',
                                                             Norm_type=Norm_type)
                u_loss = criterion(torch.from_numpy(u_raw), torch.from_numpy(u_hat_raw))
                v_loss = criterion(torch.from_numpy(v_raw), torch.from_numpy(v_hat_raw))
                loss = weighted_loss(u_loss, v_loss)
                loss_num = loss.numpy()
                loop.set_description(f'Test Epoch: [{epoch}/{epoch_size}] loss: [{loss_num}]')
                test_l_sum += loss_num * x.shape[0]
                test_ul_sum += u_loss.detach().cpu().numpy() * x.shape[0]
                test_vl_sum += v_loss.detach().cpu().numpy() * x.shape[0]
                n += x.shape[0]

            f.write('Iter: ' + str(epoch) + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
            print('Iter:', epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        test_loss = test_l_sum / n
        test_u_loss = test_ul_sum / n
        test_v_loss = test_vl_sum / n
        lr_scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, opt.param_groups[0]['lr']))
        if epoch % checkpoint_interval == 0:
            # 保存模型检查点
            checkpoint_name = checkpoint_prefix + str(epoch) + '.pt'
            model_path = os.path.join(checkpoint_path, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, model_path)
            print('Checkpoint saved:', model_path)

        f.write('Train loss: ' + str(train_loss) + ' Test loss: ' + str(test_loss) + '\n')
        f.write('Train u loss: ' + str(train_u_loss) + ' Test u loss: ' + str(test_u_loss) + '\n')
        f.write('Train v loss: ' + str(train_v_loss) + ' Test v loss: ' + str(test_v_loss) + '\n')
        print('Train loss:', train_loss, ' Test loss:', test_loss)
        print('===' * 20)
        seg_line = '=======================================================================' + '\n'
        f.write(seg_line)
        f.close()
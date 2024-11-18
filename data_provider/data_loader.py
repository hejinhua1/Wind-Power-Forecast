import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
from joblib import load
warnings.filterwarnings('ignore')



class Dataset_WindPower(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='data.feather',
                 target='power_unit', scale=True, timeenc=0, freq='t', seasonal_patterns=None, id=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.id = id
        id_caps = {0:400, 4:400, 6:300, 7:300, 9:300, 10:300, 14:250, 16:245, 17:500}
        if self.id is None:
            self.id_caps = id_caps
        else:
            self.id_caps = {i: id_caps[i] for i in self.id if i in id_caps}

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        # self.nwp_scaler = load(os.path.join(root_path, "scaler_nwp.joblib"))


    def __read_data__(self):
        # read data
        df_raw = pd.read_feather(os.path.join(self.root_path,
                                              self.data_path))
        # remove the prediction duration column
        df_raw = df_raw.drop(columns=['predict_duration'])
        # rename the TIMESTAMP column
        df_raw = df_raw.rename(columns={'TIMESTAMP': 'date'})
        if df_raw.isna().any().any():
            df_raw = df_raw.fillna(df_raw.mean())

        # scale NWP data and target data if needed
        self.scaler = StandardScaler()
        if self.scale:
            cols_data = df_raw.columns[1:-1] # id column is not needed
            norm_data = df_raw[cols_data]
            self.scaler.fit(norm_data.values)
            df_raw[cols_data] = self.scaler.transform(norm_data.values)

        # columns = ['10','4','5','6','7', #NWP 0-4
        #             'power_unit']      #power 5

        border1s = [0, 635 * 24 * 4 - self.seq_len, 635 * 24 * 4 + 20 * 24 * 4 - self.seq_len]
        border2s = [635 * 24 * 4, 635 * 24 * 4 + 20 * 24 * 4, 787 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # data only
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        # time stamp only
        df_one_station = df_raw[df_raw.id == 0]
        df_stamp = df_one_station[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)


        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        all_data = []

        for station_id in df_data.id.unique():
            if self.id is not None and station_id not in self.id:
                continue

            data_ = df_data[df_data.id == station_id][border1:border2]
            data_np = data_.drop(columns=['id']).values
            all_data.append(data_np)

        # 最后使用 np.concatenate 合并所有数据
        data = np.concatenate(all_data, axis=1)
        self.data_x = data
        self.data_y = data


        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        out = data * self.scaler.scale_[-1] + self.scaler.mean_[-1]
        return out


if __name__ == '__main__':
    dataset = Dataset_WindPower(args=None, root_path='../data/', flag='train', size=None,
                 features='M', data_path='data.feather',
                 target='power_unit', scale=True, timeenc=0, freq='t', seasonal_patterns=None, id=None)
    print(len(dataset))
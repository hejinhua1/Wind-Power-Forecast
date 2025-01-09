import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from datetime import datetime,timedelta
# from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import math
from utils.augmentation import run_augmentation_single
from joblib import load
warnings.filterwarnings('ignore')
from data.train_knowledge_graph import TransEModel



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
        id_caps = {
            16: 245,
            9: 300,
            4: 400,
            14: 250,
            0: 400,
            7: 300,
            10: 300,
            17: 500,
            6: 300
        }
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
        df_raw = df_raw.rename(columns={'TIMESTAMP': 'date', '10': 'wind_speed', '4': 'wind_direction',
                                        '5': 'temperature', '6': 'humidity', '7': 'air_pressure'})
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



class StaticData:
    def __init__(self):
        # 台风时间
        self.Typhoons = {
            'mulan': ['2022-08-09', '2022-08-11'],
            'ma-on': ['2022-08-16', '2022-08-23'],
            'nalgae': ['2022-10-31', '2022-11-03'],
            'talim': ['2023-07-17', '2023-07-19'],
            'saola': ['2023-08-31', '2023-09-03'],
            'koinu': ['2023-10-06', '2023-10-08'],
            'maliksi': ['2024-05-31', '2024-06-01'],
            'yagi': ['2024-09-05', '2024-09-07']
        }
        # 风电场容量/MW
        self.caps = {
            16: 245,
            9: 300,
            4: 400,
            14: 250,
            0: 400,
            7: 300,
            10: 300,
            17: 500,
            6: 300
        }
        # 风电场经纬度
        self.coordinates = {
            16: (116.9930, 23.2692),
            9: (111.6650, 21.3390),
            4: (111.5119, 21.2624),
            14: (114.9953, 22.7061),
            0: (112.2365, 21.4908),
            7: (113.4310, 21.9120),
            10: (110.7611, 20.6272),
            17: (111.6640, 21.0136),
            6: (112.1678, 21.4469)
        }

        # 天气预报数据均值和标准差
        self.mean = {'wind_speed': 7.686432, 'wind_direction': 113.771532, 'temperature': 24.105773,
                     'humidity': 84.693765, 'air_pressure': 1010.592787, 'power_unit': 0.292777}

        self.std = {'wind_speed': 3.779272, 'wind_direction': 75.625513, 'temperature': 4.876235,
                    'humidity': 10.244634, 'air_pressure': 26.378137, 'power_unit': 0.293349}
        self.distance = self.get_distance()
        self.adj_mx = self.get_adjacency_matrix()
        self.edge_index = self.get_edge_index()


    # 计算两个经纬度之间的距离
    def haversine(self, lon1, lat1, lon2, lat2):
        R = 6371  # 地球半径，单位：公里
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    # 计算两两风电场之间的距离
    def get_distance(self):
        distances = np.zeros((len(self.coordinates), len(self.coordinates)))
        i = -1
        for windfarm1 in self.coordinates.keys():
            i = i + 1
            j = -1
            for windfarm2 in self.coordinates.keys():
                j = j + 1
                if windfarm1 != windfarm2:
                    distances[i][j] = self.haversine(self.coordinates[windfarm1][0], self.coordinates[windfarm1][1],
                                                 self.coordinates[windfarm2][0], self.coordinates[windfarm2][1])
        return distances
    # 计算风电场之间的邻接矩阵
    def get_adjacency_matrix(self):
        adj_mx = np.zeros((len(self.coordinates), len(self.coordinates)))
        sigma = np.std(self.distance.reshape(-1))
        for i in range(len(self.coordinates)):
            for j in range(len(self.coordinates)):
                if i != j:
                    adj_mx[i, j] = np.exp(- (self.distance[i, j] / sigma) ** 2)
        return adj_mx

    def get_edge_index(self):
        # 节点数量
        num_nodes = len(self.caps)

        # 生成所有节点对的组合
        edges = torch.combinations(torch.arange(num_nodes), r=2)

        edge_index = edges.t().contiguous()

        return edge_index

class Dataset_Typhoon(Dataset):
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
        self.const = StaticData()
        if flag == 'train':
            self.typhoon = dict(list(self.const.Typhoons.items())[:6])
        else:
            self.typhoon = dict(list(self.const.Typhoons.items())[6:])

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.id = id
        id_caps = self.const.caps
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
        df_raw = df_raw.rename(columns={'TIMESTAMP': 'date', '10': 'wind_speed', '4': 'wind_direction',
                                        '5': 'temperature', '6': 'humidity', '7': 'air_pressure'})
        if df_raw.isna().any().any():
            df_raw = df_raw.fillna(df_raw.mean())

        # scale NWP data and target data if needed
        self.scaler = StandardScaler()
        if self.scale:
            cols_data = df_raw.columns[1:-1] # id column is not needed
            norm_data = df_raw[cols_data]
            self.scaler.fit(norm_data.values)
            df_raw[cols_data] = self.scaler.transform(norm_data.values)

        # 处理时间戳
        df_one_station = df_raw[df_raw.id == 0]
        df_stamp = df_one_station[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

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

        # columns = ['10','4','5','6','7', #NWP 0-4
        #             'power_unit']      #power 5

        self.seq_x = []
        self.seq_y = []
        self.seq_x_mark = []
        self.seq_y_mark = []

        for typhoon_name, typhoon_date in self.typhoon.items():
            start_time = datetime.strptime(typhoon_date[0], "%Y-%m-%d")
            end_time = datetime.strptime(typhoon_date[1], "%Y-%m-%d") + timedelta(days=1)
            start_index = df_one_station[df_one_station.date == start_time].index[0]
            end_index = df_one_station[df_one_station.date == end_time].index[0]

            # 创建空的dataFrame
            df_data = pd.DataFrame()
            for station_id in df_raw.id.unique():
                if self.id is not None and station_id not in self.id:
                    continue
                # 取出当前站点的数据
                df_station = df_raw[df_raw.id == station_id]
                # 按列合并所有站点的数据
                df_data = pd.concat([df_data, df_station], axis=1)
            # 丢掉重复的 date 列
            df_data = df_data.loc[:, ~df_data.columns.str.contains('date') | ~df_data.columns.duplicated()]

            # 丢掉所有的 id 列
            df_data = df_data.drop(['id'], axis=1, errors='ignore')  # 使用 errors='ignore' 确保如果没有 'id' 列时不报错


            # 处理一个台风期间的样本
            for i in range(start_index, end_index - self.pred_len - self.seq_len):
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.pred_len + self.label_len
                seq_x = df_data.iloc[s_begin:s_end].drop(['date'], 1).values
                seq_y = df_data.iloc[r_begin:r_end].drop(['date'], 1).values

                self.seq_x.append(seq_x)
                self.seq_y.append(seq_y)
                self.seq_x_mark.append(data_stamp[s_begin:s_end])
                self.seq_y_mark.append(data_stamp[r_begin:r_end])


        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)


    def __getitem__(self, index):

        seq_x = self.seq_x[index]
        seq_y = self.seq_y[index]
        seq_x_mark = self.seq_x_mark[index]
        seq_y_mark = self.seq_y_mark[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.seq_x)

    def inverse_transform(self, data):
        out = data * self.scaler.scale_[-1] + self.scaler.mean_[-1]
        return out




class Dataset_STGraph(Dataset):
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
        self.const = StaticData()

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.id = id
        id_caps = {
            16: 245,
            9: 300,
            4: 400,
            14: 250,
            0: 400,
            7: 300,
            10: 300,
            17: 500,
            6: 300
        }
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
        df_raw = df_raw.rename(columns={'TIMESTAMP': 'date', '10': 'wind_speed', '4': 'wind_direction',
                                        '5': 'temperature', '6': 'humidity', '7': 'air_pressure'})
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

        data = np.array(all_data) # shape: [num_station, num_time, num_feature]
        self.data_x = data.transpose((2, 0, 1)) # shape: [num_feature, num_station, num_time]
        self.data_y = data.transpose((2, 0, 1)) # shape: [num_feature, num_station, num_time]


        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[:, :, s_begin:s_end]
        seq_y = self.data_y[:, :, r_begin:r_end]
        adj = self.const.adj_mx



        return seq_x, seq_y, adj

    def __len__(self):
        return self.data_x.shape[2] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        out = data * self.scaler.scale_[-1] + self.scaler.mean_[-1]
        return out




class Dataset_KGraph(Dataset):
    '''
    Dataset for KGraph, which is a knowledge-graph-aided model for spatio-temporal forecasting.
    input data: NWP data, target data, and knowledge graph embedding
    '''
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='data_with_entity_id.feather',
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
        self.const = StaticData()

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.id = id
        id_caps = {
            16: 245,
            9: 300,
            4: 400,
            14: 250,
            0: 400,
            7: 300,
            10: 300,
            17: 500,
            6: 300
        }
        if self.id is None:
            self.id_caps = id_caps
        else:
            self.id_caps = {i: id_caps[i] for i in self.id if i in id_caps}

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        # self.nwp_scaler = load(os.path.join(root_path, "scaler_nwp.joblib"))
        # load trained TransE model
        model = TransEModel(num_entities=64, num_relations=5, embedding_dim=10)  # TransE model
        model.load_state_dict(torch.load(os.path.join(self.root_path,
                                              'best_transe_model.pth')))
        self.entity_embeddings = model.entity_embeddings
        self.relation_embeddings = model.relation_embeddings

    def __read_data__(self):
        # read data
        df_raw = pd.read_feather(os.path.join(self.root_path,
                                              self.data_path))

        # remove the prediction duration column
        df_raw = df_raw.drop(columns=['predict_duration'])
        # rename the TIMESTAMP column
        df_raw = df_raw.rename(columns={'TIMESTAMP': 'date', '10': 'wind_speed', '4': 'wind_direction',
                                        '5': 'temperature', '6': 'humidity', '7': 'air_pressure'})
        if df_raw.isna().any().any():
            df_raw = df_raw.fillna(df_raw.mean())

        # scale NWP data and target data if needed
        self.scaler = StandardScaler()
        if self.scale:
            cols_data = df_raw.columns[1:-4] # id column tryples are not needed
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

        data = np.array(all_data) # shape: [num_station, num_time, num_feature]
        self.data_x = data.transpose((2, 0, 1)) # shape: [num_feature, num_station, num_time]
        self.data_y = data.transpose((2, 0, 1)) # shape: [num_feature, num_station, num_time]


        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp




    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[:-3, :, s_begin:s_end]
        seq_y = self.data_y[:-3, :, r_begin:r_end]
        adj = self.const.adj_mx

        embedding_head_id_x = self.data_x[-3, :, s_begin:s_end]
        embedding_head_id_y = self.data_y[-3, :, r_begin:r_end]
        embedding_head_x = self.entity_embeddings(torch.tensor(embedding_head_id_x, dtype=torch.int))
        embedding_head_y = self.entity_embeddings(torch.tensor(embedding_head_id_y, dtype=torch.int))

        embedding_relation_id_x = self.data_x[-2, :, s_begin:s_end]
        embedding_relation_id_y = self.data_y[-2, :, r_begin:r_end]
        embedding_relation_x = self.entity_embeddings(torch.tensor(embedding_relation_id_x, dtype=torch.int))
        embedding_relation_y = self.entity_embeddings(torch.tensor(embedding_relation_id_y, dtype=torch.int))

        embedding_x = torch.cat([embedding_head_x, embedding_relation_x], dim=2)
        embedding_y = torch.cat([embedding_head_y, embedding_relation_y], dim=2)
        embedding_x = embedding_x.permute(2, 0, 1).data.numpy()
        embedding_y = embedding_y.permute(2, 0, 1).data.numpy()
        return seq_x, seq_y, adj, embedding_x, embedding_y

    def __len__(self):
        return self.data_x.shape[2] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        out = data * self.scaler.scale_[-1] + self.scaler.mean_[-1]
        return out


class Dataset_Typhoon_KGraph(Dataset):
    '''
    Dataset for KGraph during typhoon, which is a knowledge-graph-aided model for spatio-temporal forecasting.
    input data: NWP data, target data, and knowledge graph embedding
    '''
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='data_with_entity_id.feather',
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
        self.const = StaticData()
        if flag == 'train':
            self.typhoon = dict(list(self.const.Typhoons.items())[:6])
        else:
            self.typhoon = dict(list(self.const.Typhoons.items())[6:])

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.id = id
        id_caps = {
            16: 245,
            9: 300,
            4: 400,
            14: 250,
            0: 400,
            7: 300,
            10: 300,
            17: 500,
            6: 300
        }
        if self.id is None:
            self.id_caps = id_caps
        else:
            self.id_caps = {i: id_caps[i] for i in self.id if i in id_caps}

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        # self.nwp_scaler = load(os.path.join(root_path, "scaler_nwp.joblib"))
        # load trained TransE model
        model = TransEModel(num_entities=64, num_relations=5, embedding_dim=10)  # TransE model
        model.load_state_dict(torch.load(os.path.join(self.root_path,
                                              'best_transe_model.pth')))
        self.entity_embeddings = model.entity_embeddings
        self.relation_embeddings = model.relation_embeddings

    def __read_data__(self):
        # read data
        df_raw = pd.read_feather(os.path.join(self.root_path,
                                              self.data_path))
        # remove the prediction duration column
        df_raw = df_raw.drop(columns=['predict_duration'])
        # rename the TIMESTAMP column
        df_raw = df_raw.rename(columns={'TIMESTAMP': 'date', '10': 'wind_speed', '4': 'wind_direction',
                                        '5': 'temperature', '6': 'humidity', '7': 'air_pressure'})
        if df_raw.isna().any().any():
            df_raw = df_raw.fillna(df_raw.mean())

        # scale NWP data and target data if needed
        self.scaler = StandardScaler()
        if self.scale:
            cols_data = df_raw.columns[1:-4] # id column and triples are not needed
            norm_data = df_raw[cols_data]
            self.scaler.fit(norm_data.values)
            df_raw[cols_data] = self.scaler.transform(norm_data.values)

        # 处理时间戳
        df_one_station = df_raw[df_raw.id == 0]
        df_stamp = df_one_station[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        # data only
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

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

        # columns = ['10','4','5','6','7', #NWP 0-4
        #             'power_unit']      #power 5

        self.data_x = []
        self.data_y = []


        for typhoon_name, typhoon_date in self.typhoon.items():
            start_time = datetime.strptime(typhoon_date[0], "%Y-%m-%d")
            end_time = datetime.strptime(typhoon_date[1], "%Y-%m-%d") + timedelta(days=1)
            start_index = df_one_station[df_one_station.date == start_time].index[0]
            end_index = df_one_station[df_one_station.date == end_time].index[0]

            all_data = []
            for station_id in df_data.id.unique():
                if self.id is not None and station_id not in self.id:
                    continue

                data_ = df_data[df_data.id == station_id]
                data_np = data_.drop(columns=['id']).values
                all_data.append(data_np)# 一个台风所有场站的数据

            data = np.array(all_data)  # shape: [num_station, num_time, num_feature]

            # 处理一个台风期间的样本
            for i in range(start_index, end_index - self.pred_len - self.seq_len):
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.pred_len + self.label_len
                seq_x = data[:, s_begin:s_end, :]
                seq_y = data[:, r_begin:r_end, :]

                self.data_x.append(seq_x) # shape: [samples, num_station, seq_len, num_feature]
                self.data_y.append(seq_y) # shape: [samples, num_station, pred_len + label_len, num_feature]





    def __getitem__(self, index):
        seq_x_ = self.data_x[index, :, :, :-3] # shape: [num_station, seq_len, num_feature - 3]
        seq_y_ = self.data_y[index, :, :, :-3] # shape: [num_station, pred_len + label_len, num_feature - 3]
        adj = self.const.adj_mx

        embedding_head_id_x = self.data_x[index, :, :, -3]
        embedding_head_id_y = self.data_y[index, :, :, -3]
        embedding_head_x = self.entity_embeddings(torch.tensor(embedding_head_id_x, dtype=torch.int)) # shape: [num_station, seq_len, embed_dim]
        embedding_head_y = self.entity_embeddings(torch.tensor(embedding_head_id_y, dtype=torch.int)) # shape: [num_station, pred_len + label_len, embed_dim]

        embedding_relation_id_x = self.data_x[index, :, :, -2]
        embedding_relation_id_y = self.data_y[index, :, :, -2]
        embedding_relation_x = self.entity_embeddings(torch.tensor(embedding_relation_id_x, dtype=torch.int)) # shape: [num_station, seq_len, embed_dim]
        embedding_relation_y = self.entity_embeddings(torch.tensor(embedding_relation_id_y, dtype=torch.int)) # shape: [num_station, pred_len + label_len, embed_dim]

        embedding_x = torch.cat([embedding_head_x, embedding_relation_x], dim=2)
        embedding_y = torch.cat([embedding_head_y, embedding_relation_y], dim=2)

        seq_x = seq_x_.transpose((2, 0, 1))
        seq_y = seq_y_.transpose((2, 0, 1))
        embedding_x = embedding_x.permute(2, 0, 1).data.numpy()
        embedding_y = embedding_y.permute(2, 0, 1).data.numpy()
        return seq_x, seq_y, adj, embedding_x, embedding_y

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        out = data * self.scaler.scale_[-1] + self.scaler.mean_[-1]
        return out





if __name__ == '__main__':
    # dataset = Dataset_STGraph(args=None, root_path='../data/', flag='train', size=None,
    #              features='M', data_path='data.feather',
    #              target='power_unit', scale=True, timeenc=0, freq='t', seasonal_patterns=None, id=None)
    # dataset = Dataset_Typhoon(args=None, root_path='../data/', flag='train', size=None,
    #              features='M', data_path='data.feather',
    #              target='power_unit', scale=True, timeenc=0, freq='t', seasonal_patterns=None, id=None)
    # dataset = Dataset_WindPower(args=None, root_path='../data/', flag='train', size=None,
    #              features='M', data_path='data.feather',
    #              target='power_unit', scale=True, timeenc=0, freq='t', seasonal_patterns=None, id=None)
    dataset = Dataset_KGraph(args=None, root_path='../data/', flag='train', size=None,
                 features='M', data_path='data_with_entity_id.feather',
                 target='power_unit', scale=True, timeenc=0, freq='t', seasonal_patterns=None, id=None)
    x = dataset[0]
    print(x[0].shape)
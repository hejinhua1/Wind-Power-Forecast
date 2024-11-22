import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
from datetime import datetime,timedelta


Typhoons = {
            'mulan': ['2022-08-09', '2022-08-11'],
            'ma-on': ['2022-08-16', '2022-08-23'],
            'nalgae': ['2022-10-31', '2022-11-03'],
            'talim': ['2023-07-17', '2023-07-19'],
            'saola': ['2023-08-31', '2023-09-03'],
            'koinu': ['2023-10-06', '2023-10-08'],
            'maliksi': ['2024-05-31', '2024-06-01'],
            'yagi': ['2024-09-05', '2024-09-07']
        }



if __name__ == '__main__':
    # load data
    df_data = pd.read_feather('../data/data.feather')
    if df_data.isna().any().any():
        df_data = df_data.fillna(df_data.mean())
    wind = []
    power = []
    for station_id in df_data.id.unique():
        for typhoon_name, typhoon_date in Typhoons.items():
            start_time = datetime.strptime(typhoon_date[0], "%Y-%m-%d")
            end_time = datetime.strptime(typhoon_date[1], "%Y-%m-%d") + timedelta(days=1)
            data_ = df_data[df_data.id == station_id]
            typhoon_data = data_[(data_.TIMESTAMP >= start_time) & (data_.TIMESTAMP <= end_time)]
            data_wind = typhoon_data['10'].values
            data_power = typhoon_data['power_unit'].values
            wind.append(data_wind)
            power.append(data_power)
        winds = np.concatenate(wind)
        powers = np.concatenate(power)
        plt.scatter(winds.reshape(-1,1), powers.reshape(-1,1))
        plt.savefig(f'../data/pic/{station_id}_wind_power.png')
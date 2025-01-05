import pandas as pd
import numpy as np


# STEP 1
# 文件路径
file_path = r'E:\HJHCloud\Seafile\typhoon data\最佳路径数据集完整版\ibtracs.WP.list.v04r01.csv'
typhoon_track_data = pd.read_csv(file_path, low_memory=False)
typhoon_track_data = typhoon_track_data[200000:]
# 筛选 2022 年及以后的数据
data = typhoon_track_data[typhoon_track_data['SEASON'] >= '2022']
typhoon_name = ['mulan', 'ma-on', 'nalgae', 'talim', 'saola', 'koinu', 'maliksi', 'yagi']
# 将上面的名字转为大写
typhoon_name = [name.upper() for name in typhoon_name]

# 读取风电场数据
windfarm_data = pd.read_feather('data.feather')
windfarm_data = windfarm_data[windfarm_data['id'] == 16]
# 取风电场数据的时间列
windfarm_data = windfarm_data[['TIMESTAMP', 'id']]

dataFrame = pd.DataFrame()
for name in typhoon_name:
    typhoon_data = data[data['NAME'] == name]
    # dataFrame 只保留 ISO_TIME, LON, LAT 列
    typhoon_data = typhoon_data[['ISO_TIME', 'LON', 'LAT']]
    # 转为与windfarm_data的TIMESTAMP列相同的时间格式
    typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'])
    # 将ISO_TIME时间转化为东八区时间
    typhoon_data['ISO_TIME'] = typhoon_data['ISO_TIME'] + pd.Timedelta(hours=8)

    # 将 'ISO_TIME' 列转换为 datetime 格式
    df = typhoon_data.copy()
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])

    # 将 'ISO_TIME' 设置为索引
    df.set_index('ISO_TIME', inplace=True)

    # 重新采样为15分钟的时间间隔，并进行线性插值
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')  # 'coerce' 会将无效的转换为 NaN
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df_resampled = df.resample('15T').interpolate(method='linear')

    # 检查df_resampled时间分辨率
    time_resolution = df_resampled.index.to_series().diff().mode()[0]

    print(f"Typhoon: {name}")
    print(f"Time resolution after resampling: {time_resolution}")
    # dataFrame 合并
    dataFrame = pd.concat([dataFrame, df_resampled])


# 将台风数据与风电场数据合并
windfarm_data = pd.merge(windfarm_data, dataFrame, left_on='TIMESTAMP', right_on='ISO_TIME', how='left')
# 去掉id列
windfarm_data = windfarm_data.drop(columns='id')
# 添加一列，值为1，用于标记是否有台风,
windfarm_data['typhoon'] = np.where(windfarm_data['LON'].isnull(), 0, 1)
# 保存数据为feather格式
windfarm_data.to_feather('typhoon_data.feather')


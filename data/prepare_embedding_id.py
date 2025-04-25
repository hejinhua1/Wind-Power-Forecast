import pickle
import pandas as pd
import numpy as np



# STEP 4
# 地球平均半径（单位：km）
EARTH_RADIUS = 6371.0


# 定义 Haversine 公式计算距离
def haversine(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度点之间的球面距离（单位：km）
    """
    # 将经纬度从度转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS * c


# 定义关系映射函数
def map_relationship(distance):
    """
    根据距离映射关系：
    SS: <50 km
    S: 50-100 km
    GS: 100-150 km
    WS: 150-200 km
    N: >200 km
    """
    if distance < 50:
        return 'SS'
    elif 50 <= distance < 100:
        return 'S'
    elif 100 <= distance < 150:
        return 'GS'
    elif 150 <= distance < 200:
        return 'WS'
    else:
        return 'N'


# 加载 entity2id 和 relation2id 映射
with open('entity2id.pkl', 'rb') as f:
    entity2id = pickle.load(f)

with open('relation2id.pkl', 'rb') as f:
    relation2id = pickle.load(f)

# 读取台风数据
typhoon_data = pd.read_feather('typhoon_data.feather')
# 读取NWP数据
nwp_data = pd.read_feather('data.feather')
# 把台风数据和风电场数据合并
data = pd.merge(nwp_data, typhoon_data, left_on='TIMESTAMP', right_on='TIMESTAMP', how='left')

# 定义海上风电场的经纬度（实体2）
wind_farms = {
    'Farm16': (116.9930, 23.2692),
    'Farm9': (111.6650, 21.3390),
    'Farm4': (111.5119, 21.2624),
    'Farm14': (114.9953, 22.7061),
    'Farm0': (112.2365, 21.4908),
    'Farm7': (113.4310, 21.9120),
    'Farm10': (110.7611, 20.6272),
    'Farm17': (111.6640, 21.0136),
    'Farm6': (112.1678, 21.4469)
}

# 计算距离并映射关系
def compute_triples(row):
    lat1, lon1 = row['LAT'], row['LON']
    farmname = 'Farm' + str(row['id'])

    # 如果没有台风或台风不在范围内
    if row['typhoon'] == 0 or not (18 <= lat1 <= 23 and 110 <= lon1 <= 119):
        entity1 = entity2id['OutOfRange']
        relation = relation2id['N']
        entity2 = entity2id[farmname]
    else:
        lon2, lat2 = wind_farms[farmname]
        distance = haversine(lat1, lon1, lat2, lon2)
        relation = relation2id[map_relationship(distance)]
        entity1 = entity2id[str(int(row['LAT'])) + "_" + str(int(row['LON']))]
        entity2 = entity2id[farmname]

    return pd.Series([entity1, relation, entity2])


# 使用apply函数应用compute_triples计算实体和关系
nwp_data[['entity1', 'relation', 'entity2']] = data.apply(compute_triples, axis=1)

# 保存数据
print("Saving data with entity IDs...")
# nwp_data.to_feather('data_with_entity_id.feather')
nwp_data.reset_index(drop=True).to_feather('data_with_entity_id.feather')
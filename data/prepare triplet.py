import pandas as pd
import numpy as np

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


# 读取台风数据
typhoon_data = pd.read_csv('typhoondata_ZhenZhuWan.csv')

# 取整台风经纬度，映射为实体1
typhoon_data['LAT_int'] = typhoon_data['LAT'].astype(int)
typhoon_data['LON_int'] = typhoon_data['LON'].astype(int)
typhoon_data['Entity1'] = typhoon_data['LAT_int'].astype(str) + "_" + typhoon_data['LON_int'].astype(str)

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

# 创建存储三元组的列表
triples = []

# 遍历每个台风记录和每个风电场，计算距离和关系
for _, row in typhoon_data.iterrows():
    lat1, lon1 = row['LAT'], row['LON']
    # 如果 lat1, lon1 不在划定的范围内（假设范围为0-90纬度，0-180经度），跳过
    if not (18 <= lat1 <= 23 and 110 <= lon1 <= 119):
        continue

    entity1 = row['Entity1']

    for farm_name, (lon2, lat2) in wind_farms.items():
        # 计算距离
        distance = haversine(lat1, lon1, lat2, lon2)
        # 映射关系
        relationship = map_relationship(distance)
        # 添加三元组 (实体1, 关系, 实体2)
        triples.append((entity1, relationship, farm_name))

# 增加9对关系，即在划定范围之外，台风对所有风电场的关系为N
out_of_range_entity = 'OutOfRange'
for farm_name in wind_farms.keys():
    triples.append((out_of_range_entity, 'N', farm_name))

# 转换为 DataFrame 便于查看
triples_df = pd.DataFrame(triples, columns=['Entity1', 'Relationship', 'Entity2'])

# 保存三元组到 CSV 文件
triples_df.to_csv('typhoon_windfarm_relationships.csv', index=False)

# 输出部分结果
print(triples_df.head())

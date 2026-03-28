import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import entropy
from scipy.stats import spearmanr
import numpy as np
import math
from dtaidistance import dtw
import os


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度坐标之间的大圆距离（单位：千米）。
    """
    # 地球平均半径，单位为公里
    R = 6371.004

    # 将角度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 差值
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine 公式
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def jensen_shannon_divergence(p, q):
    """
    计算两个概率分布 p 和 q 的 Jensen-Shannon 散度。
    """
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

def jensen_shannon_distance(p, q):
    return math.sqrt(jensen_shannon_divergence(p, q))


def get_data(target):
    folder_path = './dataset/Beijing_12/AQI_processed/'
    file_names = [f'PRSA_Data_{i}.csv' for i in range(1, 13)]
    # 一次性读取所有文件并缓存数据
    data_list = []
    for file_name in file_names:
        data = pd.read_csv(os.path.join(folder_path, file_name))
        if target == 'PM25':
            target_data = data['PM2.5'].values
        elif target == 'PM10':
            target_data = data['PM10'].values
        else:
            raise ValueError(f"Invalid target: {target}")
        data_list.append(target_data)
    return data_list


# 距离图
def calculate_the_distance_matrix(threshold):
    file_path = './dataset/Beijing_12/location/location.csv'
    positions_data = pd.read_csv(file_path)
    stations = positions_data['site_name']
    longitudes = positions_data['Longitude'].values.astype('float32')
    latitudes = positions_data['Latitude'].values.astype('float32')
    # 创建一个 n×n 的距离矩阵
    num_sites = len(stations)
    distance_matrix = np.zeros((num_sites, num_sites))
    # 计算所有站点两两之间的距离
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            distance_matrix[i, j] = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    # 计算标准差和均值
    std_deviation = np.std(distance_matrix.flatten())
    mean_distance = np.mean(distance_matrix.flatten())
    print(f"Distance Matrix -  $\sigma$: 16.00, $\epsilon$: {threshold:.2f}")
    distance_matrix_exp = np.exp(-(distance_matrix ** 2) / (16 ** 2))
    adj_matrix = np.where(distance_matrix_exp > threshold, distance_matrix_exp, 0.0)
    mask = distance_matrix_exp > threshold
    edge_index = np.array(np.where(mask))
    edge_weights = distance_matrix_exp[mask]
    return adj_matrix, edge_index, edge_weights


# 邻居图
def calculate_the_neighbor_matrix(R):
    """
    根据经纬度计算 0-1 邻接矩阵（邻居矩阵），包含自环。
    :param R: 距离阈值（单位：千米）
    :return: 邻接矩阵 A, edge_index 和 edge_weights
    """
    file_path = './dataset/Beijing_12/location/location.csv'
    positions_data = pd.read_csv(file_path)
    longitudes = positions_data['Longitude'].values.astype('float32')
    latitudes = positions_data['Latitude'].values.astype('float32')

    n = len(latitudes)
    # 初始化全 0 矩阵
    A = np.zeros((n, n), dtype='float32')
    edge_index = []
    edge_weights = []

    for i in range(n):
        for j in range(n):
            # 1. 对角线元素：每个站点必然是自己的邻居
            if i == j:
                A[i, j] = 1.0
                edge_index.append([i, j])
                edge_weights.append(1.0)
            # 2. 非对角线元素：计算 Haversine 距离
            else:
                d_ij = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                # 如果距离小于阈值 R，判定为邻居
                if d_ij < R:
                    A[i, j] = 1.0
                    edge_index.append([i, j])
                    edge_weights.append(1.0)
    print(f"Neighbor Matrix -  $\epsilon$: {R:.2f}")
    # 转换为 PyG 所需的 Tensor 格式前，先转为 numpy 数组
    edge_index = np.array(edge_index).T  # 形状变为 (2, num_edges)
    edge_weights = np.array(edge_weights).astype('float32')

    return A, edge_index, edge_weights


# 分布相似图
def calculate_the_similarity_matrix(threshold, target):
    data_list = get_data(target)
    # 初始化结果矩阵
    num_sites = len(data_list)
    js_divergence_matrix = np.zeros((num_sites, num_sites))
    # 遍历所有文件组合，计算 JS 散度
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            js_divergence = jensen_shannon_distance(data_list[i], data_list[j])
            js_divergence_matrix[i, j] = js_divergence
            js_divergence_matrix[j, i] = js_divergence  # 因为 JS 散度是对称的
    # 计算标准差
    std_deviation = np.std(js_divergence_matrix.flatten())
    mean_distance = np.mean(js_divergence_matrix.flatten())
    print(f"Distributional Similarity Matrix - $\sigma$: 0.20, $\epsilon$: {threshold:.2f}")
    js_divergence_matrix_exp = np.exp(-(js_divergence_matrix ** 2) / (0.2 ** 2))
    # 应用阈值过滤
    adj_matrix = np.where(js_divergence_matrix_exp > threshold, js_divergence_matrix_exp, 0.0)
    # 获取边的索引和权重
    mask = js_divergence_matrix_exp > threshold
    edge_index = np.array(np.where(mask))
    edge_weights = js_divergence_matrix_exp[mask]
    return adj_matrix, edge_index, edge_weights
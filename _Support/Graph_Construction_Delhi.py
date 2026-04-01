"""
德里 (Delhi NCT) 多站点空气质量监测网络图构建
适配 DMGENet 框架 — 与北京版本接口一致

注意: 德里站点使用公开文献中的近似坐标;
      功能图 (Functional/POI) 由于缺乏 POI 数据，
      改用 PM2.5 时间序列 Pearson 相关替代。
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

# ─── 德里监测站近似坐标 ──────────────────────────────────────
# 来源: CPCB 站点信息 + OpenStreetMap 近似定位
# 按 download_delhi_data.py 筛选出的 12 站顺序排列
# (实际顺序由数据覆盖率决定, 此处提供已知 Delhi NCT 主要站点坐标)
DELHI_STATION_COORDS = {
    # station_safe_name: (lat, lon)
    'Anand_Vihar_Delhi_DPCC':          (28.6508, 77.3152),
    'Ashok_Vihar_Delhi_DPCC':          (28.6950, 77.1800),
    'Bawana_Delhi_DPCC':               (28.7804, 77.0349),
    'CRRI_Mathura_Road_Delhi_IMD':     (28.5505, 77.2590),
    'DTU_Delhi_CPCB':                  (28.7501, 77.1119),
    'Dwarka_Sector_8_Delhi_DPCC':      (28.5833, 77.0592),
    'IGI_Airport_T3_Delhi_IMD':        (28.5562, 77.1000),
    'ITO_Delhi_CPCB':                  (28.6289, 77.2410),
    'Jahangirpuri_Delhi_DPCC':         (28.7286, 77.1634),
    'Jawaharlal_Nehru_Stadium_Delhi':  (28.5862, 77.2370),
    'Lodhi_Road_Delhi_IMD':            (28.5900, 77.2270),
    'Mandir_Marg_Delhi_DPCC':          (28.6380, 77.2000),
    'Major_Dhyan_Chand_National_Stad': (28.6075, 77.2300),
    'NSIT_Dwarka_Delhi_CPCB':          (28.6082, 77.0307),
    'North_Campus_DU_Delhi_IMD':       (28.6869, 77.2139),
    'Okhla_Phase2_Delhi_DPCC':         (28.5304, 77.2713),
    'Patparganj_Delhi_DPCC':           (28.6283, 77.2891),
    'Punjabi_Bagh_Delhi_DPCC':         (28.6730, 77.1310),
    'Pusa_Delhi_DPCC':                 (28.6431, 77.1499),
    'R_K_Puram_Delhi_DPCC':            (28.5672, 77.1762),
    'Rohini_Delhi_DPCC':               (28.7430, 77.1161),
    'Shadipur_Delhi_CPCB':             (28.6518, 77.1451),
    'Siri_Fort_Delhi_CPCB':            (28.5500, 77.2213),
    'Sri_Aurobindo_Marg_Delhi_DPCC':   (28.5346, 77.2046),
    'Vivek_Vihar_Delhi_DPCC':          (28.6724, 77.3151),
    'Wazirpur_Delhi_DPCC':             (28.7000, 77.1645),
}

# DL 站点代码 → 坐标映射 (与 stations.csv StationName 对应)
DELHI_STATION_COORDS_BY_CODE = {
    'DL001': (28.7238, 77.1210),  # Alipur
    'DL002': (28.6508, 77.3152),  # Anand Vihar
    'DL003': (28.6950, 77.1800),  # Ashok Vihar
    'DL004': (28.5031, 77.0878),  # Aya Nagar
    'DL005': (28.7804, 77.0349),  # Bawana
    'DL006': (28.7271, 77.1984),  # Burari Crossing
    'DL007': (28.5505, 77.2590),  # CRRI Mathura Road
    'DL008': (28.7501, 77.1119),  # DTU
    'DL009': (28.6229, 77.1867),  # Dr. Karni Singh Shooting Range
    'DL010': (28.5833, 77.0592),  # Dwarka-Sector 8
    'DL011': (28.6355, 77.3106),  # East Arjun Nagar
    'DL012': (28.5562, 77.1000),  # IGI Airport (T3)
    'DL013': (28.6781, 77.3047),  # IHBAS, Dilshad Garden
    'DL014': (28.6289, 77.2410),  # ITO
    'DL015': (28.7286, 77.1634),  # Jahangirpuri
    'DL016': (28.5862, 77.2370),  # Jawaharlal Nehru Stadium
    'DL017': (28.5900, 77.2270),  # Lodhi Road
    'DL018': (28.6075, 77.2300),  # Major Dhyan Chand National Stadium
    'DL019': (28.6380, 77.2000),  # Mandir Marg
    'DL020': (28.6692, 76.9827),  # Mundka
    'DL021': (28.6082, 77.0307),  # NSIT Dwarka
    'DL022': (28.6093, 76.9858),  # Najafgarh
    'DL023': (28.8560, 77.0934),  # Narela
    'DL024': (28.5663, 77.2551),  # Nehru Nagar
    'DL025': (28.6869, 77.2139),  # North Campus, DU
    'DL026': (28.5304, 77.2713),  # Okhla Phase-2
    'DL027': (28.6283, 77.2891),  # Patparganj
    'DL028': (28.6730, 77.1310),  # Punjabi Bagh
    'DL029': (28.6431, 77.1499),  # Pusa (DPCC)
    'DL030': (28.6450, 77.1520),  # Pusa (IMD)
    'DL031': (28.5672, 77.1762),  # R K Puram
    'DL032': (28.7430, 77.1161),  # Rohini
    'DL033': (28.6518, 77.1451),  # Shadipur
    'DL034': (28.5500, 77.2213),  # Sirifort
    'DL035': (28.7100, 77.2520),  # Sonia Vihar
    'DL036': (28.5346, 77.2046),  # Sri Aurobindo Marg
    'DL037': (28.6724, 77.3151),  # Vivek Vihar
    'DL038': (28.7000, 77.1645),  # Wazirpur
}

# 通用 fallback 坐标 (Delhi 市中心，用于未知站点)
DEFAULT_LAT, DEFAULT_LON = 28.6139, 77.2090


def _haversine_km(lat1, lon1, lat2, lon2):
    """计算两点球面距离 (km)"""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def _load_station_list():
    """从 station_metadata.json 读取实际选择的站点顺序"""
    meta_path = './dataset/Delhi_NCT/station_metadata.json'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta['stations']
    # Fallback: 按坐标字典顺序
    return list(DELHI_STATION_COORDS.keys())[:12]


def _get_coords(stations):
    """获取站点坐标列表, 未知站点用市中心坐标"""
    coords = []
    for st in stations:
        # 1. DL-code 精确匹配
        if st in DELHI_STATION_COORDS_BY_CODE:
            coords.append(DELHI_STATION_COORDS_BY_CODE[st])
        # 2. 长名称精确匹配
        elif st in DELHI_STATION_COORDS:
            coords.append(DELHI_STATION_COORDS[st])
        else:
            # 3. 模糊匹配长名称 (前20字符)
            matched = None
            for key in DELHI_STATION_COORDS:
                if key[:20] == st[:20] or st[:15] in key:
                    matched = DELHI_STATION_COORDS[key]
                    break
            coords.append(matched if matched else (DEFAULT_LAT, DEFAULT_LON))
    return coords


def calculate_distance_matrix_delhi(threshold=0.5, sigma=10.0):
    """
    距离图 (Distance Graph)
    使用 Haversine 距离 + Gaussian 核
    threshold: 邻接阈值 (保留权重 > threshold 的边)
    sigma: 高斯核带宽 (km), 德里城市范围约 50km × 50km, 用 10km
    """
    stations = _load_station_list()
    N = len(stations)
    coords = _get_coords(stations)

    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dist_matrix[i, j] = _haversine_km(
                    coords[i][0], coords[i][1],
                    coords[j][0], coords[j][1]
                )

    # Gaussian 核权重
    adj = np.exp(-dist_matrix**2 / (2 * sigma**2))
    np.fill_diagonal(adj, 0)

    # 阈值过滤
    adj[adj < threshold] = 0

    # 构建 edge_index 和 edge_weight (PyG 格式)
    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=0)
    edge_weight = adj[rows, cols]

    print(f"Distance Graph (Delhi) - σ={sigma}km, ε={threshold}: {int((adj>0).sum())} edges")
    return adj, edge_index, edge_weight


def calculate_neighbor_matrix_delhi(radius_km=15.0):
    """
    邻居图 (Neighbor Graph)
    radius_km: 邻居半径 (km), 北京用 45km, 德里城市更密集用 15km
    """
    stations = _load_station_list()
    N = len(stations)
    coords = _get_coords(stations)

    adj = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                d = _haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                if d <= radius_km:
                    adj[i, j] = 1.0

    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=0)
    edge_weight = adj[rows, cols]

    print(f"Neighbor Graph (Delhi) - R={radius_km}km: {int(adj.sum())} edges")
    return adj, edge_index, edge_weight


def calculate_similarity_matrix_delhi(threshold=0.1, target='PM25', split='train'):
    """
    分布相似图 (Distribution Similarity Graph)
    使用训练集 PM2.5 时间序列 Pearson 相关系数构建
    (北京版本用 Jensen-Shannon 散度, 德里无 POI 故此处用相关作为近似)
    """
    root = f'./dataset/Delhi_NCT/train_val_test_data/72_6/train_{target}.npz'
    if not os.path.exists(root):
        print(f"相似图: 找不到训练数据 {root}, 使用单位矩阵占位")
        N = len(_load_station_list())
        adj = np.eye(N)
        return adj, np.array([[],[]], dtype=int), np.array([])

    data = np.load(root)
    X = data['X']           # (samples, N, seq_len, features)
    N = X.shape[1]
    F = X.shape[3]

    # PM2.5 是第 0 个特征
    target_idx = 0
    series = X[:, :, :, target_idx]    # (samples, N, seq_len)
    # 转置后展平: 每行对应一个站点的完整时间序列
    series_flat = series.transpose(1, 0, 2).reshape(N, -1)  # (N, samples*seq_len)

    # Pearson 相关矩阵
    corr_matrix = np.corrcoef(series_flat)  # (N, N)
    np.fill_diagonal(corr_matrix, 0)

    # 仅保留正相关且超过阈值的边
    adj = np.where(corr_matrix >= threshold, corr_matrix, 0.0)

    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=0) if len(rows) > 0 else np.array([[],[]], dtype=int)
    edge_weight = adj[rows, cols] if len(rows) > 0 else np.array([])

    print(f"Similarity Graph (Delhi, Pearson) - ε={threshold}: {int(adj.sum()):.0f} edges")
    return adj, edge_index, edge_weight


def calculate_functional_matrix_delhi(threshold=0.3):
    """
    功能相似图 (Functional Graph)
    德里无 POI 数据, 使用跨时间尺度的相关特征构建
    (用 12h 滞后相关作为功能相似度的代理指标)
    """
    root = f'./dataset/Delhi_NCT/train_val_test_data/72_6/train_PM25.npz'
    if not os.path.exists(root):
        N = len(_load_station_list())
        print(f"功能图: 数据不存在, 返回完全图")
        adj = np.ones((N, N)) - np.eye(N)
        rows, cols = np.where(adj > 0)
        return adj, np.stack([rows, cols], axis=0), adj[rows, cols]

    data = np.load(root)
    X = data['X']    # (samples, N, seq_len, F)
    N = X.shape[1]

    # 用第一预测步目标值的相关作为"功能相似度"
    Y_proxy = X[:, :, -1, 0]  # 最后一个时间步的 PM2.5, (samples, N)

    corr = np.corrcoef(Y_proxy.T)  # (N, N)
    np.fill_diagonal(corr, 0)
    adj = np.where(corr >= threshold, corr, 0.0)

    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=0) if len(rows) > 0 else np.array([[],[]], dtype=int)
    edge_weight = adj[rows, cols] if len(rows) > 0 else np.array([])

    print(f"Functional Graph (Delhi, proxy-Pearson) - ε={threshold}: {int(adj.sum()):.0f} edges")
    return adj, edge_index, edge_weight

"""
Delhi 多站点空气质量数据下载与预处理
数据源: CPCB (Central Pollution Control Board, India) via Kaggle
- "Air Quality Data in India (2015-2020)" by Rohan Rao
- 12 个德里 NCT 站点, 小时级别, PM2.5 + 多种污染物

使用方式:
  1. 从 Kaggle 下载 "rohanrao/air-quality-data-in-india" 数据集
     kaggle datasets download -d rohanrao/air-quality-data-in-india
  2. 将 zip 解压到 dataset/Delhi_NCT/raw/
  3. 运行此脚本: python data/download_delhi_data.py
"""
import os
import sys
import glob
import zipfile
import subprocess
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, 'dataset', 'Delhi_NCT')
RAW_DIR = os.path.join(DATASET_DIR, 'raw')
AQI_DIR = os.path.join(DATASET_DIR, 'AQI_processed')

# 德里 NCT 的主要监测站 (与北京 12 站对标)
DELHI_STATIONS = [
    'Anand Vihar, Delhi - DPCC',
    'Ashok Vihar, Delhi - DPCC',
    'Bawana, Delhi - DPCC',
    'CRRI Mathura Road, Delhi - IMD',
    'DTU, Delhi - CPCB',
    'Dwarka-Sector 8, Delhi - DPCC',
    'IGI Airport (T3), Delhi - IMD',
    'ITO, Delhi - CPCB',
    'Jahangirpuri, Delhi - DPCC',
    'Jawaharlal Nehru Stadium, Delhi - DPCC',
    'Lodhi Road, Delhi - IMD',
    'Mandir Marg, Delhi - DPCC',
    'Major Dhyan Chand National Stadium, Delhi - DPCC',
    'NSIT Dwarka, Delhi - CPCB',
    'North Campus, DU, Delhi - IMD',
    'Okhla Phase-2, Delhi - DPCC',
    'Patparganj, Delhi - DPCC',
    'Punjabi Bagh, Delhi - DPCC',
    'Pusa, Delhi - DPCC',
    'R K Puram, Delhi - DPCC',
    'Rohini, Delhi - DPCC',
    'Shadipur, Delhi - CPCB',
    'Siri Fort, Delhi - CPCB',
    'Sri Aurobindo Marg, Delhi - DPCC',
    'Vivek Vihar, Delhi - DPCC',
    'Wazirpur, Delhi - DPCC',
]

# 选择数据完整度较高的 12 个站
TARGET_N_STATIONS = 12

# 特征列配置 (与北京数据对齐: 6 污染物 + 6 气象特征 = 12 维)
# Kaggle 数据集特征: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
# 选择 6 种主要污染物 (对标北京)
POLLUTANTS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
# 该数据集无气象变量, 使用额外的污染物指标补齐到 12 维
EXTRA_FEATURES = ['NO', 'NOx', 'NH3', 'Benzene', 'Toluene', 'Xylene']
ALL_FEATURES = POLLUTANTS + EXTRA_FEATURES  # 12 维

TARGET = 'PM2.5'
TARGET_FILENAME = 'PM25'


def download_kaggle_data():
    """尝试通过 kaggle CLI 下载数据"""
    zip_path = os.path.join(RAW_DIR, 'air-quality-data-in-india.zip')
    if os.path.exists(zip_path):
        print(f"已存在: {zip_path}")
        return zip_path

    os.makedirs(RAW_DIR, exist_ok=True)

    # 尝试 kaggle CLI
    try:
        print("尝试通过 kaggle CLI 下载...")
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', 'rohanrao/air-quality-data-in-india',
            '-p', RAW_DIR
        ], check=True, capture_output=True, text=True)
        print(f"下载完成: {zip_path}")
        return zip_path
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"kaggle CLI 不可用: {e}")
        print("\n请手动下载数据集:")
        print("  1. 访问 https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india")
        print("  2. 下载 zip 文件")
        print(f"  3. 解压 CSV 到 {RAW_DIR}/")
        print("  4. 重新运行此脚本")

        # 检查是否已有 CSV 文件 (手动放置)
        csvs = glob.glob(os.path.join(RAW_DIR, '*.csv'))
        if csvs:
            print(f"\n发现 {len(csvs)} 个已存在的 CSV 文件，继续处理...")
            return None
        sys.exit(1)


def extract_data():
    """解压 zip 文件"""
    zip_path = os.path.join(RAW_DIR, 'air-quality-data-in-india.zip')
    if os.path.exists(zip_path):
        print(f"解压: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(RAW_DIR)


def load_and_select_stations():
    """
    加载数据并选择数据最完整的 12 个德里站点

    实际数据结构 (Kaggle rohanrao/air-quality-data-in-india):
      station_hour.csv  — StationId, Datetime, PM2.5, PM10, NO, NO2, ...
      stations.csv      — StationId, StationName, City, State, Status
    """
    hour_path     = os.path.join(RAW_DIR, 'station_hour.csv')
    stations_path = os.path.join(RAW_DIR, 'stations.csv')

    if not os.path.exists(hour_path):
        # 降级: 尝试旧格式 (单一 city 列的 CSV)
        csv_files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
        print(f"未找到 station_hour.csv, 尝试其他 {len(csv_files)} 个 CSV...")
        all_data = None
        for f in csv_files:
            df = pd.read_csv(f, low_memory=False)
            cols_lower = [c.lower() for c in df.columns]
            if 'city' in cols_lower:
                all_data = df if all_data is None else pd.concat([all_data, df], ignore_index=True)
        if all_data is None:
            print("无法识别数据格式，请检查 dataset/Delhi_NCT/raw/ 目录")
            sys.exit(1)
        all_data.rename(columns={c: 'datetime' for c in all_data.columns if c.lower() == 'date'}, inplace=True)
        all_data.rename(columns={c: 'city'     for c in all_data.columns if c.lower() == 'city'}, inplace=True)
        delhi_data = all_data[all_data['city'].str.contains('Delhi', case=False, na=False)].copy()
        delhi_data.rename(columns={c: 'station' for c in delhi_data.columns
                                   if 'station' in c.lower() or 'site' in c.lower()}, inplace=True)
    else:
        # 主路径: station_hour.csv + stations.csv
        print(f"加载 {hour_path} ...")
        all_data = pd.read_csv(hour_path, low_memory=False)
        print(f"  原始列: {list(all_data.columns)}")
        print(f"  总记录数: {len(all_data):,}")

        # 标准化列名
        all_data.rename(columns={'StationId': 'station', 'Datetime': 'datetime'}, inplace=True)

        if os.path.exists(stations_path):
            meta = pd.read_csv(stations_path)
            meta.rename(columns={'StationId': 'station'}, inplace=True)
            all_data = all_data.merge(meta[['station', 'StationName', 'City']], on='station', how='left')
            delhi_mask = all_data['City'].str.contains('Delhi', case=False, na=False)
        else:
            # 德里站点 ID 均以 DL 开头
            delhi_mask = all_data['station'].str.startswith('DL', na=False)

        delhi_data = all_data[delhi_mask].copy()

    print(f"德里数据: {len(delhi_data):,} 条")
    stations = delhi_data['station'].unique()
    print(f"德里站点数: {len(stations)}")

    # 统计每个站点的 PM2.5 数据覆盖率
    pm25_col = 'PM2.5' if 'PM2.5' in delhi_data.columns else TARGET
    station_coverage = {
        st: delhi_data.loc[delhi_data['station'] == st, pm25_col].notna().sum()
        for st in stations
    }

    sorted_stations = sorted(station_coverage.items(), key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in sorted_stations[:TARGET_N_STATIONS]]

    print(f"\n选择的 {TARGET_N_STATIONS} 个站点 (按 PM2.5 覆盖率排序):")
    for i, (st, cov) in enumerate(sorted_stations[:TARGET_N_STATIONS]):
        name = ''
        if 'StationName' in delhi_data.columns:
            nm = delhi_data.loc[delhi_data['station'] == st, 'StationName'].dropna()
            name = f" ({nm.iloc[0]})" if len(nm) > 0 else ''
        print(f"  {i+1:2d}. {st}{name}  PM2.5 records: {cov:,}")

    return delhi_data, selected


def process_and_align(delhi_data, selected_stations):
    """处理数据: 对齐时间, 插值, 特征工程"""
    # 解析时间
    if 'datetime' in delhi_data.columns:
        delhi_data['datetime'] = pd.to_datetime(delhi_data['datetime'], errors='coerce')
    elif 'Date' in delhi_data.columns:
        delhi_data['datetime'] = pd.to_datetime(delhi_data['Date'], errors='coerce')

    delhi_data = delhi_data.dropna(subset=['datetime'])
    delhi_data = delhi_data.set_index('datetime')

    # 获取所有站点的公共时间范围
    time_ranges = []
    for st in selected_stations:
        st_data = delhi_data[delhi_data['station'] == st]
        if len(st_data) > 0:
            time_ranges.append((st_data.index.min(), st_data.index.max()))

    common_start = max(tr[0] for tr in time_ranges)
    common_end = min(tr[1] for tr in time_ranges)
    print(f"\n公共时间范围: {common_start} ~ {common_end}")

    # 创建小时频率的完整时间索引
    time_index = pd.date_range(start=common_start, end=common_end, freq='h')
    print(f"总时间步: {len(time_index)}")

    all_station_data = {}
    os.makedirs(AQI_DIR, exist_ok=True)

    for st in selected_stations:
        st_data = delhi_data[delhi_data['station'] == st].copy()
        st_data = st_data[~st_data.index.duplicated(keep='first')]

        # 确保特征列存在
        available_features = []
        for feat in ALL_FEATURES:
            if feat in st_data.columns:
                available_features.append(feat)
            else:
                st_data[feat] = np.nan
                available_features.append(feat)

        st_data = st_data[ALL_FEATURES]
        st_data = st_data.reindex(time_index)

        # 数值化
        for col in ALL_FEATURES:
            st_data[col] = pd.to_numeric(st_data[col], errors='coerce')

        # 插值: 线性 + 前向填充 + 后向填充
        st_data = st_data.interpolate(method='linear', limit=24)
        st_data = st_data.ffill(limit=6)
        st_data = st_data.bfill(limit=6)

        # 剩余 NaN 填充为列均值
        for col in ALL_FEATURES:
            if st_data[col].isna().any():
                st_data[col].fillna(st_data[col].mean(), inplace=True)
            if st_data[col].isna().all():
                st_data[col].fillna(0, inplace=True)

        coverage = st_data.notna().mean().mean() * 100
        safe_name = st.replace(',', '').replace(' ', '_').replace('-', '_')[:30]
        all_station_data[safe_name] = st_data.values  # (T, F)

        # 保存单站 CSV
        csv_path = os.path.join(AQI_DIR, f'{safe_name}.csv')
        st_data.to_csv(csv_path)
        print(f"  {safe_name}: shape={st_data.shape}, coverage={coverage:.1f}%")

    return all_station_data, time_index, ALL_FEATURES


def create_sliding_window(all_station_data, feature_cols, seq_len, pred_len, target=TARGET):
    """创建滑动窗口数据集, 与北京数据格式一致"""
    station_names = list(all_station_data.keys())
    N = len(station_names)
    T = len(list(all_station_data.values())[0])
    F = len(feature_cols)

    # 堆叠为 (T, N, F)
    data_array = np.stack([all_station_data[st] for st in station_names], axis=1)  # (T, N, F)
    assert data_array.shape == (T, N, F), f"Shape mismatch: {data_array.shape} vs ({T},{N},{F})"

    # Min-Max 归一化 (全局)
    target_idx = feature_cols.index(target) if target in feature_cols else 0
    target_series = data_array[:, :, target_idx]
    _min = np.nanmin(target_series)
    _max = np.nanmax(target_series)

    # 对所有特征分别 min-max
    data_normed = np.zeros_like(data_array)
    for f_idx in range(F):
        f_min = np.nanmin(data_array[:, :, f_idx])
        f_max = np.nanmax(data_array[:, :, f_idx])
        if f_max - f_min > 1e-8:
            data_normed[:, :, f_idx] = (data_array[:, :, f_idx] - f_min) / (f_max - f_min)
        else:
            data_normed[:, :, f_idx] = 0

    # 滑动窗口
    num_samples = T - seq_len - pred_len + 1
    X = np.zeros((num_samples, N, seq_len, F), dtype=np.float32)
    y = np.zeros((num_samples, N, pred_len), dtype=np.float32)

    for i in range(num_samples):
        X[i] = data_normed[i:i + seq_len].transpose(1, 0, 2)  # (N, seq_len, F)
        y[i] = data_normed[i + seq_len:i + seq_len + pred_len, :, target_idx].T  # (N, pred_len)

    # 6:2:2 split
    n_train = int(num_samples * 0.6)
    n_val = int(num_samples * 0.2)
    splits = {
        'train': (X[:n_train], y[:n_train]),
        'val': (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        'test': (X[n_train + n_val:], y[n_train + n_val:]),
    }

    scaler = np.array([_min, _max])
    print(f"  滑窗: X=({num_samples},{N},{seq_len},{F}), y=({num_samples},{N},{pred_len})")
    for name, (sx, sy) in splits.items():
        print(f"    {name}: X={sx.shape}, y={sy.shape}")

    return splits, scaler


def save_splits(splits, scaler, seq_len, pred_len, target=TARGET_FILENAME):
    out_dir = os.path.join(DATASET_DIR, 'train_val_test_data', f'{seq_len}_{pred_len}')
    os.makedirs(out_dir, exist_ok=True)

    for name, (X, y) in splits.items():
        path = os.path.join(out_dir, f'{name}_{target}.npz')
        np.savez(path, X=X, y=y)
        print(f"  保存: {path}")

    scaler_path = os.path.join(out_dir, f'scaler_{target}.npy')
    np.save(scaler_path, scaler)
    print(f"  保存: {scaler_path} (min={scaler[0]:.4f}, max={scaler[1]:.4f})")


def save_station_metadata(selected_stations, all_station_data):
    """保存站点元数据 (后续图构建需要)"""
    meta = {
        'stations': list(all_station_data.keys()),
        'n_stations': len(all_station_data),
        'features': ALL_FEATURES,
        'original_names': selected_stations[:len(all_station_data)],
    }
    import json
    meta_path = os.path.join(DATASET_DIR, 'station_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"站点元数据: {meta_path}")


def main():
    print("=" * 60)
    print("Delhi NCT 多站点空气质量数据准备")
    print("=" * 60)

    # Step 1: 下载
    print("\n--- Step 1: 下载数据 ---")
    download_kaggle_data()
    extract_data()

    # Step 2: 加载和选站
    print("\n--- Step 2: 加载并选择站点 ---")
    delhi_data, selected_stations = load_and_select_stations()

    # Step 3: 处理和对齐
    print("\n--- Step 3: 数据处理与对齐 ---")
    all_station_data, time_index, feature_cols = process_and_align(delhi_data, selected_stations)

    # Step 4: 保存元数据
    save_station_metadata(selected_stations, all_station_data)

    # Step 5: 创建滑动窗口数据集
    print("\n--- Step 5: 创建滑动窗口数据集 ---")
    for pred_len in [1, 6, 12, 24]:
        print(f"\n  --- seq_len=72, pred_len={pred_len} ---")
        splits, scaler = create_sliding_window(all_station_data, feature_cols, 72, pred_len)
        save_splits(splits, scaler, 72, pred_len)

    print("\n" + "=" * 60)
    print("✅ Delhi 数据准备完成!")
    print(f"  AQI 数据: {AQI_DIR}")
    print(f"  训练数据: {os.path.join(DATASET_DIR, 'train_val_test_data')}")
    print("=" * 60)


if __name__ == '__main__':
    main()

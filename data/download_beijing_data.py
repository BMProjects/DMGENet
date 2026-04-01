"""
下载并预处理 UCI Beijing Multi-Station Air Quality 数据集。

数据来源: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
原始数据包含 12 个监测站 2013.3 - 2017.2 的小时级空气质量和气象数据。

输出:
  1. ./dataset/Beijing_12/AQI_processed/PRSA_Data_{1..12}.csv  (每站 PM2.5 序列，供图构建使用)
  2. ./dataset/Beijing_12/train_val_test_data/{seq}_{pred}/train_PM25.npz  等 (滑窗数据)

使用方法:
  python data/download_beijing_data.py
"""

import os, sys, zipfile, io, shutil
import numpy as np

# --------------- 工具 ---------------
def require(pkg):
    """简易 import，缺包提示安装"""
    try:
        return __import__(pkg)
    except ImportError:
        print(f"缺少依赖: {pkg}，请运行: pip install {pkg}")
        sys.exit(1)

pd = require("pandas")
requests = require("requests")

# --------------- 配置 ---------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(ROOT, "dataset", "Beijing_12", "raw")
AQI_DIR    = os.path.join(ROOT, "dataset", "Beijing_12", "AQI_processed")
DATA_URL   = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"

# 12 个站点名称 (与 location.csv 顺序一致)
STATIONS = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi",
    "Guanyuan",     "Gucheng",   "Huairou",  "Nongzhanguan",
    "Shunyi",       "Tiantan",   "Wanliu",   "Wanshouxigong",
]

# 特征列 (6 种污染物 + 6 种气象变量 = 12 维, 与论文一致)
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEO      = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]  # 5 维气象
# 注意: 原始数据没有显式 wind_direction 数值列，但有 wd (风向字符串)
# 需要将 wd 编码为数值特征以凑齐 12 维 (或者把 RAIN 单独算)

ALL_FEATURES = POLLUTANTS + METEO  # 先用 11 维, 后面加 wind_direction 编码

TARGET          = "PM2.5"   # 原始 CSV 中的列名
TARGET_FILENAME = "PM25"    # 保存文件时使用 (与 exp_base_model.py 约定一致)

# --------------- Step 1: 下载 ---------------
def download_data():
    """下载并解压 UCI 数据"""
    os.makedirs(RAW_DIR, exist_ok=True)

    zip_path = os.path.join(RAW_DIR, "beijing_air_quality.zip")
    if os.path.exists(zip_path):
        print(f"已存在: {zip_path}, 跳过下载")
    else:
        print(f"正在下载: {DATA_URL}")
        resp = requests.get(DATA_URL, timeout=120)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        print(f"下载完成: {zip_path} ({len(resp.content)/1024/1024:.1f} MB)")

    # 解压
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    print(f"解压完成: {RAW_DIR}")

    # 查找 csv 文件 (可能有嵌套 zip)
    inner_zips = [f for f in os.listdir(RAW_DIR) if f.endswith(".zip") and f != os.path.basename(zip_path)]
    for iz in inner_zips:
        iz_path = os.path.join(RAW_DIR, iz)
        with zipfile.ZipFile(iz_path, "r") as zf2:
            zf2.extractall(RAW_DIR)
        print(f"解压内层: {iz}")

    # 列出所有 PRSA 站点 csv (过滤掉 zip 包中的无关文件如 data.csv/test.csv)
    csvs = []
    for dirpath, _, filenames in os.walk(RAW_DIR):
        for fn in filenames:
            if fn.startswith("PRSA_Data_") and fn.endswith(".csv"):
                csvs.append(os.path.join(dirpath, fn))
    print(f"发现 {len(csvs)} 个 CSV 文件")
    for c in sorted(csvs):
        print(f"  {c}")
    return csvs


# --------------- Step 2: 合并 & 清洗 ---------------
def load_and_clean(csv_files):
    """
    读取 12 个站点数据, 对齐时间, 插值缺失值。
    返回:
      all_data: dict, key=站点名, value=DataFrame (columns=特征, index=时间)
      time_index: 公共时间索引
    """
    station_dfs = {}

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        # 提取站点名
        if "station" in df.columns:
            stn = df["station"].iloc[0]
        else:
            # 尝试从文件名提取
            fn = os.path.basename(csv_path)
            stn = fn.replace("PRSA_Data_", "").replace("_20130301-20170228.csv", "").strip()

        # 构建时间索引
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]]
        )
        df = df.set_index("datetime").sort_index()

        # 处理风向 (wd) -> 数值编码
        if "wd" in df.columns:
            wind_dir_map = {
                "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
                "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
                "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
                "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
            }
            df["WD_deg"] = df["wd"].map(wind_dir_map)

        station_dfs[stn] = df

    # 确认找到了 12 个站点
    found = sorted(station_dfs.keys())
    print(f"找到 {len(found)} 个站点: {found}")

    # 统一时间范围
    common_start = max(df.index.min() for df in station_dfs.values())
    common_end = min(df.index.max() for df in station_dfs.values())
    time_idx = pd.date_range(common_start, common_end, freq="h")
    print(f"公共时间范围: {common_start} ~ {common_end}, 共 {len(time_idx)} 小时")

    # 提取和对齐
    feature_cols = POLLUTANTS + METEO + (["WD_deg"] if "WD_deg" in list(station_dfs.values())[0].columns else [])

    all_data = {}
    for stn in STATIONS:
        # 尝试匹配 (原始数据站名可能带空格等)
        matched = None
        for k in station_dfs:
            if k.lower().replace(" ", "").replace("_", "") == stn.lower():
                matched = k
                break
        if matched is None:
            # 模糊匹配
            for k in station_dfs:
                if stn.lower()[:5] in k.lower():
                    matched = k
                    break
        if matched is None:
            print(f"⚠️ 未找到站点: {stn}")
            continue

        df = station_dfs[matched].reindex(time_idx)
        # 线性插值 (论文 Section 5.2: linear interpolation)
        df[feature_cols] = df[feature_cols].interpolate(method="linear", limit_direction="both")
        # 仍有少量 NaN 的列用前后填充
        df[feature_cols] = df[feature_cols].ffill().bfill()
        all_data[stn] = df[feature_cols]

    print(f"成功处理 {len(all_data)} 个站点")
    return all_data, time_idx, feature_cols


# --------------- Step 3: 保存 AQI_processed ---------------
def save_aqi_processed(all_data):
    """保存每站 PM2.5 序列到 AQI_processed/, 供图构建使用"""
    os.makedirs(AQI_DIR, exist_ok=True)
    for i, stn in enumerate(STATIONS):
        if stn not in all_data:
            continue
        df = all_data[stn]
        out_path = os.path.join(AQI_DIR, f"PRSA_Data_{i+1}.csv")
        df.to_csv(out_path)
        print(f"保存: {out_path} (shape={df.shape})")


# --------------- Step 4: 滑窗 + 归一化 + 划分 ---------------
def create_sliding_window_dataset(all_data, feature_cols, seq_len, pred_len, target=TARGET):
    """
    构建多站点滑窗数据集。
    X shape: (samples, num_nodes, seq_len, features)
    y shape: (samples, num_nodes, pred_len)  -- 仅目标变量
    """
    num_nodes = len(STATIONS)
    target_idx = feature_cols.index(target) if target in feature_cols else 0

    # 将所有站点数据堆叠为 (T, N, F)
    arrays = []
    for stn in STATIONS:
        if stn in all_data:
            arrays.append(all_data[stn].values)
        else:
            print(f"⚠️ 站点 {stn} 缺失, 用零填充")
            T = len(list(all_data.values())[0])
            F = len(feature_cols)
            arrays.append(np.zeros((T, F)))

    # stacked: (T, N, F)
    stacked = np.stack(arrays, axis=1).astype(np.float32)
    T, N, F = stacked.shape
    print(f"堆叠后: T={T}, N={N}, F={F}")

    # Min-Max 归一化 (与代码 dataloader 的 inverse_transform 一致)
    # 每个特征在所有站点上全局归一化
    feat_min = stacked.reshape(-1, F).min(axis=0)
    feat_max = stacked.reshape(-1, F).max(axis=0)
    scale = feat_max - feat_min
    scale[scale == 0] = 1.0
    stacked_norm = (stacked - feat_min) / scale

    # 目标变量的 scaler
    target_min = feat_min[target_idx]
    target_max = feat_max[target_idx]

    # 滑窗
    X_list, y_list = [], []
    total_len = T - seq_len - pred_len + 1
    for i in range(total_len):
        x = stacked_norm[i : i + seq_len]  # (seq_len, N, F)
        y = stacked_norm[i + seq_len : i + seq_len + pred_len, :, target_idx]  # (pred_len, N)
        # 转置为 (N, seq_len, F) 和 (N, pred_len)
        X_list.append(x.transpose(1, 0, 2))  # (N, seq_len, F)
        y_list.append(y.T)  # (N, pred_len)

    X = np.array(X_list, dtype=np.float32)  # (samples, N, seq_len, F)
    y = np.array(y_list, dtype=np.float32)  # (samples, N, pred_len)
    print(f"滑窗: X={X.shape}, y={y.shape}")

    # 划分: 6:2:2 (论文 Section 5.1)
    n = len(X)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    splits = {
        "train": (X[:n_train], y[:n_train]),
        "val":   (X[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        "test":  (X[n_train + n_val:], y[n_train + n_val:]),
    }

    for name, (sx, sy) in splits.items():
        print(f"  {name}: X={sx.shape}, y={sy.shape}")

    return splits, np.array([target_min, target_max], dtype=np.float32)


def save_splits(splits, scaler, seq_len, pred_len, target=TARGET_FILENAME):
    """保存为 npz 格式, 与 dataloader_Beijing_12.py 兼容 (文件名使用 PM25)"""
    out_dir = os.path.join(ROOT, "dataset", "Beijing_12", "train_val_test_data", f"{seq_len}_{pred_len}")
    os.makedirs(out_dir, exist_ok=True)

    for name, (X, y) in splits.items():
        path = os.path.join(out_dir, f"{name}_{target}.npz")
        np.savez(path, X=X, y=y)
        print(f"保存: {path}")

    scaler_path = os.path.join(out_dir, f"scaler_{target}.npy")
    np.save(scaler_path, scaler)
    print(f"保存: {scaler_path} (min={scaler[0]:.4f}, max={scaler[1]:.4f})")


# --------------- 主流程 ---------------
def main():
    print("=" * 60)
    print("Step 1: 下载数据")
    print("=" * 60)
    csv_files = download_data()

    print("\n" + "=" * 60)
    print("Step 2: 加载 & 清洗")
    print("=" * 60)
    all_data, time_idx, feature_cols = load_and_clean(csv_files)

    print(f"\n最终特征维度: {len(feature_cols)} -> {feature_cols}")
    print(f"样本总数: {len(time_idx)}")

    print("\n" + "=" * 60)
    print("Step 3: 保存 AQI_processed (供图构建)")
    print("=" * 60)
    save_aqi_processed(all_data)

    print("\n" + "=" * 60)
    print("Step 4: 构建滑窗数据集")
    print("=" * 60)

    seq_len = 72  # 代码实际使用 72 (非论文声称的 24)

    for pred_len in [1, 3, 6, 12, 24]:
        print(f"\n--- seq_len={seq_len}, pred_len={pred_len} ---")
        splits, scaler = create_sliding_window_dataset(
            all_data, feature_cols, seq_len, pred_len
        )
        save_splits(splits, scaler, seq_len, pred_len)

    print("\n✅ 数据准备完成!")
    print(f"AQI 数据: {AQI_DIR}")
    print(f"训练数据: {os.path.join(ROOT, 'dataset', 'Beijing_12', 'train_val_test_data')}")


if __name__ == "__main__":
    main()

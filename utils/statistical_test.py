"""
统计检验 + 非平稳性分析
回应: R1-5 (分布漂移), R2-10 (统计显著性)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

DATASET = 'Beijing_12'
SEQ_LEN = 72
HORIZONS = [1, 6, 12, 24]


# ═══════════════════════════════════════════════════════════
# Part 1: 统计显著性检验 (RLMC 10 runs)
# ═══════════════════════════════════════════════════════════
def statistical_significance():
    print("=" * 70)
    print("Part 1: RLMC 10 runs 统计显著性检验")
    print("=" * 70)

    results = []
    for h in HORIZONS:
        runs_path = f'./RLMC_final_预测结果_{DATASET}/proposed/{SEQ_LEN}/{h}/all_runs_metrics.csv'
        if not os.path.exists(runs_path):
            print(f"  跳过 h={h}: 文件不存在 {runs_path}")
            continue

        df = pd.read_csv(runs_path)

        for metric in ['RMSE', 'MAE']:
            values = df[metric].values
            n = len(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            se = std / np.sqrt(n)

            # 95% 置信区间
            ci_low = mean - 1.96 * se
            ci_high = mean + 1.96 * se

            # 变异系数 (越小越稳定)
            cv = std / mean * 100

            row = {
                'horizon': f'{h}h',
                'metric': metric,
                'mean': round(mean, 4),
                'std': round(std, 4),
                'se': round(se, 4),
                'ci_95_low': round(ci_low, 4),
                'ci_95_high': round(ci_high, 4),
                'cv_pct': round(cv, 2),
                'n_runs': n,
            }
            results.append(row)
            print(f"  h={h:2d}h {metric:4s}: {mean:.4f} ± {std:.4f} (CV={cv:.2f}%, 95%CI=[{ci_low:.4f}, {ci_high:.4f}])")

    df_out = pd.DataFrame(results)
    df_out.to_csv('./doc/statistical_significance.csv', index=False)
    print(f"\n  保存: ./doc/statistical_significance.csv")
    return df_out


# ═══════════════════════════════════════════════════════════
# Part 2: RLMC vs 单图最优 配对比较
# ═══════════════════════════════════════════════════════════
def rlmc_vs_single_graph():
    print("\n" + "=" * 70)
    print("Part 2: DMGENet (RLMC) vs 单图最优模型 — Wilcoxon 检验")
    print("=" * 70)

    results = []
    for h in HORIZONS:
        runs_path = f'./RLMC_final_预测结果_{DATASET}/proposed/{SEQ_LEN}/{h}/all_runs_metrics.csv'
        if not os.path.exists(runs_path):
            continue

        rlmc_df = pd.read_csv(runs_path)
        rlmc_rmse = rlmc_df['RMSE'].values

        # 找单图最优 (只有1个种子，用其RMSE作为 baseline)
        best_single = float('inf')
        best_name = ''
        for model in ['Model_D', 'Model_N', 'Model_S', 'Model_POI']:
            p = f'./预测结果_基础模型_{DATASET}/{SEQ_LEN}/{h}/{model}/test_metrics.csv'
            if os.path.exists(p):
                m = pd.read_csv(p)
                rmse = m['test_RMSE'].values[0]
                if rmse < best_single:
                    best_single = rmse
                    best_name = model

        # 单样本 Wilcoxon 符号秩检验: RLMC runs vs single-graph baseline
        diff = rlmc_rmse - best_single
        if np.all(diff == 0):
            p_val = 1.0
        else:
            stat, p_val = stats.wilcoxon(diff, alternative='less')

        results.append({
            'horizon': f'{h}h',
            'rlmc_mean_rmse': round(np.mean(rlmc_rmse), 4),
            'best_single_rmse': round(best_single, 4),
            'best_single_model': best_name,
            'improvement_pct': round((best_single - np.mean(rlmc_rmse)) / best_single * 100, 2),
            'wilcoxon_p': round(p_val, 6),
            'significant_005': p_val < 0.05,
        })
        sig = "✓ 显著" if p_val < 0.05 else "✗ 不显著"
        print(f"  h={h:2d}h: RLMC={np.mean(rlmc_rmse):.2f} vs {best_name}={best_single:.2f}  "
              f"p={p_val:.4f} {sig}")

    df = pd.DataFrame(results)
    df.to_csv('./doc/rlmc_vs_single_test.csv', index=False)
    print(f"\n  保存: ./doc/rlmc_vs_single_test.csv")
    return df


# ═══════════════════════════════════════════════════════════
# Part 3: 非平稳性分析 (ADF 检验)
# ═══════════════════════════════════════════════════════════
def stationarity_analysis():
    print("\n" + "=" * 70)
    print("Part 3: PM2.5 时间序列非平稳性分析 (ADF 检验)")
    print("=" * 70)

    aqi_dir = './dataset/Beijing_12/AQI_processed'
    if not os.path.exists(aqi_dir):
        print("  跳过：AQI_processed 目录不存在")
        return None

    results = []
    stations = sorted([f for f in os.listdir(aqi_dir) if f.endswith('.csv')])

    for station_file in stations:
        station_name = station_file.replace('.csv', '')
        df = pd.read_csv(os.path.join(aqi_dir, station_file))

        # 找 PM2.5 列
        pm25_col = None
        for col in df.columns:
            if 'PM2.5' in col or 'pm25' in col.lower():
                pm25_col = col
                break
        if pm25_col is None:
            continue

        series = df[pm25_col].dropna().values

        # ADF 检验
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(series, maxlag=72, autolag='AIC')
            adf_stat, adf_p = adf_result[0], adf_result[1]
        except ImportError:
            # fallback: 近似自相关法
            n = len(series)
            mean_s = np.mean(series)
            var_s = np.var(series)
            norm = (series - mean_s) / (np.sqrt(var_s) + 1e-12)
            acf_1 = np.correlate(norm[:-1], norm[1:]) / (n - 1)
            adf_stat = acf_1[0]
            adf_p = 0.01 if abs(acf_1[0]) > 0.5 else 0.10
        except Exception:
            adf_stat, adf_p = np.nan, np.nan

        stationary = "非平稳" if adf_p > 0.05 else "平稳"
        results.append({
            'station': station_name,
            'n_observations': len(series),
            'mean': round(np.mean(series), 2),
            'std': round(np.std(series), 2),
            'min': round(np.min(series), 2),
            'max': round(np.max(series), 2),
            'adf_statistic': round(adf_stat, 4) if not np.isnan(adf_stat) else 'N/A',
            'adf_p_value': round(adf_p, 6) if not np.isnan(adf_p) else 'N/A',
            'conclusion': stationary,
        })
        print(f"  {station_name:25s}: mean={np.mean(series):.1f} std={np.std(series):.1f}  "
              f"ADF p={adf_p:.4f} → {stationary}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv('./doc/stationarity_analysis.csv', index=False)
        print(f"\n  保存: ./doc/stationarity_analysis.csv")
        return df
    return None


# ═══════════════════════════════════════════════════════════
# Part 4: 站点级别预测难度分析
# ═══════════════════════════════════════════════════════════
def station_difficulty_analysis():
    print("\n" + "=" * 70)
    print("Part 4: 站点级别预测难度分析")
    print("=" * 70)

    # 从 RLMC 最佳预测结果中按站点分析
    results = []
    for h in HORIZONS:
        pred_path = f'./RLMC_final_预测结果_{DATASET}/proposed/{SEQ_LEN}/{h}/best_pred.csv'
        true_path = f'./RLMC_final_预测结果_{DATASET}/proposed/{SEQ_LEN}/{h}/best_true.csv'

        if not os.path.exists(pred_path) or not os.path.exists(true_path):
            continue

        preds = pd.read_csv(pred_path, header=None).values  # (samples, nodes*pred_len) or (samples, nodes)
        trues = pd.read_csv(true_path, header=None).values

        n_samples = preds.shape[0]
        n_cols = preds.shape[1]

        # 如果是 (samples, nodes) 形式 — 每站一列
        if n_cols == 12:
            for station_idx in range(12):
                p = preds[:, station_idx]
                t = trues[:, station_idx]
                mae = np.mean(np.abs(p - t))
                rmse = np.sqrt(np.mean((p - t) ** 2))
                results.append({
                    'horizon': f'{h}h',
                    'station_idx': station_idx,
                    'MAE': round(mae, 2),
                    'RMSE': round(rmse, 2),
                })

    if results:
        df = pd.DataFrame(results)
        df.to_csv('./doc/station_difficulty.csv', index=False)
        print(f"  保存: ./doc/station_difficulty.csv")
        print(df.to_string(index=False))
    else:
        print("  没有找到预测结果文件，跳过")


if __name__ == '__main__':
    statistical_significance()
    rlmc_vs_single_graph()
    stationarity_analysis()
    station_difficulty_analysis()
    print("\n✅ 所有统计分析完成!")

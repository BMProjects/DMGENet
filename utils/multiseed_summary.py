"""
多种子基础模型统计汇总
生成 mean ± std 表格，用于论文 Table 2 (统计显著性回应 R2-10)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS       = [42, 123, 456, 789, 2024]
MODELS      = ['Model_D', 'Model_N', 'Model_S', 'Model_POI']
HORIZONS    = [1, 6, 12, 24]
METRICS     = ['test_MAE', 'test_RMSE', 'test_IA']
DATASET     = 'Beijing_12'
SEQ_LEN     = 72
RESULT_ROOT = os.path.join(ROOT, f'预测结果_基础模型_{DATASET}_multiseed')


def collect_results():
    rows = []
    missing = []

    for h in HORIZONS:
        for model in MODELS:
            seed_vals = {m: [] for m in METRICS}
            for seed in SEEDS:
                path = os.path.join(RESULT_ROOT, f'seed_{seed}', str(SEQ_LEN), str(h), model, 'test_metrics.csv')
                if not os.path.exists(path):
                    missing.append(f'seed={seed} h={h} {model}')
                    continue
                df = pd.read_csv(path)
                for m in METRICS:
                    if m in df.columns:
                        seed_vals[m].append(df[m].iloc[0])

            n = len(seed_vals['test_RMSE'])
            if n == 0:
                continue

            row = {'horizon': f'{h}h', 'model': model, 'n_seeds': n}
            for m in METRICS:
                vals = np.array(seed_vals[m])
                short = m.replace('test_', '')
                row[f'{short}_mean'] = round(np.mean(vals), 4)
                row[f'{short}_std']  = round(np.std(vals, ddof=1) if len(vals) > 1 else 0.0, 4)
                row[f'{short}_cv']   = round(np.std(vals, ddof=1) / np.mean(vals) * 100 if len(vals) > 1 else 0.0, 2)
            rows.append(row)

    if missing:
        print(f"⚠️  缺失 {len(missing)} 个结果:")
        for m in missing:
            print(f"   {m}")

    return pd.DataFrame(rows)


def print_latex_table(df):
    """打印 LaTeX 格式的 mean±std 表格 (for 论文)"""
    print("\n" + "=" * 80)
    print("LaTeX Table (mean±std, 5 seeds)")
    print("=" * 80)
    for h in HORIZONS:
        sub = df[df['horizon'] == f'{h}h']
        print(f"\n% Horizon = {h}h")
        for _, row in sub.iterrows():
            mae  = f"{row['MAE_mean']:.2f}±{row['MAE_std']:.2f}"
            rmse = f"{row['RMSE_mean']:.2f}±{row['RMSE_std']:.2f}"
            ia   = f"{row['IA_mean']:.4f}±{row['IA_std']:.4f}"
            print(f"  {row['model']:12s} & {mae:14s} & {rmse:14s} & {ia:18s} \\\\")


def print_summary(df):
    print("\n" + "=" * 80)
    print("多种子基础模型结果汇总 (mean ± std, 5 seeds)")
    print("=" * 80)
    for h in HORIZONS:
        sub = df[df['horizon'] == f'{h}h']
        print(f"\n  Horizon = {h}h")
        print(f"  {'Model':12s}  {'MAE (mean±std)':20s}  {'RMSE (mean±std)':20s}  {'IA (mean±std)':18s}  CV_RMSE%")
        print(f"  {'-'*80}")
        for _, row in sub.iterrows():
            print(
                f"  {row['model']:12s}  "
                f"{row['MAE_mean']:.3f}±{row['MAE_std']:.3f}{'':8s}  "
                f"{row['RMSE_mean']:.3f}±{row['RMSE_std']:.3f}{'':8s}  "
                f"{row['IA_mean']:.4f}±{row['IA_std']:.4f}{'':4s}  "
                f"{row['RMSE_cv']:.2f}%"
            )


if __name__ == '__main__':
    print("读取多种子结果...")
    df = collect_results()

    if df.empty:
        print("❌ 没有找到任何结果，请先运行 run_multi_seed.py")
        sys.exit(1)

    print_summary(df)
    print_latex_table(df)

    out_path = os.path.join(ROOT, 'doc', 'multiseed_results.csv')
    df.to_csv(out_path, index=False)
    print(f"\n💾 保存: {out_path}")
    print(f"✅ 完成 ({len(df)} 个配置)")

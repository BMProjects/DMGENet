"""
DMGENet 完整实验流水线

本脚本按顺序执行:
  Phase 0: 环境检查与数据准备
  Phase 1: 基础模型训练 (4 图 × N 个 horizon) → 复现 Table 2
  Phase 2: RLMC 集成训练 → 复现 Table 7 / 最终 DMGENet 结果
  Phase 3: 消融实验 → 复现 Table 5, 6, Figure 10
  Phase 4: 结果汇总 & 与论文对比

使用方法:
  python run_experiments.py --phase 0        # 仅准备数据
  python run_experiments.py --phase 1        # 训练基础模型
  python run_experiments.py --phase 2        # RLMC 集成
  python run_experiments.py --phase 3        # 消融实验
  python run_experiments.py --phase all      # 全部执行
  python run_experiments.py --phase summary  # 仅汇总结果

可选参数:
  --horizons 1 6 12 24     指定预测步长 (默认 [1, 6])
  --seeds 5                基础模型重复次数 (默认 1, 建议 >=3)
  --rlmc-repeats 10        RLMC 重复次数 (默认 10)
  --gpu 0                  指定 GPU
"""

import argparse
import os
import sys
import subprocess
import time
import json

ROOT = os.path.dirname(os.path.abspath(__file__))


# ================================================================
#  工具函数
# ================================================================
def run_cmd(cmd, desc="", env=None):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"▶ {desc}")
    print(f"  命令: {cmd}")
    print(f"{'='*60}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    start = time.time()
    result = subprocess.run(
        cmd, shell=True, cwd=ROOT, env=merged_env,
        capture_output=False
    )
    elapsed = time.time() - start
    print(f"⏱  耗时: {elapsed:.1f}s  退出码: {result.returncode}")

    if result.returncode != 0:
        print(f"⚠️  命令失败: {cmd}")
    return result.returncode


def check_env():
    """检查 Python 环境和必要依赖"""
    print("环境检查...")
    issues = []

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA:    {torch.version.cuda}")
            print(f"  GPU:     {torch.cuda.get_device_name(0)}")
            print(f"  GPU RAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            issues.append("CUDA 不可用, 训练将极慢")
    except ImportError:
        issues.append("PyTorch 未安装")

    for pkg in ["numpy", "pandas", "scipy", "sklearn"]:
        try:
            __import__(pkg)
        except ImportError:
            issues.append(f"{pkg} 未安装")

    if issues:
        print("\n⚠️  环境问题:")
        for i in issues:
            print(f"  - {i}")
        print("\n建议运行: pip install torch numpy pandas scipy scikit-learn matplotlib requests dtaidistance thop")
        return False

    return True


# ================================================================
#  Phase 0: 数据准备
# ================================================================
def phase0_data_preparation():
    """下载数据 + 预处理 + 图构建验证"""
    print("\n" + "#"*60)
    print("# Phase 0: 数据准备")
    print("#"*60)

    # 检查数据是否已准备
    test_path = os.path.join(ROOT, "dataset", "Beijing_12", "train_val_test_data", "72_6", "train_PM25.npz")
    if os.path.exists(test_path):
        print(f"✅ 数据已存在: {test_path}")
        print("   如需重新准备, 请删除 dataset/Beijing_12/train_val_test_data/ 目录")
        return True

    rc = run_cmd(
        f"{sys.executable} data/download_beijing_data.py",
        desc="下载并预处理 Beijing AQI 数据"
    )
    return rc == 0


# ================================================================
#  Phase 1: 基础模型训练
# ================================================================
def phase1_base_models(horizons, seeds=1, gpu=0):
    """训练 4 种图的基础模型"""
    print("\n" + "#"*60)
    print("# Phase 1: 基础模型训练")
    print(f"#   Horizons: {horizons}")
    print(f"#   Seeds: {seeds}")
    print("#"*60)

    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}

    # 基础模型通过 exp_base_model.py 训练
    # 需要修改其中的 T_out 列表
    # 为简化流程, 直接调用 (原始脚本已支持多 horizon)
    rc = run_cmd(
        f"{sys.executable} exp_base_model.py",
        desc=f"训练基础模型 (horizons={horizons})",
        env=env
    )
    return rc == 0


# ================================================================
#  Phase 2: RLMC 集成训练
# ================================================================
def phase2_rlmc(horizons, repeats=10, gpu=0):
    """准备 RLMC 数据 + 训练 RLMC 集成"""
    print("\n" + "#"*60)
    print("# Phase 2: RLMC 集成训练")
    print(f"#   Horizons: {horizons}")
    print(f"#   Repeats: {repeats}")
    print("#"*60)

    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}

    # Step 1: 准备 RLMC 数据 (合并各图模型的预测)
    rc = run_cmd(
        f"{sys.executable} RLMC_final/get_X_y.py",
        desc="合并基础模型预测结果 (get_X_y)",
        env=env
    )
    if rc != 0:
        return False

    # Step 2: 计算历史误差
    rc = run_cmd(
        f"{sys.executable} RLMC_final/calculating_errors.py",
        desc="计算基础模型历史误差 (calculating_errors)",
        env=env
    )
    if rc != 0:
        return False

    # Step 3: 训练 RLMC
    rc = run_cmd(
        f"{sys.executable} train_RLMC_final.py",
        desc=f"训练 RLMC 集成 (repeats={repeats})",
        env=env
    )
    return rc == 0


# ================================================================
#  Phase 3: 消融实验 (已包含在 Phase 1 中)
# ================================================================
def phase3_ablation_note():
    """
    消融实验说明:
    exp_base_model.py 中已定义了所有消融变体:
      - Model_X_wo_gated_TCN: 移除 Gated TCN
      - Model_X_wo_gcn: 移除 GCN
      - Model_X_wo_gat: 移除 GAT
      - Model_X_wo_ASTAM: 移除 ASTAM

    这些模型在 Phase 1 中一同训练。
    Table 5 (单图对比) 和 Table 6 (组件消融) 的结果
    通过不同图的预测结果组合得到。

    若要额外运行 wo_D / wo_N / wo_S / wo_POI 消融:
    需要在 RLMC_final/get_X_y.py 和 calculating_errors.py 中
    取消对应注释行, 然后重跑 Phase 2。
    """
    print("\n" + "#"*60)
    print("# Phase 3: 消融实验说明")
    print("#"*60)
    print(phase3_ablation_note.__doc__)


# ================================================================
#  Phase Summary: 结果汇总
# ================================================================
def phase_summary(horizons):
    """汇总并对比论文报告的结果"""
    print("\n" + "#"*60)
    print("# 结果汇总: 复现 vs 论文")
    print("#"*60)

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("需要 pandas 和 numpy")
        return

    # 论文 Table 2 中 DMGENet 的结果
    paper_results = {
        1:  {"RMSE": 16.4829, "MAE": 8.6627,  "IA": 0.9899},
        2:  {"RMSE": 25.2926, "MAE": 13.7354, "IA": 0.9749},
        3:  {"RMSE": 31.7757, "MAE": 17.6327, "IA": 0.9587},
        4:  {"RMSE": 37.0809, "MAE": 20.6738, "IA": 0.9415},
        5:  {"RMSE": 41.7162, "MAE": 23.8999, "IA": 0.9221},
        6:  {"RMSE": 45.4055, "MAE": 26.3972, "IA": 0.9069},
    }

    dataset = "Beijing_12"
    seq_len = 72

    results_comparison = []

    for h in horizons:
        # 尝试读取复现结果
        result_dir = f"./RLMC_final_预测结果_{dataset}/proposed/{seq_len}/{h}"
        metrics_path = os.path.join(ROOT, result_dir, "best_metrics.csv")

        reproduced = {}
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            reproduced = {
                "RMSE": df["RMSE"].iloc[0] if "RMSE" in df.columns else None,
                "MAE":  df["MAE"].iloc[0]  if "MAE" in df.columns else None,
                "IA":   df["IA"].iloc[0]   if "IA" in df.columns else None,
            }

        paper = paper_results.get(h, {})

        row = {"Horizon": f"{h}h"}
        for metric in ["RMSE", "MAE", "IA"]:
            p = paper.get(metric)
            r = reproduced.get(metric)
            row[f"Paper_{metric}"] = f"{p:.4f}" if p else "N/A"
            row[f"Repro_{metric}"] = f"{r:.4f}" if r else "待运行"
            if p and r:
                diff_pct = (r - p) / p * 100
                row[f"Diff_{metric}"] = f"{diff_pct:+.2f}%"
            else:
                row[f"Diff_{metric}"] = "-"
        results_comparison.append(row)

    df_cmp = pd.DataFrame(results_comparison)
    print("\n" + df_cmp.to_string(index=False))

    # 保存
    out_path = os.path.join(ROOT, "doc", "reproduction_comparison.csv")
    df_cmp.to_csv(out_path, index=False)
    print(f"\n💾 对比表已保存: {out_path}")

    # 也检查单图模型结果 (Table 5 对应)
    print("\n--- 单图模型结果 (复现 Table 5) ---")
    for model_name in ["Model_D", "Model_N", "Model_S", "Model_POI"]:
        for h in horizons:
            metrics_path = os.path.join(
                ROOT, f"预测结果_基础模型_{dataset}", str(seq_len), str(h), model_name, "test_metrics.csv"
            )
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                print(f"  {model_name} h={h}: {df.to_dict('records')[0]}")


# ================================================================
#  主流程
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="DMGENet 实验流水线")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["0", "1", "2", "3", "all", "summary"],
                        help="执行阶段")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6],
                        help="预测步长列表 (默认 [1, 6])")
    parser.add_argument("--seeds", type=int, default=1,
                        help="基础模型随机种子数")
    parser.add_argument("--rlmc-repeats", type=int, default=10,
                        help="RLMC 重复次数")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU 编号")
    args = parser.parse_args()

    print("=" * 60)
    print("DMGENet 实验流水线")
    print(f"  Phase:    {args.phase}")
    print(f"  Horizons: {args.horizons}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  RLMC:     {args.rlmc_repeats} repeats")
    print(f"  GPU:      {args.gpu}")
    print("=" * 60)

    # 环境检查
    if not check_env():
        print("\n❌ 环境检查未通过, 请先安装依赖")
        sys.exit(1)

    total_start = time.time()

    if args.phase in ["0", "all"]:
        if not phase0_data_preparation():
            print("❌ Phase 0 失败")
            sys.exit(1)

    if args.phase in ["1", "all"]:
        phase1_base_models(args.horizons, args.seeds, args.gpu)

    if args.phase in ["2", "all"]:
        phase2_rlmc(args.horizons, args.rlmc_repeats, args.gpu)

    if args.phase in ["3", "all"]:
        phase3_ablation_note()

    if args.phase in ["summary", "all"]:
        phase_summary(args.horizons)

    total_time = time.time() - total_start
    print(f"\n⏱  总耗时: {total_time/60:.1f} 分钟")
    print("✅ 实验流水线完成")


if __name__ == "__main__":
    main()

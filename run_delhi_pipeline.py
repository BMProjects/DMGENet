"""
DMGENet — Delhi NCT 完整实验流水线
=====================================
跨地区泛化性验证: 印度德里 (新兴工业化国家)

Pipeline:
  Phase 0: 数据准备 (data/download_delhi_data.py)
  Phase 1: 单图基础模型训练 (exp_base_model_delhi.py)
  Phase 2: RLMC 数据整理 (get_X_y + calculating_errors, Delhi 路径)
  Phase 3: RLMC 集成训练 (train_RLMC_final.Exp, Delhi 路径)
  Phase 4: 结果汇总 + 与北京结果对比

使用方式:
  python run_delhi_pipeline.py             # 全部流程
  python run_delhi_pipeline.py --phase 0   # 仅准备数据
  python run_delhi_pipeline.py --phase 1   # 仅训练基础模型
  python run_delhi_pipeline.py --phase 2   # 仅整理 RLMC 数据
  python run_delhi_pipeline.py --phase 3   # 仅训练 RLMC
  python run_delhi_pipeline.py --phase 4   # 仅汇总结果
"""

import os, sys, shutil, random, argparse, time
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DATASET     = 'Delhi_NCT'
SEQ_LEN     = 72
HORIZONS    = [1, 6, 12, 24]
TARGET      = 'PM25'
NUM_NODES   = 12

# model hyper-params (identical to Beijing for fair comparison)
IN_CHANNELS  = 12
HIDDEN_SIZE  = 64
DROPOUT      = 0.2
BLOCK_NUM    = 2
NUM_HEADS    = 4
ALPHA        = 0.2
APT_SIZE     = 10
NUM_CHANNELS = [64, 64, 64, 64]
EPOCH        = 100
LR           = 0.001
BATCH_SIZE   = 64
NUM_WORKERS  = 3

# RLMC hyper-params (identical to Beijing)
RL_STATE_DIM    = 12
RL_HIDDEN_DIM   = 64
RL_GAMMA        = 0.99
RL_TAU          = 0.005
RL_LR_ACTOR     = 1e-4
RL_LR_CRITIC    = 1e-3
RL_EPISODES     = 200
RL_MAX_STEPS    = 1000
RL_BATCH_SIZE   = 64
RL_BUFFER_SIZE  = 2000
RL_REPEATS      = 10


# ══════════════════════════════════════════════════════════════════
# Phase 0 — Data Preparation
# ══════════════════════════════════════════════════════════════════

def phase0_data():
    print("\n" + "#"*60)
    print("# Phase 0: Delhi 数据准备")
    print("#"*60)

    test_path = os.path.join(
        ROOT, 'dataset', 'Delhi_NCT', 'train_val_test_data',
        f'{SEQ_LEN}_6', f'train_{TARGET}.npz'
    )
    if os.path.exists(test_path):
        print(f"✅ 数据已存在: {test_path}")
        return True

    import subprocess
    rc = subprocess.run(
        [sys.executable, 'data/download_delhi_data.py'],
        cwd=ROOT
    ).returncode
    if rc != 0:
        print("❌ 数据准备失败 — 请确认 Kaggle 凭据或手动解压数据")
        return False
    print("✅ Delhi 数据准备完成")
    return True


# ══════════════════════════════════════════════════════════════════
# Phase 1 — Single-graph Base Models
# ══════════════════════════════════════════════════════════════════

def phase1_base_models():
    print("\n" + "#"*60)
    print("# Phase 1: 单图基础模型训练 (Delhi NCT)")
    print("#"*60)

    from model.model_1 import Model
    from exp_base_model_delhi import Exp_model_Delhi, _get_delhi_graphs
    from utils.tools import setup_seed

    setup_seed(2026)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    graphs = _get_delhi_graphs()

    graph_names = ['Model_D', 'Model_N', 'Model_S', 'Model_POI']

    for T_out in HORIZONS:
        print(f"\n{'='*50}  horizon={T_out}h  {'='*50}")
        for model_name in graph_names:
            save_dir = os.path.join(
                ROOT, f'预测结果_基础模型_{DATASET}', str(SEQ_LEN), str(T_out), model_name
            )
            if os.path.exists(os.path.join(save_dir, 'test_metrics.csv')):
                print(f"  [跳过] {model_name} h={T_out} (已完成)")
                continue

            adj = graphs[model_name]
            model = Model(
                adj, IN_CHANNELS, HIDDEN_SIZE, DROPOUT, ALPHA, NUM_HEADS,
                num_channels=NUM_CHANNELS, apt_size=APT_SIZE,
                num_nodes=NUM_NODES, num_block=BLOCK_NUM,
                T_in=SEQ_LEN, predict_len=T_out,
                gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=True,
            )
            exp = Exp_model_Delhi(
                model_name, model, EPOCH, LR, TARGET,
                BATCH_SIZE, NUM_WORKERS, SEQ_LEN, T_out,
            )
            print(f"\n  ▶ {model_name}  h={T_out}  训练开始")
            exp.train()
            exp.test()
            print(f"  ✅ {model_name}  h={T_out}  完成")
            torch.cuda.empty_cache()

    print("\n✅ Phase 1 完成")
    return True


# ══════════════════════════════════════════════════════════════════
# Phase 2 — RLMC Data Assembly
# ══════════════════════════════════════════════════════════════════

def _calculate_smape(y_true, y_pred):
    eps = 1e-8
    return 100 * np.mean(
        np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + eps),
        axis=(1, 2)
    )

def _calculate_mape(y_true, y_pred):
    eps = 1e-8
    return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + eps)), axis=(1, 2))

def _calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=(1, 2))

def _shift(arr):
    shifted = np.roll(arr, shift=1)
    shifted[0] = arr[-1]
    return shifted


def phase2_rlmc_data():
    print("\n" + "#"*60)
    print("# Phase 2: 整理 RLMC 训练数据 (Delhi NCT)")
    print("#"*60)

    BASE_PRED_ROOT = os.path.join(ROOT, f'预测结果_基础模型_{DATASET}')
    RLMC_DATA_ROOT = os.path.join(ROOT, f'RLMC_final_数据集_{DATASET}')
    MODELS         = ['Model_D', 'Model_N', 'Model_S', 'Model_POI']
    GROUP          = 'proposed'

    for T_out in HORIZONS:
        dst = os.path.join(RLMC_DATA_ROOT, GROUP, str(SEQ_LEN), str(T_out))
        os.makedirs(dst, exist_ok=True)

        # ─ copy shared X / y arrays from Model_D ─
        base_d = os.path.join(BASE_PRED_ROOT, str(SEQ_LEN), str(T_out), 'Model_D')
        for split in ('val', 'test'):
            for suffix in ('_X.npy', '_y.npy', '_y_inverse.npy'):
                src = os.path.join(base_d, f'{split}{suffix}')
                if os.path.exists(src):
                    shutil.copy(src, dst)

        # ─ stack predictions from all 4 models ─
        for split in ('val', 'test'):
            preds, preds_inv = [], []
            for m in MODELS:
                m_dir = os.path.join(BASE_PRED_ROOT, str(SEQ_LEN), str(T_out), m)
                p  = os.path.join(m_dir, f'{split}_predictions.npy')
                pi = os.path.join(m_dir, f'{split}_predictions_inverse.npy')
                if os.path.exists(p) and os.path.exists(pi):
                    preds.append(np.load(p))
                    preds_inv.append(np.load(pi))
                else:
                    print(f"  ⚠️  缺少文件: {p}")

            if preds:
                np.save(os.path.join(dst, f'{split}_predictions_all.npy'),
                        np.stack(preds, axis=1))
                np.save(os.path.join(dst, f'{split}_predictions_inverse_all.npy'),
                        np.stack(preds_inv, axis=1))

        # ─ compute history errors ─
        for error in ('mae', 'mape', 'smape'):
            fn_map = {'mae': _calculate_mae, 'mape': _calculate_mape, 'smape': _calculate_smape}
            fn = fn_map[error]

            for split in ('val', 'test'):
                dfs = []
                for m in MODELS:
                    m_dir = os.path.join(BASE_PRED_ROOT, str(SEQ_LEN), str(T_out), m)
                    y_f    = os.path.join(m_dir, f'{split}_y_inverse.npy')
                    pred_f = os.path.join(m_dir, f'{split}_predictions_inverse.npy')
                    if os.path.exists(y_f) and os.path.exists(pred_f):
                        errs = fn(np.load(y_f), np.load(pred_f))
                        dfs.append(pd.DataFrame(_shift(errs), columns=[m]))

                if dfs:
                    combined = pd.concat(dfs, axis=1)
                    combined.to_csv(
                        os.path.join(dst, f'combined_{split}_{error}_history_errors.csv'),
                        index=False
                    )

        print(f"  ✅ h={T_out}  RLMC 数据已整理 → {dst}")

    print("\n✅ Phase 2 完成")
    return True


# ══════════════════════════════════════════════════════════════════
# Phase 3 — RLMC Ensemble Training
# ══════════════════════════════════════════════════════════════════

def phase3_rlmc_train():
    print("\n" + "#"*60)
    print("# Phase 3: RLMC 集成训练 (Delhi NCT)")
    print("#"*60)

    from train_RLMC_final import Exp as RLExp

    RLMC_DATA_ROOT   = os.path.join(ROOT, f'RLMC_final_数据集_{DATASET}')
    RLMC_RESULT_ROOT = os.path.join(ROOT, f'RLMC_final_预测结果_{DATASET}')
    GROUP            = 'proposed'
    ACTION_DIM       = 4  # 4 graph models

    all_summary = []

    for T_out in HORIZONS:
        base_dir     = os.path.join(RLMC_DATA_ROOT,   GROUP, str(SEQ_LEN), str(T_out))
        results_dir  = os.path.join(RLMC_RESULT_ROOT, GROUP, str(SEQ_LEN), str(T_out))
        os.makedirs(results_dir, exist_ok=True)

        # skip if already done
        if os.path.exists(os.path.join(results_dir, 'all_runs_metrics.csv')):
            print(f"  [跳过] h={T_out} (已完成)")
            df = pd.read_csv(os.path.join(results_dir, 'all_runs_metrics.csv'))
            m = df.mean(numeric_only=True)
            all_summary.append({'horizon': T_out, 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'IA': m['IA']})
            continue

        print(f"\n  ▶ RLMC  h={T_out}")

        # load data
        train_X              = np.load(os.path.join(base_dir, 'val_X.npy'))
        train_hist_errors    = pd.read_csv(
            os.path.join(base_dir, 'combined_val_mae_history_errors.csv')
        ).values.astype('float32')
        train_y              = np.load(os.path.join(base_dir, 'val_y.npy'))
        train_preds          = np.load(os.path.join(base_dir, 'val_predictions_all.npy'))

        test_X               = np.load(os.path.join(base_dir, 'test_X.npy'))
        test_hist_errors     = pd.read_csv(
            os.path.join(base_dir, 'combined_test_mae_history_errors.csv')
        ).values.astype('float32')
        test_y               = np.load(os.path.join(base_dir, 'test_y.npy'))
        test_preds           = np.load(os.path.join(base_dir, 'test_predictions_all.npy'))
        test_y_inv           = np.load(os.path.join(base_dir, 'test_y_inverse.npy'))
        test_preds_inv       = np.load(os.path.join(base_dir, 'test_predictions_inverse_all.npy'))

        best_val   = float('inf')
        best_run   = -1
        run_results = []

        for run in range(RL_REPEATS):
            print(f"\n    🔥 Run {run+1}/{RL_REPEATS}")
            random.seed(run); np.random.seed(run); torch.manual_seed(run)

            exp = RLExp(
                RL_STATE_DIM, ACTION_DIM, RL_GAMMA, RL_LR_ACTOR, RL_LR_CRITIC,
                RL_TAU, RL_HIDDEN_DIM, RL_EPISODES, RL_MAX_STEPS,
                RL_BATCH_SIZE, RL_BUFFER_SIZE,
                train_X, train_hist_errors, train_y, train_preds,
                test_X,  test_hist_errors,  test_y, test_preds,
                test_y_inv, test_preds_inv,
                results_dir,
            )
            exp.train()

            val_loss = exp.compute_test_loss(
                train_X, train_hist_errors, train_y, train_preds, batch_size=256
            )
            metrics, pred, true, weights = exp.test(run_id=run)
            mae, rmse, ia, r2 = metrics

            run_results.append({'run': run+1, 'MAE': mae, 'RMSE': rmse,
                                 'IA': ia, 'R2': r2, 'val_loss': val_loss})
            print(f"    Run {run+1}: MAE={mae:.3f} RMSE={rmse:.3f} IA={ia:.4f} val={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                best_run = run
                pd.DataFrame([metrics], columns=['MAE','RMSE','IA','R2']).to_csv(
                    os.path.join(results_dir, 'best_metrics.csv'), index=False)
                torch.save(exp.agent.actor.state_dict(),
                           os.path.join(results_dir, 'best_actor.pth'))
                np.save(os.path.join(results_dir, 'best_pred.npy'), pred)
                np.save(os.path.join(results_dir, 'best_true.npy'), true)
                pd.DataFrame(pred.reshape(pred.shape[0], -1)).to_csv(
                    os.path.join(results_dir, 'best_pred.csv'), index=False)
                pd.DataFrame(true.reshape(true.shape[0], -1)).to_csv(
                    os.path.join(results_dir, 'best_true.csv'), index=False)
                print(f"    🏆 新最优 Run={run+1}")

            torch.cuda.empty_cache()

        df = pd.DataFrame(run_results)
        df.to_csv(os.path.join(results_dir, 'all_runs_metrics.csv'), index=False)
        m, s = df.mean(numeric_only=True), df.std(numeric_only=True)
        print(f"\n  h={T_out}  MAE={m['MAE']:.3f}±{s['MAE']:.3f}  "
              f"RMSE={m['RMSE']:.3f}±{s['RMSE']:.3f}  IA={m['IA']:.4f}±{s['IA']:.4f}")
        all_summary.append({'horizon': T_out, 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'IA': m['IA']})

    print("\n✅ Phase 3 完成")
    return True


# ══════════════════════════════════════════════════════════════════
# Phase 4 — Summary & Beijing Comparison
# ══════════════════════════════════════════════════════════════════

def phase4_summary():
    print("\n" + "#"*60)
    print("# Phase 4: 结果汇总 — Delhi vs Beijing")
    print("#"*60)

    rows = []
    for T_out in HORIZONS:
        # Delhi RLMC
        delhi_path = os.path.join(
            ROOT, f'RLMC_final_预测结果_{DATASET}',
            'proposed', str(SEQ_LEN), str(T_out), 'all_runs_metrics.csv'
        )
        if os.path.exists(delhi_path):
            df = pd.read_csv(delhi_path)
            m = df.mean(numeric_only=True)
            rows.append({
                'region': 'Delhi_NCT', 'horizon': T_out,
                'MAE': round(m['MAE'], 3), 'RMSE': round(m['RMSE'], 3),
                'IA':  round(m['IA'], 4),  'R2':   round(m['R2'],  4),
            })

        # Beijing RLMC (existing results)
        bj_path = os.path.join(
            ROOT, 'RLMC_final_预测结果_Beijing_12',
            'proposed', str(SEQ_LEN), str(T_out), 'all_runs_metrics.csv'
        )
        if os.path.exists(bj_path):
            df = pd.read_csv(bj_path)
            m = df.mean(numeric_only=True)
            rows.append({
                'region': 'Beijing_12', 'horizon': T_out,
                'MAE': round(m['MAE'], 3), 'RMSE': round(m['RMSE'], 3),
                'IA':  round(m['IA'], 4),  'R2':   round(m['R2'],  4),
            })

    if not rows:
        print("  ⚠️  暂无结果，请先完成 Phase 1-3")
        return False

    df_all = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, 'doc'), exist_ok=True)
    out_path = os.path.join(ROOT, 'doc', 'cross_region_comparison.csv')
    df_all.to_csv(out_path, index=False)

    print("\n跨地区性能对比 (DMGENet):")
    print(df_all.to_string(index=False))
    print(f"\n保存: {out_path}")

    # Print single-graph base model results too
    print("\n\n单图基础模型结果 (Delhi NCT):")
    sg_rows = []
    for T_out in HORIZONS:
        for m_name in ['Model_D', 'Model_N', 'Model_S', 'Model_POI']:
            p = os.path.join(ROOT, f'预测结果_基础模型_{DATASET}',
                             str(SEQ_LEN), str(T_out), m_name, 'test_metrics.csv')
            if os.path.exists(p):
                row = pd.read_csv(p).iloc[0].to_dict()
                sg_rows.append({'horizon': T_out, 'model': m_name,
                                'MAE': row.get('test_MAE'), 'RMSE': row.get('test_RMSE'),
                                'IA':  row.get('test_IA')})
    if sg_rows:
        print(pd.DataFrame(sg_rows).to_string(index=False))

    print("\n✅ Phase 4 完成")
    return True


# ══════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all',
                        choices=['0','1','2','3','4','all'])
    args = parser.parse_args()

    t0 = time.time()
    ok = True

    if args.phase in ('0', 'all'):
        ok = phase0_data()
        if not ok and args.phase == 'all':
            print("Phase 0 失败，退出")
            sys.exit(1)

    if args.phase in ('1', 'all'):
        ok = phase1_base_models()

    if args.phase in ('2', 'all'):
        ok = phase2_rlmc_data()

    if args.phase in ('3', 'all'):
        ok = phase3_rlmc_train()

    if args.phase in ('4', 'all'):
        phase4_summary()

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed/3600:.1f}h ({elapsed:.0f}s)")


if __name__ == '__main__':
    main()

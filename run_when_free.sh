#!/usr/bin/env bash
# run_when_free.sh — 监控 GPU，空闲后自动运行 DMGENet 实验复现流程
#
# 用法:
#   ./run_when_free.sh [GPU_IDLE_THRESHOLD_%] [IDLE_CONFIRM_COUNT]
#   默认: GPU 利用率 < 15% 连续 3 次 (每次间隔 60s) 才视为空闲
#
# 后台运行:
#   nohup ./run_when_free.sh > logs/monitor.log 2>&1 &
#   echo $! > logs/monitor.pid
#
# 查看进度:
#   tail -f logs/monitor.log
#   tail -f logs/experiment.log
#   cat  logs/monitor.pid | xargs ps -p

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
EXP_LOG="$LOG_DIR/experiment.log"
MONITOR_LOG="$LOG_DIR/monitor.log"   # this script's stdout when run via nohup
PID_FILE="$LOG_DIR/experiment.pid"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

GPU_IDLE_THRESHOLD="${1:-15}"      # GPU 利用率低于此值视为空闲
IDLE_CONFIRM_COUNT="${2:-3}"        # 连续 N 次检查都空闲才启动
POLL_INTERVAL=60                    # 每次检查间隔 (秒)

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ──────────────────────────────────────────────
# Phase 0: 确保虚拟环境就绪
# ──────────────────────────────────────────────
ensure_env() {
    if [ ! -f "$VENV_PYTHON" ]; then
        log "虚拟环境不存在，正在安装..."
        bash "$SCRIPT_DIR/setup_env.sh" 2>&1 | tee -a "$EXP_LOG"
        if [ ! -f "$VENV_PYTHON" ]; then
            log "❌ 环境安装失败，退出"
            exit 1
        fi
        log "✅ 虚拟环境安装完成"
    else
        log "✅ 虚拟环境已存在: $VENV_PYTHON"
    fi
}

# ──────────────────────────────────────────────
# GPU 利用率获取
# ──────────────────────────────────────────────
get_gpu_util() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

get_gpu_mem_used() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

# ──────────────────────────────────────────────
# 等待 GPU 空闲
# ──────────────────────────────────────────────
wait_for_gpu() {
    log "开始监控 GPU (阈值 <${GPU_IDLE_THRESHOLD}%, 需连续 ${IDLE_CONFIRM_COUNT} 次确认)..."
    idle_count=0
    while true; do
        util=$(get_gpu_util)
        mem=$(get_gpu_mem_used)
        log "GPU 利用率: ${util}%  显存: ${mem}MiB  (空闲计数: ${idle_count}/${IDLE_CONFIRM_COUNT})"

        if [ -n "$util" ] && [ "$util" -lt "$GPU_IDLE_THRESHOLD" ]; then
            idle_count=$((idle_count + 1))
            if [ "$idle_count" -ge "$IDLE_CONFIRM_COUNT" ]; then
                log "✅ GPU 已连续 ${IDLE_CONFIRM_COUNT} 次检测为空闲，准备启动实验"
                return 0
            fi
        else
            idle_count=0
        fi
        sleep "$POLL_INTERVAL"
    done
}

# ──────────────────────────────────────────────
# 运行实验
# ──────────────────────────────────────────────
run_experiments() {
    log "=========================================="
    log "开始 DMGENet 实验复现"
    log "日志: $EXP_LOG"
    log "=========================================="

    run_step() {
        local desc="$1"; shift
        log "▶ $desc"
        "$@" >> "$EXP_LOG" 2>&1
        local rc=$?
        if [ $rc -ne 0 ]; then
            log "❌ 步骤失败 (exit $rc): $desc"
            log "   详情见: $EXP_LOG"
            return $rc
        fi
        log "✅ 完成: $desc"
    }

    # Step 0: 下载数据
    run_step "数据下载与预处理" \
        "$VENV_PYTHON" data/download_beijing_data.py || return 1

    # Step 1: 验证图构建
    run_step "验证图构建" \
        "$VENV_PYTHON" -c "
from _Support.Graph_Construction_Beijing_12 import *
adj_D, _, _ = calculate_the_distance_matrix(threshold=0.4)
print(f'Distance graph: {adj_D.shape}, edges={int(adj_D.sum())}')
adj_N, _, _ = calculate_the_neighbor_matrix(45)
print(f'Neighbor graph: {adj_N.shape}, edges={int(adj_N.sum())}')
adj_S, _, _ = calculate_the_similarity_matrix(threshold=0.6, target='PM25')
print(f'Similarity graph: {adj_S.shape}, edges={int(adj_S.sum())}')
import pandas as pd, torch
adj_F = pd.read_csv('./dataset/Beijing_12/POI/adjacency_matrix.csv', header=None)
adj_F = torch.where(torch.tensor(adj_F.values.astype(float))>=0.7, torch.tensor(adj_F.values.astype(float)), torch.zeros(12,12))
print(f'Functional graph: {adj_F.shape}, edges={int((adj_F>0).sum())}')
print('图构建验证通过')
" || return 1

    # Step 2: 训练基础模型 (4 图 × 5 变体 × 4 horizons)
    run_step "基础模型训练 (全量, 约 10-15h)" \
        "$VENV_PYTHON" exp_base_model.py || return 1

    # Step 3: 合并预测结果
    run_step "合并预测结果 (get_X_y)" \
        "$VENV_PYTHON" RLMC_final/get_X_y.py || return 1

    # Step 4: 计算历史误差
    run_step "计算历史误差 (calculating_errors)" \
        "$VENV_PYTHON" RLMC_final/calculating_errors.py || return 1

    # Step 5: RLMC 集成训练
    run_step "RLMC 集成训练 (10 runs × 4 horizons)" \
        "$VENV_PYTHON" train_RLMC_final.py || return 1

    # Step 6: 汇总结果
    run_step "汇总结果与论文对比" \
        "$VENV_PYTHON" run_experiments.py --phase summary --horizons 1 6 12 24 || true

    log "=========================================="
    log "✅ 所有实验完成！"
    log "结果对比: doc/reproduction_comparison.csv"
    log "RLMC 结果: RLMC_final_预测结果_Beijing_12/proposed/"
    log "基础模型: 预测结果_基础模型_Beijing_12/"
    log "=========================================="
}

# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
main() {
    log "DMGENet GPU 监控器启动 (PID=$$)"
    echo $$ > "$LOG_DIR/monitor.pid"

    ensure_env
    wait_for_gpu
    run_experiments &
    EXP_PID=$!
    echo "$EXP_PID" > "$PID_FILE"
    log "实验进程已启动 (PID=$EXP_PID)"
    log "查看实验日志: tail -f $EXP_LOG"

    wait "$EXP_PID"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        log "✅ 实验流程正常结束"
    else
        log "⚠️  实验流程以退出码 $EXIT_CODE 结束"
    fi
}

main

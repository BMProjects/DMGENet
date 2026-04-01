#!/usr/bin/env bash
# status.sh — 快速查看实验状态
# 用法: ./status.sh [watch]  (加 watch 参数则每5秒刷新)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

show_status() {
    clear 2>/dev/null || true
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║           DMGENet 实验状态监控                        ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""

    # GPU 状态
    echo "━━━ GPU 状态 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader 2>/dev/null | \
        awk -F',' '{printf "  %-20s  利用率:%-4s  显存:%s/%s  温度:%s\n",$1,$2,$3,$4,$5}'
    echo ""

    # 监控进程状态
    echo "━━━ 监控进程 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    MONITOR_PID_FILE="$LOG_DIR/monitor.pid"
    if [ -f "$MONITOR_PID_FILE" ]; then
        MPID=$(cat "$MONITOR_PID_FILE")
        if kill -0 "$MPID" 2>/dev/null; then
            echo "  ✅ 监控器运行中 (PID=$MPID)"
        else
            echo "  ❌ 监控器已退出 (PID=$MPID)"
        fi
    else
        echo "  ⚪ 监控器未启动"
        echo "     启动命令: nohup ./run_when_free.sh > logs/monitor.log 2>&1 & echo \$! > logs/monitor.pid"
    fi

    # 实验进程状态
    EXP_PID_FILE="$LOG_DIR/experiment.pid"
    if [ -f "$EXP_PID_FILE" ]; then
        EPID=$(cat "$EXP_PID_FILE")
        if kill -0 "$EPID" 2>/dev/null; then
            RUNTIME=$(ps -o etime= -p "$EPID" 2>/dev/null | tr -d ' ')
            echo "  🔥 实验进程运行中 (PID=$EPID, 运行时间=$RUNTIME)"
        else
            echo "  ⬜ 实验进程已结束 (PID=$EPID)"
        fi
    else
        echo "  ⚪ 实验尚未启动"
    fi
    echo ""

    # 实验进度检测
    echo "━━━ 实验进度 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    BASE="$SCRIPT_DIR"

    # 数据准备
    if [ -f "$BASE/dataset/Beijing_12/train_val_test_data/72_6/train_PM25.npz" ]; then
        echo "  ✅ Phase 0: 数据已准备"
    else
        echo "  ⬜ Phase 0: 数据未准备"
    fi

    # 基础模型结果
    MODEL_COUNT=$(find "$BASE/预测结果_基础模型_Beijing_12" -name "test_metrics.csv" 2>/dev/null | wc -l)
    EXPECTED=80  # 20 models × 4 horizons
    if [ "$MODEL_COUNT" -ge "$EXPECTED" ]; then
        echo "  ✅ Phase 1: 基础模型训练完成 ($MODEL_COUNT/$EXPECTED)"
    elif [ "$MODEL_COUNT" -gt 0 ]; then
        echo "  🔄 Phase 1: 基础模型训练中 ($MODEL_COUNT/$EXPECTED 完成)"
        # 显示最近保存的模型
        LATEST=$(find "$BASE/预测结果_基础模型_Beijing_12" -name "test_metrics.csv" 2>/dev/null | \
                 xargs ls -t 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            MODEL=$(echo "$LATEST" | awk -F'/' '{print $(NF-1)}')
            HORIZON=$(echo "$LATEST" | awk -F'/' '{print $(NF-2)}')
            echo "     最新: horizon=${HORIZON}h  model=${MODEL}"
        fi
    else
        echo "  ⬜ Phase 1: 基础模型未开始"
    fi

    # RLMC 数据准备
    RLMC_DATA=$(find "$BASE/RLMC_final_数据集_Beijing_12" -name "val_predictions_all.npy" 2>/dev/null | wc -l)
    if [ "$RLMC_DATA" -ge 4 ]; then
        echo "  ✅ Phase 2a: RLMC 数据已准备 ($RLMC_DATA/4 horizons)"
    elif [ "$RLMC_DATA" -gt 0 ]; then
        echo "  🔄 Phase 2a: RLMC 数据准备中 ($RLMC_DATA/4)"
    else
        echo "  ⬜ Phase 2a: RLMC 数据未准备"
    fi

    # RLMC 训练结果
    RLMC_RESULTS=$(find "$BASE/RLMC_final_预测结果_Beijing_12" -name "best_metrics.csv" 2>/dev/null | wc -l)
    if [ "$RLMC_RESULTS" -ge 4 ]; then
        echo "  ✅ Phase 2b: RLMC 训练完成 ($RLMC_RESULTS/4 horizons)"
        echo ""
        echo "━━━ 当前最优指标 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        for h in 1 6 12 24; do
            F="$BASE/RLMC_final_预测结果_Beijing_12/proposed/72/$h/best_metrics.csv"
            if [ -f "$F" ]; then
                METRICS=$(tail -1 "$F" | awk -F',' '{printf "RMSE=%-8s MAE=%-8s IA=%s",$1,$2,$3}')
                echo "  h=${h}h: $METRICS"
            fi
        done
    elif [ "$RLMC_RESULTS" -gt 0 ]; then
        echo "  🔄 Phase 2b: RLMC 训练中 ($RLMC_RESULTS/4 horizons 完成)"
    else
        echo "  ⬜ Phase 2b: RLMC 训练未开始"
    fi
    echo ""

    # 最近日志
    echo "━━━ 最近日志 (最后 8 行) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ -f "$LOG_DIR/experiment.log" ]; then
        tail -8 "$LOG_DIR/experiment.log" | sed 's/^/  /'
    elif [ -f "$LOG_DIR/monitor.log" ]; then
        tail -8 "$LOG_DIR/monitor.log" | sed 's/^/  /'
    else
        echo "  (无日志)"
    fi
    echo ""
    echo "━━━ 查看完整日志 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  tail -f $LOG_DIR/experiment.log   # 实验详细日志"
    echo "  tail -f $LOG_DIR/monitor.log      # 监控器日志"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
}

if [ "$1" = "watch" ]; then
    while true; do
        show_status
        sleep 5
    done
else
    show_status
fi

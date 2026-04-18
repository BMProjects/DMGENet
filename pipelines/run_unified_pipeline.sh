#!/usr/bin/env bash
# =============================================================================
# pipelines/run_unified_pipeline.sh
# Paper-aligned end-to-end experiment pipeline for the current DMGENet release.
# =============================================================================
# Protocol summary:
#   - local NPZ split generation from compact station panels
#   - train-only scaler fitting
#   - unified OSM-based F graph: radius=500 m, log1p scaling, threshold=0.65
#   - result roots:
#       ./results/base_models/{dataset}/
#       ./results/rlmc_data/{dataset}/
#       ./results/rlmc/{dataset}/
#
# Public benchmark scope:
#   Beijing_12, Beijing_Recent_12, Chengdu_10, Delhi_NCT_Meteo
#
# Usage:
#   bash pipelines/run_unified_pipeline.sh
#   bash pipelines/run_unified_pipeline.sh beijing_12 chengdu
#
# Environment variables:
#   SKIP_PREPARE_SPLITS=1
#   SKIP_GRAPHS=1
#   SKIP_BASE=1
#   SKIP_RLMC_PREP=1
#   SKIP_RLMC=1
#   RLMC_REPEAT=5
#   RLMC_EPISODES=60
#   SEED=2026
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_phase() { echo -e "\n${BOLD}${CYAN}━━ Phase $1: $2 ━━${NC}"; }
log_ok()    { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
log_err()   { echo -e "${RED}[✗]${NC} $1" >&2; }
log_info()  { echo -e "    $1"; }

DEFAULT_DATASETS="beijing_12 beijing_recent chengdu delhi"
DATASETS="${*:-$DEFAULT_DATASETS}"

SKIP_PREPARE_SPLITS="${SKIP_PREPARE_SPLITS:-0}"
SKIP_GRAPHS="${SKIP_GRAPHS:-0}"
SKIP_BASE="${SKIP_BASE:-0}"
SKIP_RLMC_PREP="${SKIP_RLMC_PREP:-0}"
SKIP_RLMC="${SKIP_RLMC:-0}"
RLMC_REPEAT="${RLMC_REPEAT:-5}"
RLMC_EPISODES="${RLMC_EPISODES:-60}"
SEED="${SEED:-2026}"

dataset_name() {
    case "$1" in
        beijing_12)      echo "Beijing_12" ;;
        beijing_recent)  echo "Beijing_Recent_12" ;;
        chengdu)         echo "Chengdu_10" ;;
        delhi)           echo "Delhi_NCT_Meteo" ;;
        *)
            log_err "Unknown dataset alias: $1 (supported: beijing_12, beijing_recent, chengdu, delhi)"
            exit 1
            ;;
    esac
}

city_flag() {
    case "$1" in
        Beijing_12)          echo "beijing_12" ;;
        Beijing_Recent_12)   echo "beijing_recent" ;;
        Chengdu_10)          echo "chengdu" ;;
        Delhi_NCT_Meteo)     echo "delhi" ;;
    esac
}

num_nodes() {
    local meta="./dataset/$1/graphs/graphs_metadata.json"
    if [ -f "${meta}" ]; then
        python3 -c "import json; print(json.load(open('${meta}'))['num_nodes'])" 2>/dev/null || echo "12"
    else
        case "$1" in
            Chengdu_10) echo "8" ;;
            *)          echo "12" ;;
        esac
    fi
}

if [ ! -f "train_base.py" ] || [ ! -f "train_rlmc.py" ]; then
    log_err "Please run this script from the DMGENet repository root."
    exit 1
fi

SCRIPT_START=$(date +%s)
FAILED=()

echo -e "\n${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   DMGENet unified experiment pipeline  $(date '+%Y-%m-%d %H:%M')        ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
echo -e "Datasets: ${DATASETS}"
echo -e "OSM F graph: radius=500m, log1p, eps=0.65 | seed=${SEED} | RLMC=${RLMC_REPEAT}x${RLMC_EPISODES} episodes\n"

for KEY in ${DATASETS}; do
    DS=$(dataset_name "${KEY}")
    CITY_START=$(date +%s)
    FAILED_PHASE=0

    echo -e "\n${BOLD}┌─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}│  Dataset: ${DS}${NC}"
    echo -e "${BOLD}└─────────────────────────────────────────────────────────┘${NC}"

    set +e

    log_phase 1 "Local split preparation — ${DS}"
    if [ "${SKIP_PREPARE_SPLITS}" = "1" ]; then
        log_warn "Phase 1 skipped (SKIP_PREPARE_SPLITS=1)"
    else
        python data/compact_dataset.py --dataset "${DS}"
        if [ $? -ne 0 ]; then
            log_err "Phase 1 failed: ${DS}"; FAILED_PHASE=1
        else
            log_ok "Phase 1 complete: ${DS}"
        fi
    fi

    log_phase 2 "Graph construction — ${DS}"
    GRAPHS_EXIST=0
    [ -f "dataset/${DS}/graphs/F_adj.npy" ] && GRAPHS_EXIST=1

    if [ "${SKIP_GRAPHS}" = "1" ] || [ "${FAILED_PHASE}" = "1" ]; then
        [ "${FAILED_PHASE}" = "1" ] && log_warn "Phase 2 skipped (upstream failure)" || log_warn "Phase 2 skipped (SKIP_GRAPHS=1)"
    elif [ "${GRAPHS_EXIST}" = "1" ]; then
        log_warn "Phase 2 skipped (graphs already exist; delete dataset/${DS}/graphs/ to rebuild)"
    else
        python graphs/build_graphs.py \
            --dataset "${DS}" \
            --functional-radius 500 \
            --functional-threshold 0.65
        if [ $? -ne 0 ]; then
            log_err "Phase 2 failed: ${DS}"; FAILED_PHASE=1
        else
            log_ok "Phase 2 complete: ${DS}"
        fi
    fi

    N=$(num_nodes "${DS}")

    log_phase 3 "Base-model training — ${DS} (N=${N}, 4 graphs x 4 horizons)"
    if [ "${SKIP_BASE}" = "1" ] || [ "${FAILED_PHASE}" = "1" ]; then
        [ "${FAILED_PHASE}" = "1" ] && log_warn "Phase 3 skipped (upstream failure)" || log_warn "Phase 3 skipped (SKIP_BASE=1)"
    else
        python train_base.py \
            --dataset "${DS}" \
            --pred-lens 1 6 12 24 \
            --seed "${SEED}"
        if [ $? -ne 0 ]; then
            log_err "Phase 3 failed: ${DS}"; FAILED_PHASE=1
        else
            log_ok "Phase 3 complete: ${DS}"
            log_info "Results: ./results/base_models/${DS}/"
        fi
    fi

    log_phase 4 "RLMC data preparation — ${DS}"
    if [ "${SKIP_RLMC_PREP}" = "1" ] || [ "${FAILED_PHASE}" = "1" ]; then
        [ "${FAILED_PHASE}" = "1" ] && log_warn "Phase 4 skipped (upstream failure)" || log_warn "Phase 4 skipped (SKIP_RLMC_PREP=1)"
    else
        python rlmc/prepare_data.py --dataset "${DS}"
        if [ $? -ne 0 ]; then
            log_err "Phase 4a failed (prepare_data): ${DS}"; FAILED_PHASE=1
        fi
        if [ "${FAILED_PHASE}" = "0" ]; then
            python rlmc/errors.py --dataset "${DS}"
            if [ $? -ne 0 ]; then
                log_err "Phase 4b failed (errors): ${DS}"; FAILED_PHASE=1
            fi
        fi
        [ "${FAILED_PHASE}" = "0" ] && log_ok "Phase 4 complete: ${DS}" && log_info "Data: ./results/rlmc_data/${DS}/"
    fi

    log_phase 5 "RL integration training — ${DS} (${RLMC_REPEAT} runs x 4 horizons)"
    if [ "${SKIP_RLMC}" = "1" ] || [ "${FAILED_PHASE}" = "1" ]; then
        [ "${FAILED_PHASE}" = "1" ] && log_warn "Phase 5 skipped (upstream failure)" || log_warn "Phase 5 skipped (SKIP_RLMC=1)"
    else
        python train_rlmc.py \
            --dataset "${DS}" \
            --pred-lens 1 6 12 24 \
            --repeat "${RLMC_REPEAT}" \
            --episodes "${RLMC_EPISODES}"
        if [ $? -ne 0 ]; then
            log_err "Phase 5 failed: ${DS}"; FAILED_PHASE=1
        else
            log_ok "Phase 5 complete: ${DS}"
            log_info "Results: ./results/rlmc/${DS}/"
        fi
    fi

    set -e

    ELAPSED=$(( $(date +%s) - CITY_START ))
    echo -e "\n${DS} elapsed: $((ELAPSED/3600))h$(( (ELAPSED%3600)/60 ))m$((ELAPSED%60))s"

    if [ "${FAILED_PHASE}" = "1" ]; then
        FAILED+=("${DS}")
        log_warn "${DS} contains failed stages"
    else
        log_ok "${DS} completed all stages"
    fi
done

TOTAL=$(( $(date +%s) - SCRIPT_START ))
echo -e "\n${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Finished  $(date '+%H:%M:%S')  Total: $((TOTAL/3600))h$(( (TOTAL%3600)/60 ))m$((TOTAL%60))s${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo -e "${GREEN}All datasets completed successfully.${NC}"
    echo ""
    echo "Result directories:"
    for KEY in ${DATASETS}; do
        DS=$(dataset_name "${KEY}")
        echo "  Base models: ./results/base_models/${DS}/"
        echo "  RLMC:        ./results/rlmc/${DS}/"
    done
else
    echo -e "${YELLOW}Failed datasets: ${FAILED[*]}${NC}"
    echo ""
    echo "Resume examples:"
    for DS in "${FAILED[@]}"; do
        FLAG=$(city_flag "${DS}")
        echo "  SKIP_PREPARE_SPLITS=1 SKIP_GRAPHS=1 bash pipelines/run_unified_pipeline.sh ${FLAG}"
    done
fi

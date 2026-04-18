"""
Per-sample error histories for each base model — the RL agent's state input.

For each (dataset, horizon, metric) triple, compute the sample-level error
(MAE/MAPE/SMAPE) of every base model on the val and test splits, shift each
series forward by one step (so at time t the state holds the error from t-1),
and write the combined series as a CSV keyed by model name.

Output directory: ./results/rlmc_data/{dataset}/proposed/{seq_len}/{pred_len}/
  combined_val_{mae|mape|smape}_history_errors.csv
  combined_test_{mae|mape|smape}_history_errors.csv

Example:
  python rlmc/errors.py --dataset Beijing_12
  python rlmc/errors.py --dataset Chengdu_10 --pred-lens 1 6 12 24
"""

import os
import argparse
import numpy as np
import pandas as pd

MODEL_GROUPS = {
    'proposed': ['Model_D', 'Model_N', 'Model_S', 'Model_POI'],
}

SEQ_LEN   = 72
PRED_LENS = [1, 6, 12, 24]
ERRORS    = ['mae', 'mape', 'smape']


def calculate_smape(y_true, y_pred):
    epsilon = 1e-8
    return 100 * np.mean(
        np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon),
        axis=(1, 2),
    )


def calculate_mape(y_true, y_pred):
    epsilon = 1e-8
    return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)), axis=(1, 2))


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=(1, 2))


def compute_error_metrics(true_path: str, pred_path: str, metric: str) -> np.ndarray:
    y_true = np.load(true_path)
    y_pred = np.load(pred_path)
    if metric == 'smape':
        return calculate_smape(y_true, y_pred)
    elif metric == 'mape':
        return calculate_mape(y_true, y_pred)
    elif metric == 'mae':
        return calculate_mae(y_true, y_pred)
    raise ValueError(f"Unsupported error metric: {metric!r}")


def shift_error_values(error_values: np.ndarray) -> np.ndarray:
    """Shift an error series one step forward so state[t] reflects the error at t-1."""
    shifted = np.roll(error_values, shift=1)
    shifted[0] = error_values[-1]
    return shifted


def main():
    parser = argparse.ArgumentParser(description="Compute base-model error histories for RLMC state")
    parser.add_argument("--dataset",   required=True,
                        help="Dataset name, e.g. Beijing_12, Chengdu_10, Delhi_NCT_Meteo")
    parser.add_argument("--seq-len",   type=int, default=SEQ_LEN)
    parser.add_argument("--pred-lens", nargs="+", type=int, default=PRED_LENS)
    parser.add_argument("--base-results-dir", default=None,
                        help="Base-model results dir (default: ./results/base_models/{dataset})")
    parser.add_argument("--rlmc-data-dir", default=None,
                        help="RLMC output dir (default: ./results/rlmc_data/{dataset})")
    args = parser.parse_args()

    dataset   = args.dataset
    seq_len   = args.seq_len
    pred_lens = args.pred_lens

    base_results_dir  = args.base_results_dir or f"./results/base_models/{dataset}"
    rlmc_dataset_root = args.rlmc_data_dir    or f"./results/rlmc_data/{dataset}"

    print(f"\n{'='*60}")
    print(f"Computing error histories: {dataset}")
    print(f"{'='*60}")

    for group_name, model_list in MODEL_GROUPS.items():
        for predict_len in pred_lens:
            for error in ERRORS:
                val_dfs  = []
                test_dfs = []
                any_found = False

                for model in model_list:
                    in_dir = os.path.join(base_results_dir, str(seq_len), str(predict_len), model)

                    val_y    = os.path.join(in_dir, "val_y_inverse.npy")
                    val_pred = os.path.join(in_dir, "val_predictions_inverse.npy")
                    test_y   = os.path.join(in_dir, "test_y_inverse.npy")
                    test_pred= os.path.join(in_dir, "test_predictions_inverse.npy")

                    if not (os.path.exists(val_y) and os.path.exists(val_pred)):
                        print(f"  [warn] missing val files: {in_dir}")
                        continue
                    if not (os.path.exists(test_y) and os.path.exists(test_pred)):
                        print(f"  [warn] missing test files: {in_dir}")
                        continue

                    val_err  = shift_error_values(compute_error_metrics(val_y,  val_pred,  error))
                    test_err = shift_error_values(compute_error_metrics(test_y, test_pred, error))

                    val_dfs.append(pd.DataFrame(val_err,   columns=[model]))
                    test_dfs.append(pd.DataFrame(test_err, columns=[model]))
                    any_found = True

                if not any_found:
                    print(f"  [warn] [{group_name}] pred_len={predict_len} {error}: no valid files, skipping")
                    continue

                out_dir = os.path.join(rlmc_dataset_root, group_name, str(seq_len), str(predict_len))
                os.makedirs(out_dir, exist_ok=True)

                if val_dfs:
                    pd.concat(val_dfs, axis=1).to_csv(
                        os.path.join(out_dir, f"combined_val_{error}_history_errors.csv"), index=False
                    )
                if test_dfs:
                    pd.concat(test_dfs, axis=1).to_csv(
                        os.path.join(out_dir, f"combined_test_{error}_history_errors.csv"), index=False
                    )
                print(f"  [{group_name}] pred_len={predict_len} {error}: saved")

    print(f"\nDone. Error history directory: {rlmc_dataset_root}")


if __name__ == "__main__":
    main()

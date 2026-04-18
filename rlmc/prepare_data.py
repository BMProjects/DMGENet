"""
Collect per-split predictions from the four base-model runs and stack them
into the tensor layout consumed by train_rlmc.py.

Output directory: ./results/rlmc_data/{dataset}/proposed/{seq_len}/{pred_len}/
  val_predictions_all.npy          # (samples, 4, N, pred_len) normalized
  val_predictions_inverse_all.npy  # (samples, 4, N, pred_len) original scale
  test_predictions_all.npy
  test_predictions_inverse_all.npy
  val_X.npy, val_y.npy, val_y_inverse.npy
  test_X.npy, test_y.npy, test_y_inverse.npy

Example:
  python rlmc/prepare_data.py --dataset Beijing_12
  python rlmc/prepare_data.py --dataset Chengdu_10 --pred-lens 1 6 12 24
"""

import os
import shutil
import argparse
import numpy as np

MODEL_GROUPS = {
    'proposed': ['Model_D', 'Model_N', 'Model_S', 'Model_POI'],
}

SEQ_LEN     = 72
PRED_LENS   = [1, 6, 12, 24]


def main():
    parser = argparse.ArgumentParser(description="Stack base-model predictions for RLMC training")
    parser.add_argument("--dataset",    required=True,
                        help="Dataset name, e.g. Beijing_12, Chengdu_10, Delhi_NCT_Meteo")
    parser.add_argument("--seq-len",    type=int, default=SEQ_LEN)
    parser.add_argument("--pred-lens",  nargs="+", type=int, default=PRED_LENS)
    parser.add_argument("--base-results-dir", default=None,
                        help="Base-model results dir (default: ./results/base_models/{dataset})")
    parser.add_argument("--rlmc-data-dir", default=None,
                        help="RLMC output dir (default: ./results/rlmc_data/{dataset})")
    args = parser.parse_args()

    dataset  = args.dataset
    seq_len  = args.seq_len
    pred_lens = args.pred_lens

    base_results_dir  = args.base_results_dir or f"./results/base_models/{dataset}"
    rlmc_dataset_root = args.rlmc_data_dir    or f"./results/rlmc_data/{dataset}"

    print(f"\n{'='*60}")
    print(f"Collecting base-model predictions: {dataset}")
    print(f"{'='*60}")

    for group_name, models in MODEL_GROUPS.items():
        for predict_len in pred_lens:
            dst_folder = os.path.join(rlmc_dataset_root, group_name, str(seq_len), str(predict_len))
            os.makedirs(dst_folder, exist_ok=True)

            # X/y files are identical across models, so copy from any one (Model_D).
            base_dir = os.path.join(base_results_dir, str(seq_len), str(predict_len), "Model_D")
            for fname in ["val_X.npy", "val_y.npy", "val_y_inverse.npy",
                          "test_X.npy", "test_y.npy", "test_y_inverse.npy"]:
                src = os.path.join(base_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, dst_folder)
                else:
                    print(f"  [warn] missing file: {src}")

            # Stack predictions from all four base models along axis=1.
            for split in ["val", "test"]:
                pred_list         = []
                pred_inverse_list = []

                for model in models:
                    model_dir = os.path.join(base_results_dir, str(seq_len), str(predict_len), model)
                    pred_file         = os.path.join(model_dir, f"{split}_predictions.npy")
                    pred_inverse_file = os.path.join(model_dir, f"{split}_predictions_inverse.npy")

                    if os.path.exists(pred_file) and os.path.exists(pred_inverse_file):
                        pred_list.append(np.load(pred_file))
                        pred_inverse_list.append(np.load(pred_inverse_file))
                    else:
                        print(f"  [warn] missing {model} {split} (pred_len={predict_len})")

                if pred_list:
                    stacked         = np.stack(pred_list,         axis=1)  # (samples, 4, N, pred_len)
                    stacked_inverse = np.stack(pred_inverse_list, axis=1)
                    out_pred         = os.path.join(dst_folder, f"{split}_predictions_all.npy")
                    out_pred_inverse = os.path.join(dst_folder, f"{split}_predictions_inverse_all.npy")
                    np.save(out_pred,         stacked)
                    np.save(out_pred_inverse, stacked_inverse)
                    print(f"  [{group_name}] pred_len={predict_len} {split}: shape={stacked.shape} -> {out_pred}")
                else:
                    print(f"  [warn] [{group_name}] pred_len={predict_len} {split}: no valid predictions, skipping")

    print(f"\nDone. RLMC data directory: {rlmc_dataset_root}")


if __name__ == "__main__":
    main()

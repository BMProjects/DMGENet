"""
RLMC (reinforcement-learned model combination) training entry point.

Trains a DDPG actor-critic that outputs softmax weights over the four
single-graph DMGENet predictors (Model_D/N/S/POI). Expects that the base
predictions and error histories have already been produced by the base-model
pipeline (see pipelines/run_unified_pipeline.sh).

  - state_dim is fixed at 12 (feature dimension, independent of node count)
  - action_dim is fixed at 4 (one weight per base predictor)
  - Training is repeated N times (default 5) and the run with the lowest
    val_loss is kept, to reduce variance from DDPG instability.

Example:
  python train_rlmc.py --dataset Beijing_12
  python train_rlmc.py --dataset Chengdu_10 --repeat 5 --episodes 60
"""

import os
import time
import argparse
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from rlmc.actor_critic import DDPG, RLMC_env, ReplayBuffer
from utils.metrics import metric_multi_sites

torch.backends.cudnn.benchmark    = True
torch.backends.cudnn.deterministic = False


# ──────────────────────────────────────────────────────────────────────────────
# Experiment driver
# ──────────────────────────────────────────────────────────────────────────────

class Exp:
    """Single RLMC training run (one repeat) on pre-computed base predictions."""

    def __init__(
        self,
        state_dim, action_dim, gamma, lr_actor, lr_critic, tau, hidden_dim,
        episodes, max_steps, batch_size, replay_buffer_size,
        # train split (note: we use the val split as the RL training set because
        # the base models were trained on the train split)
        train_X, train_history_errors, train_y, train_predictions_all,
        # test split
        test_X, test_history_errors, test_y, test_predictions_all,
        test_y_inverse, test_predictions_inverse_all,
        results_folder,
        early_stop_patience: int = 20,
        min_episodes: int = 15,
        early_stop_min_delta: float = 1e-6,
    ):
        self.episodes   = episodes
        self.max_steps  = max_steps
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.update_freq = 10
        self.early_stop_patience = early_stop_patience
        self.min_episodes        = min_episodes
        self.early_stop_min_delta = early_stop_min_delta

        self.train_X               = train_X
        self.train_history_errors  = train_history_errors
        self.train_y               = train_y
        self.train_predictions_all = train_predictions_all

        self.test_X                          = test_X
        self.test_history_errors             = test_history_errors
        self.test_y                          = test_y
        self.test_predictions_all            = test_predictions_all
        self.test_y_inverse                  = test_y_inverse
        self.test_predictions_inverse_all    = test_predictions_inverse_all

        self.env = RLMC_env(train_X, train_history_errors, train_y, train_predictions_all, action_dim)
        self.agent  = DDPG(state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, tau)
        self.buffer = ReplayBuffer(replay_buffer_size)
        self.results_folder = results_folder

    def load_best_actor(self):
        best_path = os.path.join(self.results_folder, "best_actor.pth")
        if os.path.exists(best_path):
            device = next(self.agent.actor.parameters()).device
            self.agent.actor.load_state_dict(torch.load(best_path, map_location=device))
        return best_path

    def train(self):
        episode_time    = []
        episode_rewards = []
        val_losses      = []
        best_val_loss   = float("inf")
        no_improve_count = 0

        for episode in range(self.episodes):
            t0 = time.time()
            observation, error = self.env.reset()
            episode_reward = 0

            for step in range(self.max_steps):
                action = self.agent.select_action(observation, error, noise_std=0.1, eval_mode=False)
                next_obs, next_err, reward, done, _ = self.env.step(action)
                self.buffer.push(observation, error, action, reward, next_obs, next_err, float(done))

                if len(self.buffer) > self.batch_size and step % self.update_freq == 0:
                    self.agent.update(self.buffer, self.batch_size)

                observation, error = next_obs, next_err
                episode_reward += reward
                if done:
                    break

            ep_time = time.time() - t0
            episode_time.append(ep_time)
            episode_rewards.append(episode_reward)

            val_loss = self.compute_loss(
                self.train_X, self.train_history_errors, self.train_y, self.train_predictions_all
            )
            val_losses.append(val_loss)
            print(
                f"Episode: {episode+1:<4} | Time: {ep_time:6.2f}s "
                f"| Reward: {episode_reward:12.6f} | Val Loss: {val_loss:15.10f}",
                end=" ",
            )
            if val_loss < (best_val_loss - self.early_stop_min_delta):
                best_val_loss = val_loss
                no_improve_count = 0
                torch.save(self.agent.actor.state_dict(),  os.path.join(self.results_folder, "best_actor.pth"))
                torch.save(self.agent.critic.state_dict(), os.path.join(self.results_folder, "best_critic.pth"))
                print(f"[new best val_loss={best_val_loss:.10f}]")
            else:
                no_improve_count += 1
                print()
                if (
                    (episode + 1) >= self.min_episodes
                    and self.early_stop_patience > 0
                    and no_improve_count >= self.early_stop_patience
                ):
                    print(
                        f"[early stop @ episode {episode+1}] "
                        f"no improvement for {no_improve_count} episodes "
                        f"(min_delta={self.early_stop_min_delta:g})"
                    )
                    break

        pd.DataFrame({
            "episode":      list(range(1, len(val_losses) + 1)),
            "episode_time": episode_time,
            "reward":       episode_rewards,
            "val_loss":     val_losses,
        }).to_csv(os.path.join(self.results_folder, "train_log.csv"), index=False)

    def compute_loss(self, X, errors, y, preds_all, batch_size=256):
        """Mean MSE on the given set. Used for model selection — never call on test."""
        self.agent.actor.eval()
        device = next(self.agent.actor.parameters()).device
        X_t  = torch.FloatTensor(X).to(device)
        e_t  = torch.FloatTensor(errors).to(device)
        y_t  = torch.FloatTensor(y).to(device)
        p_t  = torch.FloatTensor(preds_all).to(device)
        n    = X_t.shape[0]
        all_preds, all_targets = [], []
        with torch.no_grad():
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                logits  = self.agent.actor(X_t[s:e], e_t[s:e])
                action  = torch.softmax(logits, dim=1)
                bm_pred = p_t[s:e]
                pred    = torch.sum(action.unsqueeze(-1).unsqueeze(-1) * bm_pred, dim=1)
                all_preds.append(pred.cpu())
                all_targets.append(y_t[s:e].cpu())
        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        loss = torch.mean((all_preds - all_targets) ** 2, dim=[1, 2]).mean().item()
        self.agent.actor.train()
        return loss

    def test(self, batch_size=256):
        self.load_best_actor()
        device = next(self.agent.actor.parameters()).device
        self.agent.actor.eval()

        pred_list, true_list, weight_list = [], [], []
        n = len(self.test_X)
        with torch.no_grad():
            for s in range(0, n, batch_size):
                e   = min(s + batch_size, n)
                obs = torch.FloatTensor(self.test_X[s:e]).to(device)
                err = torch.FloatTensor(self.test_history_errors[s:e]).to(device)
                bm  = torch.FloatTensor(self.test_predictions_inverse_all[s:e]).to(device)
                logits = self.agent.actor(obs, err)
                action = torch.softmax(logits, dim=1)
                fp = torch.sum(action[:, :, None, None] * bm, dim=1)
                pred_list.append(fp.cpu().numpy())
                true_list.append(self.test_y_inverse[s:e])
                weight_list.append(action.cpu().numpy())

        pred    = np.concatenate(pred_list,   axis=0)
        true    = np.concatenate(true_list,   axis=0)
        weights = np.concatenate(weight_list, axis=0)

        np.save(os.path.join(self.results_folder, "final_pred.npy"), pred)
        pd.DataFrame(weights).to_csv(os.path.join(self.results_folder, "weights.csv"), index=False)

        metrics = metric_multi_sites(pred, true)
        print(
            f"test_MAE:{metrics[0]:.3f}  test_RMSE:{metrics[1]:.3f}  "
            f"test_IA:{metrics[2]:.4f}  test_R2:{metrics[3]:.4f}"
        )
        pd.DataFrame([metrics], columns=["test_MAE", "test_RMSE", "test_IA", "test_R2"]) \
          .to_csv(os.path.join(self.results_folder, "test_metrics.csv"), index=False)

        self.agent.actor.train()
        return metrics, pred, true, weights


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RLMC training")
    parser.add_argument("--dataset",      required=True,
                        help="Dataset name, e.g. Beijing_12, Chengdu_10, Delhi_NCT_Meteo")
    parser.add_argument("--repeat",       type=int,   default=5,
                        help="Number of independent runs (default: 5)")
    parser.add_argument("--episodes",     type=int,   default=60,
                        help="Maximum episodes per run (default: 60; usually stops earlier via patience)")
    parser.add_argument("--max-steps",    type=int,   default=1000)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--replay-size",  type=int,   default=2000)
    parser.add_argument("--hidden-dim",   type=int,   default=64)
    parser.add_argument("--gamma",        type=float, default=0.99)
    parser.add_argument("--tau",          type=float, default=0.005)
    parser.add_argument("--lr-actor",     type=float, default=1e-4)
    parser.add_argument("--lr-critic",    type=float, default=1e-3)
    parser.add_argument("--pred-lens",    nargs="+",  type=int, default=[1, 6, 12, 24])
    parser.add_argument("--seq-len",      type=int,   default=72)
    parser.add_argument("--rlmc-data-dir",  default=None,
                        help="RLMC input directory (default: ./results/rlmc_data/{dataset})")
    parser.add_argument("--rlmc-results-dir", default=None,
                        help="RLMC output directory (default: ./results/rlmc/{dataset})")
    # Empirically best val_loss appears around episode ~30; patience=20 avoids
    # running the fixed 200-episode loop that wasted ~85% of wall time in v1.
    parser.add_argument("--early-stop-patience", type=int, default=20,
                        help="Episodes without improvement before stopping (default: 20; set > --episodes to disable)")
    parser.add_argument("--min-episodes",        type=int, default=15,
                        help="Minimum episodes before early stop can trigger — lets the replay buffer fill (default: 15)")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-6,
                        help="Minimum val_loss improvement to count as a new best (default: 1e-6)")
    args = parser.parse_args()

    dataset   = args.dataset
    seq_len   = args.seq_len
    STATE_DIM = 12     # feature dimension (same across datasets)
    ACTION_DIM = 4     # four base predictors: Model_D / N / S / POI

    model_group = "proposed"

    for predict_len in args.pred_lens:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset}   Model: {model_group}   Horizon: {predict_len}")
        print(f"{'='*70}")

        _rlmc_data_root    = args.rlmc_data_dir    or f"./results/rlmc_data/{dataset}"
        _rlmc_results_root = args.rlmc_results_dir or f"./results/rlmc/{dataset}"
        base_dir       = f"{_rlmc_data_root}/{model_group}/{seq_len}/{predict_len}"
        results_folder = f"{_rlmc_results_root}/{model_group}/{seq_len}/{predict_len}"
        os.makedirs(results_folder, exist_ok=True)

        # ── Load precomputed base-model outputs ─────────────────
        def load(name):
            return np.load(os.path.join(base_dir, name))

        def load_csv(name):
            return pd.read_csv(os.path.join(base_dir, name)).values.astype("float32")

        train_X                      = load("val_X.npy")
        train_history_errors         = load_csv("combined_val_mae_history_errors.csv")
        train_y                      = load("val_y.npy")
        train_predictions_all        = load("val_predictions_all.npy")

        test_X                       = load("test_X.npy")
        test_history_errors          = load_csv("combined_test_mae_history_errors.csv")
        test_y                       = load("test_y.npy")
        test_predictions_all         = load("test_predictions_all.npy")
        test_y_inverse               = load("test_y_inverse.npy")
        test_predictions_inverse_all = load("test_predictions_inverse_all.npy")

        # ── Repeated runs ───────────────────────────────────────
        best_val_metric = float("inf")
        best_run        = -1
        all_results     = []

        for run in range(args.repeat):
            print(f"\n--- Run {run+1}/{args.repeat} ---")
            seed = run
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            exp = Exp(
                state_dim             = STATE_DIM,
                action_dim            = ACTION_DIM,
                gamma                 = args.gamma,
                lr_actor              = args.lr_actor,
                lr_critic             = args.lr_critic,
                tau                   = args.tau,
                hidden_dim            = args.hidden_dim,
                episodes              = args.episodes,
                max_steps             = args.max_steps,
                batch_size            = args.batch_size,
                replay_buffer_size    = args.replay_size,
                train_X               = train_X,
                train_history_errors  = train_history_errors,
                train_y               = train_y,
                train_predictions_all = train_predictions_all,
                test_X                = test_X,
                test_history_errors   = test_history_errors,
                test_y                = test_y,
                test_predictions_all  = test_predictions_all,
                test_y_inverse        = test_y_inverse,
                test_predictions_inverse_all = test_predictions_inverse_all,
                results_folder        = results_folder,
                early_stop_patience   = args.early_stop_patience,
                min_episodes          = args.min_episodes,
                early_stop_min_delta  = args.early_stop_min_delta,
            )

            exp.train()
            exp.load_best_actor()
            val_loss = exp.compute_loss(train_X, train_history_errors, train_y, train_predictions_all)
            metrics, pred, true, weights = exp.test()
            mae, rmse, ia, r2 = metrics

            print(f"Run {run+1}: MAE={mae:.3f}  RMSE={rmse:.3f}  IA={ia:.4f}  R2={r2:.4f}  val_loss={val_loss:.6f}")
            all_results.append({"run": run+1, "MAE": mae, "RMSE": rmse, "IA": ia, "R2": r2, "val_loss": val_loss})

            if val_loss < best_val_metric:
                best_val_metric = val_loss
                best_run = run
                pd.DataFrame([metrics], columns=["MAE","RMSE","IA","R2"]) \
                  .to_csv(os.path.join(results_folder, "best_metrics.csv"), index=False)
                torch.save(exp.agent.actor.state_dict(), os.path.join(results_folder, "best_actor.pth"))
                np.save(os.path.join(results_folder, "best_pred.npy"),    pred)
                np.save(os.path.join(results_folder, "best_true.npy"),    true)
                np.save(os.path.join(results_folder, "best_weights.npy"), weights)
                pd.DataFrame(pred.reshape(pred.shape[0], -1)).to_csv(os.path.join(results_folder, "best_pred.csv"),  index=False)
                pd.DataFrame(true.reshape(true.shape[0], -1)).to_csv(os.path.join(results_folder, "best_true.csv"),  index=False)
                with open(os.path.join(results_folder, "best_run.txt"), "w") as f:
                    f.write(str(run + 1))
                print(f"[best] Run {run+1}, val_loss={val_loss:.6f}")

        # ── Aggregate across runs ───────────────────────────────
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(results_folder, "all_runs_metrics.csv"), index=False)

        mean = df.mean(numeric_only=True)
        std  = df.std(numeric_only=True)

        print(f"\n{'='*70}")
        print(f"Best Run (by val_loss): {best_run+1},  val_loss={best_val_metric:.6f}")
        print(
            f"MAE  = {mean['MAE']:.3f} +/- {std['MAE']:.3f}\n"
            f"RMSE = {mean['RMSE']:.3f} +/- {std['RMSE']:.3f}\n"
            f"IA   = {mean['IA']:.4f} +/- {std['IA']:.4f}\n"
            f"R2   = {mean['R2']:.4f} +/- {std['R2']:.4f}"
        )
        print(f"{'='*70}")

        summary_df = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "IA", "R2"],
            "Mean":   [mean["MAE"], mean["RMSE"], mean["IA"], mean["R2"]],
            "Std":    [std["MAE"],  std["RMSE"],  std["IA"],  std["R2"]],
        })
        summary_df.to_csv(os.path.join(results_folder, "summary_metrics.csv"), index=False)
        print(f"Summary saved to: {results_folder}/summary_metrics.csv")


if __name__ == "__main__":
    main()

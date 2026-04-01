import os
import time
import numpy as np
import pandas as pd
import random
import torch
from RLMC_final.RLMC import DDPG, RLMC_env, ReplayBuffer
from utils.metrics import metric_mutil_sites
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class Exp:
    def __init__(self,
                 state_dim,
                 action_dim,
                 gamma,
                 lr_actor,
                 lr_critic,
                 tau,
                 hidden_dim,
                 episodes,
                 max_steps,
                 batch_size,
                 replay_buffer_size,

                 # ===== RL训练数据 =====
                 train_X,
                 train_history_errors,
                 train_y,
                 train_predictions_all,

                 # ===== test数据 =====
                 test_X,
                 test_history_errors,
                 test_y,
                 test_predictions_all,
                 test_y_inverse,
                 test_predictions_inverse_all,

                 results_folder):
        
        self.episodes = episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.update_freq = 10

        # ========= train dataset =========
        self.train_X = train_X
        self.train_history_errors = train_history_errors
        self.train_y = train_y
        self.train_predictions_all = train_predictions_all

        # ========= test dataset =========
        self.test_X = test_X
        self.test_history_errors = test_history_errors
        self.test_y = test_y
        self.test_predictions_all = test_predictions_all

        self.test_y_inverse = test_y_inverse
        self.test_predictions_inverse_all = test_predictions_inverse_all

        # ========= env =========
        self.env = RLMC_env(
            train_X,
            train_history_errors,
            train_y,
            train_predictions_all,
            action_dim
        )

        # ========= agent =========
        self.agent = DDPG(
            state_dim,
            action_dim,
            hidden_dim,
            lr_actor,
            lr_critic,
            gamma,
            tau
        )

        self.buffer = ReplayBuffer(replay_buffer_size)
        self.results_folder = results_folder

    # =========================================================
    # TRAIN
    # =========================================================
    def train(self):
        print("🚀 RL训练开始")

        episode_time = []
        episode_rewards = []
        val_losses = []

        best_val_loss = float("inf")  # 基于验证集（RL训练数据）选择最佳模型

        for episode in range(self.episodes):
            start_time = time.time()

            observation, error = self.env.reset()
            episode_reward = 0

            for step in range(self.max_steps):
                action = self.agent.select_action(
                    observation,
                    error,
                    noise_std=0.1,
                    eval_mode=False
                )

                next_observation, next_error, reward, done, _ = self.env.step(action)

                self.buffer.push(
                    observation,
                    error,
                    action,
                    reward,
                    next_observation,
                    next_error,
                    float(done)
                )

                # 更新网络
                if len(self.buffer) > self.batch_size and step % self.update_freq == 0:
                    self.agent.update(self.buffer, self.batch_size)

                observation, error = next_observation, next_error
                episode_reward += reward

                if done:
                    break

            end_time = time.time()
            episode_time.append(end_time - start_time)
            episode_rewards.append(episode_reward)

            # ===== 验证集 loss（使用RL训练数据评估，不使用测试集选模型）=====
            val_loss = self.compute_test_loss(
                self.train_X,
                self.train_history_errors,
                self.train_y,
                self.train_predictions_all,
                batch_size=256
            )
            val_losses.append(val_loss)

            # ===== 打印信息 =====
            print(
            f"Episode: {episode+1:<4}"
            f" | Time: {end_time-start_time:6.2f}s"
            f" | Reward: {episode_reward:12.6f}"
            f" | Val Loss: {val_loss:15.10f}", end=" "
            )

            # ===== 基于验证集 loss 保存最佳模型 =====
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.agent.actor.state_dict(),
                    os.path.join(self.results_folder, "best_actor.pth")
                )
                torch.save(
                    self.agent.critic.state_dict(),
                    os.path.join(self.results_folder, "best_critic.pth")
                )
                print(f"✅ Episode {episode+1}: 保存新的最佳模型, val_loss={best_val_loss:.10f}")
            else:
                print("")

        # ===== 保存训练日志 =====
        log_df = pd.DataFrame({
            "episode": list(range(1, self.episodes + 1)),
            "episode_time": episode_time,
            "reward": episode_rewards,
            "val_loss": val_losses
        })
        log_path = os.path.join(self.results_folder, "train_log.csv")
        log_df.to_csv(log_path, index=False)
        print(f"📄 训练日志已保存: {log_path}")

    def compute_test_loss(self, test_X, test_error, test_y, test_predictions_all, batch_size=256):
        """
        在测试集上计算全局 MSE loss（拼接所有预测值再计算）
        支持批量计算，加速 GPU
        """
        self.agent.actor.eval()
        self.agent.critic.eval()
        
        device = next(self.agent.actor.parameters()).device  # 自动使用模型所在 GPU/CPU

        test_X = torch.FloatTensor(test_X).to(device)
        test_error = torch.FloatTensor(test_error).to(device)
        test_y = torch.FloatTensor(test_y).to(device)
        test_predictions_all = torch.FloatTensor(test_predictions_all).to(device)
        
        n_samples = test_X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size  # 向上取整

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_samples)
                
                obs_batch = test_X[start:end]
                err_batch = test_error[start:end]
                target_batch = test_y[start:end]

                # Actor 预测动作
                logits = self.agent.actor(obs_batch, err_batch)
                action = torch.softmax(logits, dim=1)

                # 获取对应 bm_pred
                bm_pred_batch = test_predictions_all[start:end] # [batch, action_dim, N, T_out]

                # 扩展 action 维度，对齐 bm_pred
                # action: [batch, action_dim] -> [batch, action_dim, 1, 1]
                action_exp = action.unsqueeze(-1).unsqueeze(-1)

                # 预测值 pred: [batch, N, T_out]
                pred = torch.sum(action_exp * bm_pred_batch, dim=1)

                # 保存到列表，统一拼接
                all_preds.append(pred.cpu())
                all_targets.append(target_batch.cpu())

        # 拼接整个测试集
        all_preds = torch.cat(all_preds, dim=0)       # [n_samples, N, T_out]
        all_targets = torch.cat(all_targets, dim=0)   # [n_samples, N, T_out]

        # 每个样本的 MSE
        sample_loss = torch.mean((all_preds - all_targets) ** 2, dim=[1, 2])  # [n_samples]

        # 平均所有样本
        avg_loss = sample_loss.mean().item()

        self.agent.actor.train()
        self.agent.critic.train()
        return avg_loss

    # =========================================================
    # TEST
    # =========================================================
    def test(self, batch_size=256, run_id=None):

        print("🧪 开始测试 (batch mode)")

        # =========================
        # 🔴 加载最佳模型
        # =========================
        best_model_path = os.path.join(self.results_folder, "best_actor.pth")

        if os.path.exists(best_model_path):
            self.agent.actor.load_state_dict(
                torch.load(best_model_path, map_location="cuda")
            )
            print(f"✅ 已加载最佳模型: {best_model_path}")
        else:
            print("⚠️ 未找到best_actor.pth，使用当前模型")

        self.agent.actor.eval()

        pred = []
        true = []
        weights = []

        total = len(self.test_X)

        with torch.no_grad():
            for start in range(0, total, batch_size):

                end = min(start + batch_size, total)

                data = torch.FloatTensor(self.test_X[start:end]).cuda()
                error = torch.FloatTensor(self.test_history_errors[start:end]).cuda()
                bm_pred = torch.FloatTensor(
                    self.test_predictions_inverse_all[start:end]
                ).cuda()

                logits = self.agent.actor(data, error)
                action = torch.softmax(logits, dim=1)

                final_pred = torch.sum(
                    action[:, :, None, None] * bm_pred,
                    dim=1
                )

                target = self.test_y_inverse[start:end]

                pred.append(final_pred.cpu().numpy())
                true.append(target)
                weights.append(action.cpu().numpy())

        pred = np.concatenate(pred, axis=0)
        true = np.concatenate(true, axis=0)
        weights = np.concatenate(weights, axis=0)

        # =========================
        # 保存预测
        # =========================
        np.save(os.path.join(self.results_folder, "final_pred.npy"), pred)

        pd.DataFrame(weights).to_csv(
            os.path.join(self.results_folder, "weights.csv"),
            index=False
        )

        metrics = metric_mutil_sites(pred, true)

        print(
            f"test_MAE:{metrics[0]:.3f}, "
            f"test_RMSE:{metrics[1]:.3f}, "
            f"test_IA:{metrics[2]:.4f}, "
            f"test_R2:{metrics[3]:.4f}"
        )

        pd.DataFrame([metrics],
            columns=["test_MAE","test_RMSE","test_IA","test_R2"]
        ).to_csv(
            os.path.join(self.results_folder,"test_metrics.csv"),
            index=False
        )

        self.agent.actor.train()
        # ⭐⭐⭐关键
        return metrics, pred, true, weights
    

# =========================
# 超参数
# =========================
dataset = 'Beijing_12'
seq_len = 72
state_dim = 12
hidden_dim = 64

gamma = 0.99
tau = 0.005

lr_actor = 1e-4
lr_critic = 1e-3

episodes = 200
max_steps = 1000
batch_size = 64
replay_buffer_size = 2000


# def main():
#     for predict_len in [1, 6, 12, 24]:
#         for model in [
#             'proposed',
#             # 'wo_gated_TCN',
#             # 'wo_gcn',
#             # 'wo_gat',
#             # 'wo_ASTAM',
#             # 'wo_D',
#             # 'wo_N',
#             # 'wo_S',
#             # 'wo_POI'
#         ]:

#             print("\n============================")
#             print(f"Model: {model}  Horizon:{predict_len}")
#             print("============================")

#             # ⭐ action维度
#             if model in ['wo_D', 'wo_N', 'wo_S', 'wo_POI']:
#                 action_dim = 3
#             else:
#                 action_dim = 4

#             base_dir = f'./RLMC_final_数据集_{dataset}/{model}/{seq_len}/{predict_len}'
#             results_folder = f'./RLMC_final_预测结果_{dataset}/{model}/{seq_len}/{predict_len}'
#             os.makedirs(results_folder, exist_ok=True)

#             # =========================
#             # 训练数据（原val当train）
#             # =========================
#             train_X = np.load(os.path.join(base_dir, 'val_X.npy'))
#             train_history_errors = pd.read_csv(
#                 os.path.join(base_dir, 'combined_val_mae_history_errors.csv')
#             ).values.astype('float32')

#             train_y = np.load(os.path.join(base_dir, 'val_y.npy'))
#             train_predictions_all = np.load(
#                 os.path.join(base_dir, 'val_predictions_all.npy')
#             )

#             # =========================
#             # test数据
#             # =========================
#             test_X = np.load(os.path.join(base_dir, 'test_X.npy'))
#             test_history_errors = pd.read_csv(
#                 os.path.join(base_dir, 'combined_test_mae_history_errors.csv')
#             ).values.astype('float32')

#             test_y = np.load(os.path.join(base_dir, 'test_y.npy'))
#             test_predictions_all = np.load(
#                 os.path.join(base_dir, 'test_predictions_all.npy')
#             )

#             test_y_inverse = np.load(
#                 os.path.join(base_dir, 'test_y_inverse.npy')
#             )
#             test_predictions_inverse_all = np.load(
#                 os.path.join(base_dir, 'test_predictions_inverse_all.npy')
#             )

#             # =========================
#             # 创建实验
#             # =========================
#             exp = Exp(
#                 state_dim,
#                 action_dim,
#                 gamma,
#                 lr_actor,
#                 lr_critic,
#                 tau,
#                 hidden_dim,
#                 episodes,
#                 max_steps,
#                 batch_size,
#                 replay_buffer_size,

#                 # train
#                 train_X,
#                 train_history_errors,
#                 train_y,
#                 train_predictions_all,

#                 # test
#                 test_X,
#                 test_history_errors,
#                 test_y,
#                 test_predictions_all,
#                 test_y_inverse,
#                 test_predictions_inverse_all,

#                 results_folder
#             )

#             print("🚀 训练开始")
#             exp.train()
#             print("✅ 训练结束")

#             print("🧪 测试开始")
#             exp.test()
#             print("✅ 测试结束")


# # =========================
# # 入口
# # =========================
# if __name__ == "__main__":
#     main()

def main():
    repeat_times = 10  # ⭐训练次数

    for predict_len in [1, 6, 12, 24]:
        for model in [
            'proposed',
        ]:

            print("\n============================")
            print(f"Model: {model}  Horizon:{predict_len}")
            print("============================")

            # ⭐ action维度
            if model in ['wo_D', 'wo_N', 'wo_S', 'wo_POI']:
                action_dim = 3
            else:
                action_dim = 4

            base_dir = f'./RLMC_final_数据集_{dataset}/{model}/{seq_len}/{predict_len}'
            results_folder = f'./RLMC_final_预测结果_{dataset}/{model}/{seq_len}/{predict_len}'
            os.makedirs(results_folder, exist_ok=True)

            # =========================
            # 数据加载
            # =========================
            train_X = np.load(os.path.join(base_dir, 'val_X.npy'))
            train_history_errors = pd.read_csv(
                os.path.join(base_dir, 'combined_val_mae_history_errors.csv')
            ).values.astype('float32')

            train_y = np.load(os.path.join(base_dir, 'val_y.npy'))
            train_predictions_all = np.load(
                os.path.join(base_dir, 'val_predictions_all.npy')
            )

            test_X = np.load(os.path.join(base_dir, 'test_X.npy'))
            test_history_errors = pd.read_csv(
                os.path.join(base_dir, 'combined_test_mae_history_errors.csv')
            ).values.astype('float32')

            test_y = np.load(os.path.join(base_dir, 'test_y.npy'))
            test_predictions_all = np.load(
                os.path.join(base_dir, 'test_predictions_all.npy')
            )

            test_y_inverse = np.load(
                os.path.join(base_dir, 'test_y_inverse.npy')
            )
            test_predictions_inverse_all = np.load(
                os.path.join(base_dir, 'test_predictions_inverse_all.npy')
            )

            # =========================
            # ⭐ 多次训练
            # =========================
            best_val_metric = float('inf')
            best_run = -1

            all_results = []  # ⭐保存所有run指标

            for run in range(repeat_times):

                print(f"\n🔥 第 {run+1}/{repeat_times} 次训练")

                # ⭐ 固定随机种子
                seed = run
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                exp = Exp(
                    state_dim,
                    action_dim,
                    gamma,
                    lr_actor,
                    lr_critic,
                    tau,
                    hidden_dim,
                    episodes,
                    max_steps,
                    batch_size,
                    replay_buffer_size,

                    train_X,
                    train_history_errors,
                    train_y,
                    train_predictions_all,

                    test_X,
                    test_history_errors,
                    test_y,
                    test_predictions_all,
                    test_y_inverse,
                    test_predictions_inverse_all,

                    results_folder
                )

                print("🚀 训练开始")
                exp.train()
                print("✅ 训练结束")

                # ===== 基于验证集（RL训练数据）选择最佳 run =====
                val_loss = exp.compute_test_loss(
                    train_X, train_history_errors, train_y, train_predictions_all,
                    batch_size=256
                )

                print("🧪 测试开始")

                # ⭐⭐⭐ 关键：接收预测结果
                metrics, pred, true, weights = exp.test(run_id=run)

                print("✅ 测试结束")

                mae, rmse, ia, r2 = metrics

                print(
                    f"📊 Run {run+1}: "
                    f"MAE={mae:.3f}, RMSE={rmse:.3f}, IA={ia:.4f}, R2={r2:.4f}, "
                    f"Val Loss={val_loss:.6f}"
                )

                # =========================
                # 保存每次指标
                # =========================
                all_results.append({
                    "run": run + 1,
                    "MAE": mae,
                    "RMSE": rmse,
                    "IA": ia,
                    "R2": r2,
                    "val_loss": val_loss
                })

                # =========================
                # ⭐ 基于验证集 loss 更新最优模型（不使用测试集选择）
                # =========================
                if val_loss < best_val_metric:
                    pd.DataFrame([metrics], columns=["MAE","RMSE","IA","R2"]).to_csv(os.path.join(results_folder, "best_metrics.csv"),index=False)

                    best_val_metric = val_loss
                    best_run = run

                    print(f"🏆 当前最优: Run {run+1}, val_loss={val_loss:.6f}")

                    # 保存模型
                    torch.save(
                        exp.agent.actor.state_dict(),
                        os.path.join(results_folder, "best_actor.pth")
                    )

                    # ⭐ 保存最优预测
                    np.save(os.path.join(results_folder, "best_pred.npy"), pred)
                    np.save(os.path.join(results_folder, "best_true.npy"), true)
                    np.save(os.path.join(results_folder, "best_weights.npy"), weights)

                    # ⭐ CSV（方便论文画图）
                    pd.DataFrame(pred.reshape(pred.shape[0], -1)).to_csv(
                        os.path.join(results_folder, "best_pred.csv"),
                        index=False
                    )

                    pd.DataFrame(true.reshape(true.shape[0], -1)).to_csv(
                        os.path.join(results_folder, "best_true.csv"),
                        index=False
                    )

                    # ⭐ 记录best run
                    with open(os.path.join(results_folder, "best_run.txt"), "w") as f:
                        f.write(str(run + 1))

            # =========================
            # 保存所有 run 指标
            # =========================
            df = pd.DataFrame(all_results)

            all_csv_path = os.path.join(results_folder, "all_runs_metrics.csv")
            df.to_csv(all_csv_path, index=False)

            print(f"📁 所有实验结果已保存: {all_csv_path}")

            # =========================
            # mean ± std
            # =========================
            mean = df.mean(numeric_only=True)
            std = df.std(numeric_only=True)

            print("\n============================")
            print(f"🎯 最优Run(by val_loss): {best_run+1}, val_loss={best_val_metric:.6f}")
            print(
                f"📊 MAE = {mean['MAE']:.3f} ± {std['MAE']:.3f}\n"
                f"📊 RMSE = {mean['RMSE']:.3f} ± {std['RMSE']:.3f}\n"
                f"📊 IA = {mean['IA']:.4f} ± {std['IA']:.4f}\n"
                f"📊 R2 = {mean['R2']:.4f} ± {std['R2']:.4f}"
            )
            print("============================")

            # =========================
            # 保存 summary（论文用）
            # =========================
            summary_df = pd.DataFrame({
                "Metric": ["MAE", "RMSE", "IA", "R2"],
                "Mean": [mean["MAE"], mean["RMSE"], mean["IA"], mean["R2"]],
                "Std": [std["MAE"], std["RMSE"], std["IA"], std["R2"]],
            })

            summary_path = os.path.join(results_folder, "summary_metrics.csv")
            summary_df.to_csv(summary_path, index=False)

            print(f"📁 汇总结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
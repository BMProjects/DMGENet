import os
import numpy as np
import pandas as pd


def calculate_smape(y_true, y_pred):
    epsilon = 1e-8  # 防止除零错误
    smape = 100 * np.mean(
        np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon),
        axis=(1, 2)
    )
    return smape


def calculate_mape(y_true, y_pred):
    epsilon = 1e-8  # 防止除零错误
    mape = 100 * np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)), axis=(1, 2))
    return mape


def calculate_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred), axis=(1, 2))
    return mae


def compute_error_metrics(true_file_path, pred_file_path, metric='smape'):
    y_true = np.load(true_file_path)
    y_pred = np.load(pred_file_path)

    if metric == 'smape':
        error_values = calculate_smape(y_true, y_pred)
    elif metric == 'mape':
        error_values = calculate_mape(y_true, y_pred)
    elif metric == 'mae':
        error_values = calculate_mae(y_true, y_pred)
    else:
        raise ValueError("Unsupported metric. Choose from 'smape', 'mape', 'mae'.")

    return error_values


def shift_error_values(error_values):
    shifted_error_values = np.roll(error_values, shift=1)
    shifted_error_values[0] = error_values[-1]  # 第一个元素设置为最后一个元素的值
    return shifted_error_values


# 定义所有模型分组
models = {
    'proposed': ['Model_D', 'Model_N', 'Model_S', 'Model_POI'],
    # 'wo_gated_TCN': ['Model_D_wo_gated_TCN', 'Model_N_wo_gated_TCN', 'Model_S_wo_gated_TCN', 'Model_POI_wo_gated_TCN'],
    # 'wo_gcn': ['Model_D_wo_gcn', 'Model_N_wo_gcn', 'Model_S_wo_gcn', 'Model_POI_wo_gcn'],
    # 'wo_gat': ['Model_D_wo_gat', 'Model_N_wo_gat', 'Model_S_wo_gat', 'Model_POI_wo_gat'],
    # 'wo_ASTAM': ['Model_D_wo_ASTAM', 'Model_N_wo_ASTAM', 'Model_S_wo_ASTAM', 'Model_POI_wo_ASTAM'],
    # 'wo_D': ['Model_N', 'Model_S', 'Model_POI'],
    # 'wo_N': ['Model_D', 'Model_S', 'Model_POI'],
    # 'wo_S': ['Model_D', 'Model_N', 'Model_POI'],
    # 'wo_POI': ['Model_D', 'Model_N', 'Model_S']
}

# 配置运行参数
seq_len = 72
predict_lens = [1, 6, 12, 24]  # 预测长度
errors = ['mae', 'mape', 'smape']  # 错误指标
dataset = 'Beijing_12'

# 遍历所有运行配置
for group_name, model_list in models.items():  # 遍历每个分组及其对应模型
    for predict_len in predict_lens:
        for error in errors:
            val_dfs = []
            test_dfs = []
            any_file_found = False  # 标记是否找到任何有效文件

            # 遍历每个模型
            for model in model_list:
                in_dir = f'./预测结果_基础模型_{dataset}/{seq_len}/{predict_len}/{model}'

                try:
                    # 检查文件是否存在
                    val_y_file = os.path.join(in_dir, 'val_y_inverse.npy')
                    val_pred_file = os.path.join(in_dir, 'val_predictions_inverse.npy')
                    test_y_file = os.path.join(in_dir, 'test_y_inverse.npy')
                    test_pred_file = os.path.join(in_dir, 'test_predictions_inverse.npy')

                    if not (os.path.exists(val_y_file) and os.path.exists(val_pred_file)):
                        print(f"Missing files for validation in {in_dir}")
                        continue
                    if not (os.path.exists(test_y_file) and os.path.exists(test_pred_file)):
                        print(f"Missing files for testing in {in_dir}")
                        continue

                    # 计算验证集和测试集的误差
                    val_error_values = compute_error_metrics(val_y_file, val_pred_file, metric=error)
                    test_error_values = compute_error_metrics(test_y_file, test_pred_file, metric=error)

                    # 计算历史误差
                    val_history_errors = shift_error_values(val_error_values)
                    test_history_errors = shift_error_values(test_error_values)

                    # 将误差值转换为 DataFrame
                    val_df = pd.DataFrame(val_history_errors, columns=[model])
                    test_df = pd.DataFrame(test_history_errors, columns=[model])

                    val_dfs.append(val_df)
                    test_dfs.append(test_df)
                    any_file_found = True  # 找到至少一个有效文件

                except FileNotFoundError as e:
                    print(f"File not found for model {model} in {in_dir}: {e}")

            # 如果没有找到任何文件，跳过当前配置
            if not any_file_found:
                print(f"No valid files found for group {group_name}, predict_len {predict_len}, error {error}")
                continue

            # 拼接所有模型的 DataFrame
            if val_dfs:
                combined_val_df = pd.concat(val_dfs, axis=1)
            if test_dfs:
                combined_test_df = pd.concat(test_dfs, axis=1)

            # 创建输出目录
            out_dir = f'./RLMC_final_数据集_{dataset}/{group_name}/{seq_len}/{predict_len}'
            os.makedirs(out_dir, exist_ok=True)

            # 保存结果为 CSV 文件
            if val_dfs:
                combined_val_df.to_csv(os.path.join(out_dir, f'combined_val_{error}_history_errors.csv'), index=False)
            if test_dfs:
                combined_test_df.to_csv(os.path.join(out_dir, f'combined_test_{error}_history_errors.csv'), index=False)

print('Done!')

import shutil
import os
import numpy as np

# 定义分组及其对应的模型列表
model_groups = {
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
dataset = 'Beijing_12'
seq_len = 72
predict_lens = [1, 6, 12, 24]  # 预测长度

# 遍历所有运行配置
for group_name, models in model_groups.items():  # 遍历每个分组及其对应的模型列表
    for predict_len in predict_lens:
        # 创建目标文件夹
        dst_folder = f'./RLMC_final_数据集_{dataset}/{group_name}/{seq_len}/{predict_len}'
        os.makedirs(dst_folder, exist_ok=True)

        # 复制基础文件
        base_dir = f'./预测结果_基础模型_{dataset}/{seq_len}/{predict_len}/Model_D'  # 使用第一个模型的基础文件夹路径
        val_X = os.path.join(base_dir, 'val_X.npy')
        val_y = os.path.join(base_dir, 'val_y.npy')
        val_y_inverse = os.path.join(base_dir, 'val_y_inverse.npy')
        test_X = os.path.join(base_dir, 'test_X.npy')
        test_y = os.path.join(base_dir, 'test_y.npy')
        test_y_inverse = os.path.join(base_dir, 'test_y_inverse.npy')

        # 检查文件是否存在并复制
        if os.path.exists(val_X): shutil.copy(val_X, dst_folder)
        if os.path.exists(val_y): shutil.copy(val_y, dst_folder)
        if os.path.exists(val_y_inverse): shutil.copy(val_y_inverse, dst_folder)
        if os.path.exists(test_X): shutil.copy(test_X, dst_folder)
        if os.path.exists(test_y): shutil.copy(test_y, dst_folder)
        if os.path.exists(test_y_inverse): shutil.copy(test_y_inverse, dst_folder)

        # 处理预测文件
        for i in ['val', 'test']:  # 分别处理验证集和测试集
            pred_all = []
            pred_inverse_all = []

            for model in models:  # 遍历分组中的每个模型
                model_base_dir = f'./预测结果_基础模型_{dataset}/{seq_len}/{predict_len}/{model}'
                val_pred = os.path.join(model_base_dir, f'{i}_predictions.npy')
                val_pred_inverse = os.path.join(model_base_dir, f'{i}_predictions_inverse.npy')

                # 检查文件是否存在
                if os.path.exists(val_pred) and os.path.exists(val_pred_inverse):
                    pred_all.append(np.load(val_pred))
                    pred_inverse_all.append(np.load(val_pred_inverse))
                else:
                    print(f"Warning: Missing file for model {model} in {i} predictions")

            # 如果有数据则保存堆叠后的结果
            if pred_all and pred_inverse_all:
                pred_all = np.stack(pred_all, axis=1)
                pred_inverse_all = np.stack(pred_inverse_all, axis=1)
                np.save(os.path.join(dst_folder, f'{i}_predictions_all.npy'), pred_all)
                np.save(os.path.join(dst_folder, f'{i}_predictions_inverse_all.npy'), pred_inverse_all)

print('Done!')

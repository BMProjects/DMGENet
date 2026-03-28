import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model.model_1 import Model

from data.dataloader_Beijing_12 import Dataloader_Beijing_12

from _Support.Graph_Construction_Beijing_12 import calculate_the_distance_matrix
from _Support.Graph_Construction_Beijing_12 import calculate_the_neighbor_matrix
from _Support.Graph_Construction_Beijing_12 import calculate_the_similarity_matrix

from utils.metrics import metric_mutil_sites
from utils.tools import adjust_learning_rate, EarlyStopping, setup_seed, plot_loss

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False


# Beijing_12距离图
adj_matrix_D, edge_index_D, edge_weight_D = calculate_the_distance_matrix(threshold=0.4)
adj_matrix_D = torch.tensor(adj_matrix_D, dtype=torch.float).cuda()
edge_index_D = torch.tensor(edge_index_D, dtype=torch.long).cuda()
edge_weight_D = torch.tensor(edge_weight_D, dtype=torch.float).cuda()

# Beijing_12邻居图
adj_matrix_N, edge_index_N, edge_weight_N = calculate_the_neighbor_matrix(45)
adj_matrix_N = torch.tensor(adj_matrix_N, dtype=torch.float).cuda()
edge_index_N = torch.tensor(edge_index_N, dtype=torch.long).cuda()
edge_weight_N = torch.tensor(edge_weight_N, dtype=torch.float).cuda()

# Beijing_12分布相似图
adj_matrix_S, edge_index_S, edge_weight_S = calculate_the_similarity_matrix(threshold=0.6, target='PM25')
adj_matrix_S = torch.tensor(adj_matrix_S, dtype=torch.float).cuda()
edge_index_S = torch.tensor(edge_index_S, dtype=torch.long).cuda()
edge_weight_S = torch.tensor(edge_weight_S, dtype=torch.float).cuda()

# Beijing_12功能相似图
adj_matrix_POI = pd.read_csv('./dataset/Beijing_12/POI/adjacency_matrix.csv', header=None)
adj_matrix_POI = torch.tensor(adj_matrix_POI.values, dtype=torch.float).cuda()
# 只有大于等于阈值的权重被保留，其余设为 0
adj_matrix_POI = torch.where(adj_matrix_POI >= 0.7, adj_matrix_POI, torch.zeros_like(adj_matrix_POI))
print("Functional similarity matrix - 阈值: 0.7")


class Exp_model:
    def __init__(self, model_name, model, epoch, learning_rate, target, batch_size, num_workers, dataset, seq_len, predict_len):
        self.model_name = model_name
        self.model = model.cuda()
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.target = target

        print(f'chosen dataset:{dataset} model_name:{self.model_name} forecasting target:{self.target} predict_len:{predict_len}')

        # 预测结果保存路径
        self.results_folder = os.path.join(f'./预测结果_基础模型_{dataset}', str(seq_len), str(predict_len), self.model_name)
        os.makedirs(self.results_folder, exist_ok=True)

        # 数据集文件夹路径
        root_path = f'./dataset/{dataset}/train_val_test_data/{seq_len}_{predict_len}'

        # dataset
        if dataset == 'Beijing_12':
            self.train_dataloader = Dataloader_Beijing_12(os.path.join(root_path, f'train_{self.target}.npz'), 'train', batch_size, num_workers, target)
            self.val_dataloader = Dataloader_Beijing_12(os.path.join(root_path, f'val_{self.target}.npz'), 'val', batch_size, num_workers, target)
            self.test_dataloader = Dataloader_Beijing_12(os.path.join(root_path, f'test_{self.target}.npz'), 'test', batch_size, num_workers, target)
            self.train_loader = self.train_dataloader.get_dataloader()
            self.val_loader = self.val_dataloader.get_dataloader()
            self.test_loader = self.test_dataloader.get_dataloader()
        else:
            raise ValueError('Unsupported dataset: {}'.format(dataset))

    def val(self, criterion):
        val_loss = []
        test_loss = []

        self.model.eval()
        for i, (features, target) in enumerate(self.val_loader):
            features = features.cuda()
            target = target.cuda()
            pred, true = self.model(features), target
            loss = criterion(pred, true)
            val_loss.append(loss.item())
        val_loss = np.average(val_loss)

        for i, (features, target) in enumerate(self.test_loader):
            features = features.cuda()
            target = target.cuda()
            pred, true = self.model(features), target
            loss = criterion(pred, true)
            test_loss.append(loss.item())
        test_loss = np.average(test_loss)
        self.model.train()

        return val_loss, test_loss

    def train(self):
        criterion = nn.MSELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_loss = []
        val_loss = []
        test_loss = []

        epoch_time = []

        self.model.train()
        time_start = time.time()

        # 初始化早停对象
        early_stopping = EarlyStopping(patience=7, verbose=True, path=os.path.join(self.results_folder, self.model_name + '.pth'))

        for epoch in range(self.epoch):
            epoch_train_loss = []
            epoch_start_time = time.time()
            for i, (features, target) in enumerate(self.train_loader):
                features = features.cuda()
                target = target.cuda()
                optim.zero_grad()
                pred, true = self.model(features), target
                loss = criterion(pred, true)
                epoch_train_loss.append(loss.item())
                loss.backward()
                optim.step()

            epoch_end_time = time.time()

            epoch_time.append(epoch_end_time - epoch_start_time)

            epoch_train_loss = np.average(epoch_train_loss)
            epoch_val_loss, epoch_test_loss = self.val(criterion)
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)
            test_loss.append(epoch_test_loss)
            print(
                "Epoch [{:<3}/{:<3}] cost time:{:.8f} train_loss:{:.8f} val_loss:{:.8f} test_loss:{:.5f}".format(epoch + 1, 100,
                                                                                                         epoch_end_time - epoch_start_time,
                                                                                                         epoch_train_loss,
                                                                                                         epoch_val_loss,
                                                                                                         epoch_test_loss), end=" ")
            adjust_learning_rate(optim, epoch + 1, self.learning_rate)

            # 调用早停
            early_stopping(epoch_val_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        time_end = time.time()
        print("训练时间为{:.2f}秒，即{:.2f}分钟".format(time_end - time_start, (time_end - time_start) / 60))

        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        test_loss = np.array(test_loss)
        epoch_time = np.array(epoch_time)
        train_loss_df = pd.DataFrame(
            {'epoch_time': epoch_time, 'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss})

        plot_loss(train_loss, val_loss, test_loss)

        # train_loss saving
        train_loss_df.to_csv(os.path.join(self.results_folder, 'loss.csv'), index=True, index_label='epoch')

        # 加载早停保存的最佳模型
        best_model_path = os.path.join(self.results_folder, self.model_name + '.pth')
        self.model.load_state_dict(torch.load(best_model_path))

    def _evaluate(self, dataloader, loader, flag):
        predictions = []
        trues = []
        features_list = []

        for i, (features, target) in enumerate(loader):
            features = features.cuda()
            features_np = features.cpu().numpy()
            features_list.append(features_np)
            with torch.no_grad():
                pred = self.model(features)
            true = target
            predictions.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        all_features = np.concatenate(features_list, axis=0)
        trues = np.concatenate(trues, axis=0)
        predictions = np.concatenate(predictions, axis=0)

        print(
            f'{flag}_all_features shape:{all_features.shape}, {flag}_predictions shape:{predictions.shape}, {flag}_trues shape:{trues.shape}')

        np.save(os.path.join(self.results_folder, f'{flag}_X.npy'), all_features)
        np.save(os.path.join(self.results_folder, f'{flag}_y.npy'), trues)
        np.save(os.path.join(self.results_folder, f'{flag}_predictions.npy'), predictions)

        trues_inverse = dataloader.inverse_transform(trues)
        predictions_inverse = dataloader.inverse_transform(predictions)

        np.save(os.path.join(self.results_folder, f'{flag}_y_inverse.npy'), trues_inverse)
        np.save(os.path.join(self.results_folder, f'{flag}_predictions_inverse.npy'), predictions_inverse)

        metrics = metric_mutil_sites(predictions_inverse, trues_inverse)

        # printing metrics
        print(f'{flag}_MAE:{metrics[0]:.3f}, {flag}_RMSE:{metrics[1]:.3f}, {flag}_IA:{metrics[2]:.4f}, {flag}_R2:{metrics[3]:.3f}')

        # saving metrics
        metrics_df = pd.DataFrame([metrics], columns=[f'{flag}_MAE', f'{flag}_RMSE', f'{flag}_IA', f'{flag}_R2'])
        metrics_df.to_csv(os.path.join(self.results_folder, f'{flag}_metrics.csv'), index=False)

    def test(self):
        self.model.eval()

        # 训练集
        # self._evaluate(self.train_dataloader, self.train_loader, 'train')

        # 验证集
        self._evaluate(self.val_dataloader, self.val_loader, 'val')

        # 测试集
        self._evaluate(self.test_dataloader, self.test_loader, 'test')

if __name__ == '__main__':
    # 模型参数
    in_channels = 12
    T_in = 72
    num_nodes = 12
    hidden_size = 64
    dropout = 0.2
    block_num = 2
    num_heads = 4
    alpha = 0.2
    apt_size = 10
    num_channels = [64, 64, 64, 64]

    # 训练参数
    dataset = 'Beijing_12'
    batch_size = 64
    epoch = 100
    learning_rate = 0.001
    num_workers = 3
    setup_seed(2026)

    for T_out in [1, 6, 12, 24]:
        models = {
            ################################################################################################################################################
            ###############################################################################################################################################
            'Model_D': Model(adj_matrix_D, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=True),
            'Model_N': Model(adj_matrix_N, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=True),
            'Model_S': Model(adj_matrix_S, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=True),
            'Model_POI': Model(adj_matrix_POI, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=True),
            ###############################################################################################################################################
            ################################################################################################################################################
            'Model_D_wo_gated_TCN': Model(adj_matrix_D, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=False, gcn_bool=True, gat_bool=True, ASTAM_bool=True),
            'Model_N_wo_gated_TCN': Model(adj_matrix_N, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=False, gcn_bool=True, gat_bool=True, ASTAM_bool=True),
            'Model_S_wo_gated_TCN': Model(adj_matrix_S, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=False, gcn_bool=True, gat_bool=True, ASTAM_bool=True),
            'Model_POI_wo_gated_TCN': Model(adj_matrix_POI, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=False, gcn_bool=True, gat_bool=True,ASTAM_bool=True),
            ################################################################################################################################################
            ################################################################################################################################################
            'Model_D_wo_gcn': Model(adj_matrix_D, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=False, gat_bool=True, ASTAM_bool=True),
            'Model_N_wo_gcn': Model(adj_matrix_N, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=False, gat_bool=True, ASTAM_bool=True),
            'Model_S_wo_gcn': Model(adj_matrix_S, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=False, gat_bool=True, ASTAM_bool=True),
            'Model_POI_wo_gcn': Model(adj_matrix_POI, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=False, gat_bool=True, ASTAM_bool=True),
            ################################################################################################################################################
            ################################################################################################################################################
            'Model_D_wo_gat': Model(adj_matrix_D, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=False, ASTAM_bool=True),
            'Model_N_wo_gat': Model(adj_matrix_N, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=False, ASTAM_bool=True),
            'Model_S_wo_gat': Model(adj_matrix_S, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=False, ASTAM_bool=True),
            'Model_POI_wo_gat': Model(adj_matrix_POI, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=False, ASTAM_bool=True),
            ################################################################################################################################################
            ################################################################################################################################################
            'Model_D_wo_ASTAM': Model(adj_matrix_D, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=False),
            'Model_N_wo_ASTAM': Model(adj_matrix_N, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=False),
            'Model_S_wo_ASTAM': Model(adj_matrix_S, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=False),
            'Model_POI_wo_ASTAM': Model(adj_matrix_POI, in_channels, hidden_size, dropout, alpha, num_heads, num_channels=num_channels, apt_size=apt_size, num_nodes=num_nodes, num_block=block_num, T_in=T_in, predict_len=T_out,
            gated_TCN_bool=True, gcn_bool=True, gat_bool=True, ASTAM_bool=False),
            ################################################################################################################################################
            ################################################################################################################################################
        }
        for model_name, model_instance in models.items():
            exp = Exp_model(model_name, model_instance, epoch, learning_rate, "PM25", batch_size, num_workers, dataset, T_in, T_out)
            print(model_name + "训练开始！")
            exp.train()
            print(model_name + "训练结束！")
            print(model_name + "测试开始！")
            exp.test()
            print(model_name + "测试结束！")
            print("================================================================================================================================================")
            print()
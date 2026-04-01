import numpy as np
import random
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互后端，不弹窗，支持无头服务器
import matplotlib.pyplot as plt
import scipy.sparse as sp


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def normalize_adj(adj_matrix):
    """Row-normalize sparse matrix"""
    row_sum = np.array(adj_matrix.sum(1))
    r_inv_sqrt = np.power(row_sum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return adj_matrix.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def plot_loss(train_loss, val_loss, test_loss, save_path=None):
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.plot(test_loss, label='Test Loss')
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training, Validation and Test Loss')
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# def adjust_learning_rate(optimizer, epoch, start_lr):
#     lr = start_lr * (0.5 ** (epoch // 10))
#     if epoch % 10 == 0:
#         print('Updating learning rate to {:.2e}'.format(lr))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, start_lr):
    if epoch < 6:
        lr = start_lr
    else:
        lr = start_lr * (0.8 ** (epoch - 5))

    # print('Epoch [{:<3}/{:<3}], Learning Rate: {:.2e}'.format(epoch, 100, lr))
    print('Learning Rate: {:.2e}'.format(lr), end=" ")

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_RLMC(optimizer, epoch, start_lr):
    lr = start_lr * (0.8 ** epoch)

    print('Epoch [{:<3}/{:<3}], Learning Rate: {:.2e}'.format(epoch, 100, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# 定义早停类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): 允许验证集性能下降的次数
            verbose (bool): 是否打印早停信息
            delta (float): 最小性能提升的阈值
            path (str): 模型保存的路径
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存当前最佳模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
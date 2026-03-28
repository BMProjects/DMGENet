import numpy as np
from scipy.sparse import csr_matrix, diags


def symmetric_normalize_adjacency_matrix(adj_matrix):
    """
    计算对称归一化的邻接矩阵。

    参数:
    adj_matrix (csr_matrix): 邻接矩阵，可以是稀疏矩阵。

    返回:
    csr_matrix: 对称归一化的邻接矩阵。
    """
    if not isinstance(adj_matrix, csr_matrix):
        adj_matrix = csr_matrix(adj_matrix)

    # 计算度矩阵 D
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(degrees, -0.5).flatten()
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

    # 构建 D^(-1/2) 矩阵
    degree_inv_sqrt_matrix = diags(degree_inv_sqrt)

    # 计算对称归一化的邻接矩阵
    normalized_adj_matrix = degree_inv_sqrt_matrix @ adj_matrix @ degree_inv_sqrt_matrix

    return normalized_adj_matrix

import numpy as np
from scipy.sparse import csr_matrix, diags


def symmetric_normalize_adjacency_matrix(adj_matrix):
    """Return the symmetric normalization D^{-1/2} A D^{-1/2} of a (possibly sparse) adjacency."""
    if not isinstance(adj_matrix, csr_matrix):
        adj_matrix = csr_matrix(adj_matrix)

    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(degrees, -0.5).flatten()
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

    degree_inv_sqrt_matrix = diags(degree_inv_sqrt)
    normalized_adj_matrix = degree_inv_sqrt_matrix @ adj_matrix @ degree_inv_sqrt_matrix
    return normalized_adj_matrix

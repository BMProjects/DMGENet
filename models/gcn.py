import torch
import torch.nn as nn
import torch.nn.init as init
import math

def normalize_adj(adj, eps=1e-5):
    """Symmetric-normalize an adjacency matrix, with eps to guard zero-degree nodes.

    Args:
        adj: Adjacency matrix of shape [N, N].
        eps: Small positive constant added to degrees before the sqrt.

    Returns:
        Symmetrically normalized adjacency D^{-1/2} (A + I) D^{-1/2}.
    """
    if torch.any(torch.diag(adj) == 0):
        adj = adj + torch.eye(adj.size(0), device=adj.device)

    degree = adj.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + eps))
    adj_normalized = torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)
    return adj_normalized


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adj, bias=True):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adj = adj
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        :param x: [B, T, N, in]
        :return:  [B, T, N, out]
        """
        first_mul = torch.einsum('hi,btij->bthj', self.adj, x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul

        return graph_conv


class GCNLayer(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, adj, bias=True):
        super(GCNLayer, self).__init__()
        adj = normalize_adj(adj)
        self.conv1 = GraphConv(in_channels, hidden_size, adj, bias=bias)
        self.conv2 = GraphConv(hidden_size, out_channels, adj, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
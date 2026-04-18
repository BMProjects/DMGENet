"""
baselines/pm25_gnn/model.py
===========================
PM2.5-GNN local adaptation, kept as close as possible to the official implementation:
  - Official repository: https://github.com/shuowang-ai/PM2.5-GNN
  - Key files: model/PM25_GNN.py, graph.py

Local adaptation boundaries:
  1. Preserve the official GraphGNN + GRUCell + autoregressive decoder structure.
  2. Use native PyTorch accumulation instead of depending on `torch_scatter`.
  3. Construct the local city graph directly from station coordinates.
  4. For datasets without wind fields (for example Delhi), enable a static inverse-distance edge-weight fallback.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def scatter_add_nodes(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    src:   (B, E, C)
    index: (E,)
    out:   (B, N, C)
    """
    out = src.new_zeros(src.size(0), dim_size, src.size(-1))
    expand_index = index.view(1, -1, 1).expand(src.size(0), -1, src.size(-1))
    out.scatter_add_(1, expand_index, src)
    return out


class GraphGNN(nn.Module):
    def __init__(
        self,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        in_dim: int,
        out_dim: int,
        wind_mean: tuple[float, float] | None = None,
        wind_std: tuple[float, float] | None = None,
        use_wind: bool = True,
    ):
        super().__init__()
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        edge_attr_t = torch.as_tensor(edge_attr, dtype=torch.float32)
        self.register_buffer("edge_attr", edge_attr_t)
        edge_mean = edge_attr_t.mean(dim=0, keepdim=True)
        edge_std = edge_attr_t.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("edge_attr_norm", (edge_attr_t - edge_mean) / edge_std)

        self.use_wind = use_wind
        if use_wind:
            if wind_mean is None or wind_std is None:
                raise ValueError("wind_mean/std are required when use_wind=True")
            self.register_buffer("wind_mean", torch.as_tensor(wind_mean, dtype=torch.float32))
            self.register_buffer("wind_std", torch.as_tensor(wind_std, dtype=torch.float32).clamp_min(1e-6))

        edge_hidden = 32
        edge_out = 30
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 2 + 1, edge_hidden),
            nn.Sigmoid(),
            nn.Linear(edge_hidden, edge_out),
            nn.Sigmoid(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(edge_out, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C_in)
        returns: (B, N, C_out)
        """
        edge_src, edge_target = self.edge_index
        node_src = x[:, edge_src]
        node_target = x[:, edge_target]

        edge_attr = self.edge_attr.unsqueeze(0).expand(x.size(0), -1, -1)
        edge_attr_norm = self.edge_attr_norm.unsqueeze(0).expand(x.size(0), -1, -1)
        city_dist = edge_attr[..., 0].clamp_min(1e-3)
        city_direc = edge_attr[..., 1]

        if self.use_wind:
            src_wind = node_src[..., -2:] * self.wind_std.view(1, 1, 2) + self.wind_mean.view(1, 1, 2)
            src_wind_speed = src_wind[..., 0].clamp_min(0.0)
            src_wind_direc = torch.deg2rad(src_wind[..., 1])
            theta = torch.abs(city_direc - src_wind_direc)
            edge_weight = F.relu(3.0 * src_wind_speed * torch.cos(theta) / city_dist)
        else:
            edge_weight = 1.0 / city_dist

        out = torch.cat([node_src, node_target, edge_attr_norm, edge_weight.unsqueeze(-1)], dim=-1)
        out = self.edge_mlp(out)

        out_add = scatter_add_nodes(out, edge_target, dim_size=x.size(1))
        out_sub = scatter_add_nodes(-out, edge_src, dim_size=x.size(1))
        out = out_add + out_sub
        return self.node_mlp(out)


class PM25GNN(nn.Module):
    def __init__(
        self,
        hist_len: int,
        pred_len: int,
        in_dim: int,
        city_num: int,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        wind_mean: tuple[float, float] | None = None,
        wind_std: tuple[float, float] | None = None,
        use_wind: bool = True,
        hid_dim: int = 64,
        gnn_out: int = 13,
    ):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.hid_dim = hid_dim
        self.out_dim = 1
        self.gnn_out = gnn_out

        self.graph_gnn = GraphGNN(
            edge_index=edge_index,
            edge_attr=edge_attr,
            in_dim=in_dim,
            out_dim=gnn_out,
            wind_mean=wind_mean,
            wind_std=wind_std,
            use_wind=use_wind,
        )
        self.gru_cell = nn.GRUCell(in_dim + gnn_out, hid_dim)
        self.fc_out = nn.Linear(hid_dim, self.out_dim)

    def forward(self, pm25_hist: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """
        pm25_hist: (B, T_hist, N, 1)
        feature:   (B, T_hist + H, N, F_exog)
        returns:   (B, N, H)
        """
        batch_size, _, num_nodes, _ = pm25_hist.shape
        pm25_pred = []
        hn = torch.zeros(batch_size * num_nodes, self.hid_dim, device=pm25_hist.device)
        xn = pm25_hist[:, -1]  # (B, N, 1)

        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)  # (B,N,1+F)
            x_gnn = self.graph_gnn(x.contiguous())
            x = torch.cat([x_gnn, x], dim=-1)
            hn = self.gru_cell(x.view(batch_size * num_nodes, -1), hn)
            xn = hn.view(batch_size, num_nodes, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pred = torch.stack(pm25_pred, dim=1)  # (B,H,N,1)
        return pred[..., 0].permute(0, 2, 1).contiguous()

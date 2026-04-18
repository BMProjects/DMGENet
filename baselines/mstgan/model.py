"""
baselines/mstgan/model.py
=========================
Local adaptation of MSTGAN (the official class is named `MSTAN`):
  - Official repository: https://github.com/HPSCIL/MSTGAN-airquality-prediction
  - Key files:
      MSTGAN/code/model/MSTGAN.py
      MSTGAN/code/model/ST_block.py
      MSTGAN/code/model/moduel/Individual_Temporal_Pattern.py   (ITPM)
      MSTGAN/code/model/moduel/Global_ST_model.py                (GSTDM)
      MSTGAN/code/model/moduel/Local_Feature_model.py            (STDG_CGRU)
      MSTGAN/code/model/components/{embed,transformer,Bilinear_Temporal_Attention,Bilinear_Spatial_Attention,GRU}.py

Strictly preserved:
  1. The three-branch MST block structure (ITPM -> GSTDM -> STDG_CGRU + residual)
  2. The GSTDM bilinear temporal/spatial attention scoring from the paper
  3. The STDG-modulated Chebyshev convolution design inside STDG_CGRU
  4. The prediction head `Conv2d(num_of_timesteps, pred_len, kernel_size=(1, out_channels))`

Local adaptation:
  - The original paper uses 35 Beijing stations and six input features
    (including PM2.5). Here `input_dim` remains configurable, and the outer
    data adapter fills it with the features available in each dataset
    (12 for CN cities, 11 for Delhi in the legacy setting).
  - Input keeps the official shape `(B, N, F, T)`.
  - The official code converts inverse-distance adjacency into a scaled
    Laplacian plus Chebyshev polynomials. Here those quantities are prepared
    once in the data adapter from haversine distances.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# components/embed.py
# ---------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, F)
        return self.pe[:, : x.size(2)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.token_conv = nn.Conv2d(in_channels=c_in, out_channels=d_model, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, F) -> permute to (B, F, T, N)
        return self.token_conv(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, F)
        pos = self.position_embedding(x)
        x = self.value_embedding(x) + pos
        return F.relu(x)


# ---------------------------------------------------------------------------
# components/transformer.py  (multi-station self-attention)
# ---------------------------------------------------------------------------
class Transform(nn.Module):
    def __init__(self, outfea: int, d: int):
        super().__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)
        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)
        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea),
        )
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(x)

        query = torch.cat(torch.split(query, self.d, -1), 0)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 1, 3, 2)
        value = torch.cat(torch.split(value, self.d, -1), 0)

        A = torch.matmul(query, key) / (self.d ** 0.5)
        A = torch.softmax(A, -1)
        value = torch.matmul(A, value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1)
        value = value + x

        value = self.ln(value)
        x = self.ff(value)
        return self.lnff(x)


# ---------------------------------------------------------------------------
# Individual Temporal Pattern Module (ITPM)
# ---------------------------------------------------------------------------
class ITPM(nn.Module):
    def __init__(self, in_channels: int, d_model: int, out_channels: int):
        super().__init__()
        self.enc_embedding = DataEmbedding(in_channels, d_model)
        self.transformer = Transform(d_model, d_model)
        self.linear = nn.Conv2d(d_model, out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F, T)  -> enc expects (B, N, T, F)
        x_embed = self.enc_embedding(x.permute(0, 1, 3, 2))
        x_trans = self.transformer(x_embed)
        x_trans = F.relu(self.linear(x_trans.permute(0, 3, 2, 1)).permute(0, 3, 1, 2))
        # output: (B, N, out_channels, T)
        return x_trans


# ---------------------------------------------------------------------------
# Global Spatio-Temporal Dependence Module (GSTDM)
# ---------------------------------------------------------------------------
class TemporalAttentionLayer(nn.Module):
    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super().__init__()
        self.U1 = nn.Parameter(torch.randn(num_of_vertices))
        self.U2 = nn.Parameter(torch.randn(in_channels, num_of_vertices))
        self.U3 = nn.Parameter(torch.randn(in_channels))
        self.be = nn.Parameter(torch.randn(1, num_of_timesteps, num_of_timesteps))
        self.Ve = nn.Parameter(torch.randn(num_of_timesteps, num_of_timesteps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F, T)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        rhs = torch.matmul(self.U3, x)
        product = torch.matmul(lhs, rhs)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        return F.softmax(E, dim=1)


class SpatialAttentionLayer(nn.Module):
    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(num_of_timesteps))
        self.W2 = nn.Parameter(torch.randn(in_channels, num_of_timesteps))
        self.W3 = nn.Parameter(torch.randn(in_channels))
        self.bs = nn.Parameter(torch.randn(1, num_of_vertices, num_of_vertices))
        self.Vs = nn.Parameter(torch.randn(num_of_vertices, num_of_vertices))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F, T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        product = torch.matmul(lhs, rhs)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        return F.softmax(S, dim=1)


class GSTDM(nn.Module):
    def __init__(self, in_channels: int, num_nodes: int, num_of_timesteps: int):
        super().__init__()
        self.temporal_attn = TemporalAttentionLayer(in_channels, num_nodes, num_of_timesteps)
        self.spatial_attn = SpatialAttentionLayer(in_channels, num_nodes, num_of_timesteps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, F_in, T = x.shape
        T_attn = self.temporal_attn(x)  # (B, T, T)
        x_tat = torch.matmul(x.reshape(B, -1, T), T_attn).reshape(B, N, F_in, T)
        STDG = self.spatial_attn(x_tat)  # (B, N, N)
        return STDG


# ---------------------------------------------------------------------------
# Local Feature Module (STDG + Chebyshev graph conv + GRU)
# ---------------------------------------------------------------------------
class MSTGAN_GRUCell(nn.Module):
    def __init__(self, node_num: int, hidden_dim: int):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.linear1(combined))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, r * state), dim=-1)
        hc = torch.tanh(self.linear2(candidate))
        return z * state + (1 - z) * hc


class STDG_CGRU(nn.Module):
    def __init__(self, K: int, num_nodes: int, in_channels: int, out_channels: int, num_of_timesteps: int):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = out_channels
        self.num_nodes = num_nodes
        self.theta = nn.ParameterList(
            [nn.Parameter(torch.randn(in_channels, out_channels)) for _ in range(K)]
        )
        for p in self.theta:
            nn.init.xavier_uniform_(p)
        self.gru_cell = MSTGAN_GRUCell(num_nodes, self.hidden_dim)
        self.conv = nn.Conv2d(1, num_of_timesteps, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, STDG: torch.Tensor, cheb_polynomials: list[torch.Tensor]) -> torch.Tensor:
        # x: (B, N, F_in, T), STDG: (B, N, N)
        B, N, _, T = x.shape
        state = x.new_zeros(B, N, self.hidden_dim)
        inner_states = []
        for t in range(T):
            graph_signal = x[:, :, :, t]  # (B, N, F_in)
            output = x.new_zeros(B, N, self.out_channels)
            for k in range(self.K):
                T_k = cheb_polynomials[k]  # (N, N)
                T_k_with_stdg = T_k.mul(STDG)  # (B, N, N)
                rhs = T_k_with_stdg.permute(0, 2, 1).matmul(graph_signal)
                output = output + rhs.matmul(self.theta[k])
            state = self.gru_cell(output, state)
            inner_states.append(state)
        current_inputs = torch.stack(inner_states, dim=-1)  # (B, N, H, T)
        # The official implementation takes only the last hidden step and then
        # broadcasts it back to T steps through Conv2d(1, T, ...).
        out = self.conv(current_inputs[:, :, :, -1:].permute(0, 3, 1, 2))  # (B, T, N, H)
        return out  # (B, T, N, H)


# ---------------------------------------------------------------------------
# MST block
# ---------------------------------------------------------------------------
class MSTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        num_of_timesteps: int,
        K: int,
        dropout: float,
        d_model: int,
    ):
        super().__init__()
        self.itpm = ITPM(in_channels, d_model, out_channels)
        self.gstdm = GSTDM(out_channels, num_nodes, num_of_timesteps)
        self.stdg_cgru = STDG_CGRU(K, num_nodes, in_channels, out_channels, num_of_timesteps)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cheb_polynomials: list[torch.Tensor]) -> torch.Tensor:
        # x: (B, N, F, T)
        x_itpm = self.itpm(x)                       # (B, N, F_out, T)
        STDG = self.gstdm(x_itpm)                   # (B, N, N)
        ST_out = self.stdg_cgru(x, STDG, cheb_polynomials)  # (B, T, N, H)

        # residual path: (B, N, F, T) -> (B, F, N, T)
        x_res = self.residual_conv(x.permute(0, 2, 1, 3))   # (B, F_out, N, T)
        merged = F.relu(x_res + ST_out.permute(0, 3, 2, 1))  # (B, F_out, N, T)
        merged = self.ln(merged.permute(0, 3, 2, 1))         # (B, T, N, F_out)
        merged = self.dropout(merged).permute(0, 2, 3, 1)    # (B, N, F_out, T)
        return merged


# ---------------------------------------------------------------------------
# MSTGAN top-level module
# ---------------------------------------------------------------------------
class MSTGAN(nn.Module):
    """
    Multi-Spatio-Temporal Attention Network for air quality forecasting.

    The repository names the class MSTAN; the paper / doc label it MSTGAN.
    We keep the MSTGAN name here for consistency with project documentation.

    Forward signature:
        x:                (B, N, F, T_in)
        cheb_polynomials: list of (N, N) Chebyshev polynomial tensors T_0..T_{K-1}
    Returns:
        y_hat: (B, N, pred_len)  — univariate PM2.5 forecast
    """

    def __init__(
        self,
        input_dim: int,
        block1_hidden: int,
        block2_hidden: int,
        num_nodes: int,
        num_of_timesteps: int,
        pred_len: int,
        K: int,
        dropout: float,
        d_model: int,
        output_dim: int = 1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.blocks = nn.ModuleList(
            [
                MSTBlock(input_dim, block1_hidden, num_nodes, num_of_timesteps, K, dropout, d_model),
                MSTBlock(block1_hidden, block2_hidden, num_nodes, num_of_timesteps, K, dropout, d_model),
            ]
        )
        # Official predict_layer: Conv2d(num_of_timesteps, pred_len, kernel_size=(1, out_channels))
        self.predict_layer = nn.Conv2d(num_of_timesteps, pred_len * output_dim, kernel_size=(1, block2_hidden))

    def forward(self, x: torch.Tensor, cheb_polynomials: list[torch.Tensor]) -> torch.Tensor:
        # x: (B, N, F, T)
        for block in self.blocks:
            x = block(x, cheb_polynomials)
        # x: (B, N, F_out, T)  ->  Conv2d on (B, T, N, F_out) over "T" axis
        out = self.predict_layer(x.permute(0, 3, 1, 2))  # (B, pred_len*od, N, 1)
        out = out.squeeze(-1)                            # (B, pred_len*od, N)
        if self.output_dim == 1:
            return out.permute(0, 2, 1).contiguous()     # (B, N, pred_len)
        out = out.view(x.size(0), self.pred_len, self.output_dim, x.size(1))
        return out.permute(0, 3, 2, 1).contiguous()      # (B, N, od, pred_len)

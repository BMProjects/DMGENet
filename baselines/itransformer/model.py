"""
baselines/itransformer/model.py
================================
iTransformer local adaptation, kept as close as possible to the official implementation:
  - Official repository: https://github.com/thuml/iTransformer
  - Key files: model/iTransformer.py, layers/Embed.py, layers/Transformer_EncDec.py,
             layers/SelfAttention_Family.py

Local adaptation:
  - Main input sequence is multi-station PM2.5: x_enc shape (B, T, N)
  - Time covariates use x_mark_enc shape (B, T, 4) = [month, day, weekday, hour]
  - Output is unified as (B, N, pred_len)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DataEmbeddingInverted(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, N, T)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], dim=1))
        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, mask_flag: bool = False, scale: float | None = None, attention_dropout: float = 0.1):
        super().__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int, d_keys: int | None = None, d_values: int | None = None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, d_ff: int | None = None, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = torch.relu if activation == "relu" else torch.nn.functional.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers: list[nn.Module], norm_layer: nn.Module | None = None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class iTransformer(nn.Module):
    def __init__(
        self,
        n_variates: int,
        seq_len: int = 72,
        pred_len: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_norm: bool = True,
        freq: str = "h",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_variates = n_variates
        self.use_norm = use_norm
        self.freq = freq

        # x_mark has 4 hourly calendar covariates
        self.enc_embedding = DataEmbeddingInverted(seq_len, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(n_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        """
        x_enc:      (B, T, N)
        x_mark_enc: (B, T, 4)
        returns:    (B, N, pred_len)
        """
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, : self.n_variates]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out.permute(0, 2, 1).contiguous()

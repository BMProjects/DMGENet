import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tcn import TemporalConvNet
from models.gat import GAT
from models.gcn import GCNLayer

class PredictionLayer(nn.Module):
    """Shrink the temporal dimension (T_in -> T_out), then collapse channels to 1."""

    def __init__(self, T_in, T_out, hidden_size):
        super(PredictionLayer, self).__init__()
        self.conv1 = nn.Conv2d(T_in, T_out, 1)
        self.conv2 = nn.Conv2d(hidden_size, 1, 1)

    def forward(self, input_prediction_layer):
        """
        :param input_prediction_layer: [B, T_in, N, D]
        :return: [B, N, T_out]
        """
        out = self.conv1(input_prediction_layer)   # [B, T_out, N, D]
        out = out.permute(0, 3, 2, 1)              # [B, D, N, T_out]
        out = self.conv2(out)                      # [B, 1, N, T_out]
        out = out.squeeze(1)
        return out

class gated_TCN(nn.Module):
    def __init__(self, input_size, num_channels):
        super(gated_TCN, self).__init__()
        self.num_channels = num_channels
        self.TCN1 = TemporalConvNet(input_size, num_channels)
        self.TCN2 = TemporalConvNet(input_size, num_channels)

    def forward(self, x):
        """
        :param x:(batch_size, seq_len, num_nodes, features)
        :return: (batch_size, seq_len, num_nodes, num_channels[-1])
        """
        x = x.transpose(1, 2)
        B, N, t, d = x.shape
        x = x.reshape(B * N, t, d)
        x = x.transpose(1, 2)
        TCN1_output = torch.tanh(self.TCN1(x))
        TCN1_output = TCN1_output.transpose(1, 2).reshape(B, N, t, self.num_channels[-1])
        TCN2_output = torch.sigmoid(self.TCN2(x))
        TCN2_output = TCN2_output.transpose(1, 2).reshape(B, N, t, self.num_channels[-1])
        output = (TCN1_output * TCN2_output).transpose(1, 2)
        return output


class SpitalBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, adj, dropout, alpha, n_heads, gcn_bool, gat_bool):
        super(SpitalBlock, self).__init__()
        self.gcn_bool = gcn_bool
        self.gat_bool = gat_bool
        self.adj = adj
        self.gcn = GCNLayer(in_channels, hidden_size, hidden_size, adj, bias=True)
        self.gat = GAT(in_channels, hidden_size, dropout, alpha, n_heads)

        self.f_gcn = nn.Linear(hidden_size, hidden_size)
        self.f_gat = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        if self.gcn_bool and self.gat_bool:
            out_gcn = self.gcn(x)
            out_gat = self.gat(x, self.adj)
            gate = torch.sigmoid(self.f_gcn(out_gcn) + self.f_gat(out_gat))
            out = gate * out_gcn + (1 - gate) * out_gat
        elif self.gcn_bool:
            out = self.gcn(x)
        elif self.gat_bool:
            out = self.gat(x, self.adj)
        return out


class ComputeAttentionScore(nn.Module):
    def __init__(self):
        super(ComputeAttentionScore, self).__init__()

    def forward(self, x, node_vec):
        n_q = node_vec.unsqueeze(dim=-1)
        x_t_a = torch.einsum('btnd,ndl->btnl', (x, n_q)).contiguous()
        return x_t_a


class SpatioTemporal_block(nn.Module):
    def __init__(self, in_channels, hidden_size, num_channels, dropout, alpha, n_heads, num_nodes, apt_size, adj, gated_TCN_bool, gcn_bool, gat_bool, ASTAM_bool):
        super(SpatioTemporal_block, self).__init__()
        self.gated_TCN_bool = gated_TCN_bool
        self.ASTAM_bool = ASTAM_bool
        # Lift input channels into the model's hidden dimension.
        self.start_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=(1, 1))
        # Temporal branch.
        self.gated_TCN = gated_TCN(hidden_size, num_channels)
        # Spatial branch (GCN + GAT gated mixture).
        self.spital_block = SpitalBlock(hidden_size, hidden_size, adj, dropout, alpha, n_heads, gcn_bool, gat_bool)
        # Adaptive spatio-temporal attention module (ASTAM): per-node
        # embeddings drive per-node query projections for each axis.
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, apt_size), requires_grad=True)
        self.ComputeAttentionScore = ComputeAttentionScore()
        self.w_s = (nn.Conv2d(in_channels=apt_size, out_channels=hidden_size, kernel_size=(1, 1)))
        self.w_t = (nn.Conv2d(in_channels=apt_size, out_channels=hidden_size, kernel_size=(1, 1)))

    def forward(self, x):
        """
        :param x: (batch_size, num_nodes, seq_len, in_dim)
        :return: (batch_size, num_nodes, seq_len. hidden_size)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.start_conv(x).permute(0, 3, 2, 1)
        if self.gated_TCN_bool:
            x_t = self.gated_TCN(x)
        else:
            x_t = x
        x_s = self.spital_block(x_t)
        if self.ASTAM_bool:
            # Per-node query for the temporal axis, derived from the node embedding.
            n_q_t = self.w_t(self.node_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze(-1).squeeze(-1)
            x_t_a = self.ComputeAttentionScore(x_t, n_q_t)

            # Per-node query for the spatial axis.
            n_q_s = self.w_s(self.node_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze(-1).squeeze(-1)
            x_s_a = self.ComputeAttentionScore(x_s, n_q_s)

            # Softmax over the two axes to produce node-level adaptive weights.
            x_a = torch.cat((x_t_a, x_s_a), -1)
            x_att = F.softmax(x_a, dim=-1)

            x = x_att[:, :, :, 0].unsqueeze(dim=-1) * x_t + x_att[:, :, :, 1].unsqueeze(dim=-1) * x_s
            x = x.transpose(1, 2)
        else:
            x = x_t.transpose(1, 2) + x_s.transpose(1, 2)
        return x


class Model(nn.Module):
    def __init__(self, adj, input_size, hidden_size, dropout, alpha, n_heads, num_channels, apt_size, num_nodes, num_block, T_in, predict_len, gated_TCN_bool, gcn_bool, gat_bool, ASTAM_bool):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.predict_len = predict_len
        self.Blocklist = nn.ModuleList([
            SpatioTemporal_block(in_channels=input_size, hidden_size=hidden_size, num_channels=num_channels, dropout=dropout, alpha=alpha, n_heads=n_heads, num_nodes=num_nodes, apt_size=apt_size, adj=adj,
                                             gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            if i == 0 else
            SpatioTemporal_block(in_channels=hidden_size, hidden_size=hidden_size, num_channels=num_channels, dropout=dropout, alpha=alpha, n_heads=n_heads, num_nodes=num_nodes, apt_size=apt_size, adj=adj,
                                             gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            for i in range(num_block)
        ])
        self.prediction_layer = PredictionLayer(T_in, T_out=predict_len, hidden_size=hidden_size)

    def forward(self, x):
        """
        :param x: (batch_size, num_nodes, seq_len, in_dim)
        :return:  (batch_size, num_nodes, T_out)
        """
        for i in range(len(self.Blocklist)):
            x = self.Blocklist[i](x)
        x = x.permute(0, 2, 1, 3)  # prediction layer expects [B, T, N, D]
        x = self.prediction_layer(x)
        return x
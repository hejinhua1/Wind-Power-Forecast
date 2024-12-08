import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian

class GCNConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_nodes, **kwargs):
        super(GCNConv, self).__init__()
        adj_mat = torch.ones(num_nodes, num_nodes)
        adj_mat[torch.arange(num_nodes), torch.arange(num_nodes)] = 0
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj_mat))
        )
        self._num_nodes = num_nodes
        self._input_dim = input_dim  # seq_len for prediction
        self._output_dim = output_dim  # hidden_dim for prediction
        self.weights = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, inputs):
        # (batch_size, num_nodes, seq_len)
        batch_size = inputs.shape[0]
        inputs = inputs.transpose(0, 1)
        # (num_nodes, batch_size, seq_len)
        inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
        # (num_nodes, batch_size * seq_len)


        # AX (num_nodes, batch_size * seq_len)
        ax = self.laplacian @ inputs
        # (num_nodes, batch_size, seq_len)
        ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        # (num_nodes * batch_size, seq_len)
        ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        # act(AXW) (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.weights)
        # (num_nodes, batch_size, output_dim)
        outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        # (batch_size, num_nodes, output_dim)
        outputs = outputs.transpose(0, 1)
        return outputs


class Model(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.gcn_layers = configs.gcn_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # BatchNorm layers for each GCN layer

        # 第一层
        self.convs.append(GCNConv(configs.num_node_features * configs.timestep_max, configs.num_node_features * configs.timestep_max, configs.num_nodes))
        self.bns.append(torch.nn.BatchNorm1d(configs.num_node_features * configs.timestep_max))

        # 后续层
        for _ in range(configs.gcn_layers - 1):
            self.convs.append(GCNConv(configs.num_node_features * configs.timestep_max, configs.num_node_features * configs.timestep_max, configs.num_nodes))
            self.bns.append(torch.nn.BatchNorm1d(configs.num_node_features * configs.timestep_max))

        # dropout
        self.dropout = nn.Dropout(configs.dropout)
        # 输出层
        self.out_conv = GCNConv(configs.num_node_features * configs.timestep_max, configs.timestep_max, configs.num_nodes)

    def forward(self, x, adj, adj_hat):
        x = x.permute(0, 2, 1, 3).flatten(2, 3)  # [B, C, N, T] -> [B, N, C*T]
        residual = x  # 残差连接的起始值

        for i in range(self.gcn_layers):
            # GCN layer + BatchNorm + ReLU
            x = self.convs[i](x)    # [B, N, C*T]
            x = x.permute(0, 2, 1)  # [B, N, C*T] -> [B, C*T, N]
            x = self.bns[i](x)
            x = x.permute(0, 2, 1)  # [B, C*T, N] -> [B, N, C*T]
            x = F.relu(x)

            # 残差连接
            if i < self.gcn_layers - 1:  # 最后一层不需要残差连接
                x = x + residual  # 添加残差连接
                residual = x  # 更新残差值

            x = self.dropout(x)

        # 输出层
        x = self.out_conv(x)    # [B, N, T]

        return x


if __name__ == '__main__':
    class Configs:
        def __init__(self):
            self.num_node_features = 6
            self.hidden_channels = 16
            self.timestep_max = 96
            self.gcn_layers = 3
            self.dropout = 0.1
            self.num_nodes = 9
    args = Configs()
    model = Model(args)
    x = torch.randn(2, 6, 9, 96)
    adj = torch.randn(2, 9, 9)
    adj_hat = torch.randn(2, 9, 9)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                [1, 2, 3, 4, 5, 6, 7, 8, 0]])
    out = model(x, adj, adj_hat)
    print(out.shape)
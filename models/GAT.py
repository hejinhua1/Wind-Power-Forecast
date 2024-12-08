import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, num_nodes, adj_mat, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        self.adj_mat = adj_mat

    def forward(self, h):
        device = h.device
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(self.adj_mat.to(device) > 0, e, zero_vec.to(device))
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class Model(nn.Module):
    def __init__(self, configs):
        """Dense version of GAT."""
        super(Model, self).__init__()

        adj_mat = torch.ones(configs.num_nodes, configs.num_nodes)
        adj_mat[torch.arange(configs.num_nodes), torch.arange(configs.num_nodes)] = 0
        self.adj_mat = adj_mat

        self.attentions = [GraphAttentionLayer(configs.num_node_features * configs.timestep_max, configs.hidden_channels, configs.num_nodes, adj_mat, concat=True) for _ in range(configs.n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(configs.hidden_channels * configs.n_heads, configs.timestep_max, configs.num_nodes, adj_mat, concat=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, adj, adj_hat):
        assert x.size(0)==1, 'Batch size should be 1'
        x = x.squeeze()
        x = x.permute(1, 0, 2).flatten(1, 2)  # [C, N, T] -> [N, C*T]
        x = self.dropout(x)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        x = self.dropout(x)
        x = self.out_att(x)
        return x


if __name__ == '__main__':
    # simple test
    class Configs:
        def __init__(self):
            self.num_node_features = 6
            self.hidden_channels = 16
            self.timestep_max = 96
            self.gcn_layers = 3
            self.dropout = 0.1
            self.num_nodes = 9
            self.n_heads = 8
    args = Configs()
    model = Model(args)
    x = torch.randn(9, 96*6)
    adj = torch.randn(9, 9)
    adj_hat = torch.randn(9, 9)
    output = model(x, adj, adj_hat)
    print(output)
    print(output.size())
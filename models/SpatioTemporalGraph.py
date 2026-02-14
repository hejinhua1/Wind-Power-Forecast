import logging
from pathlib import Path
import torch
import numpy as np







class GraphAttention(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        timestep_max: int,
        time_filter: int,
        show_scores: bool = False,
    ):
        """
        @params
        in_channels: number of input channels (C)
        out_channels: number of output channels (C')
        timestep_max: number of time steps (T)
        time_filter: size of the time filter (t)

        @Inputs
        x: tensor of shape (B, C, N, T)
        a: tensor of shape (B, N, N)
        a_hat: tensor of shape (B, N, N)

        @Outputs
        out: tensor of shape (B, C', N, T')
        """
        super().__init__()

        self.show_scores = show_scores

        # Keys, values and queries
        self.k = torch.nn.Conv2d(
            in_channels=in_channels,  # C
            out_channels=timestep_max,  # T'
            kernel_size=(1, time_filter),  # 1, t
            stride=1,
        )
        self.q = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=timestep_max,
            kernel_size=(1, time_filter),
            stride=1,
        )
        self.v = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=timestep_max,
            kernel_size=(1, time_filter),
            stride=1,
        )

        # Additional layers
        self.fc_res = torch.nn.Conv2d(
            in_channels=in_channels,  # C
            out_channels=out_channels,  # C'
            kernel_size=(1, 1),
            stride=1,
        )
        self.fc_out = torch.nn.Conv2d(
            in_channels=(timestep_max - time_filter) + 1,
            out_channels=out_channels,  # C'
            kernel_size=(1, 1),
            stride=1,
        )

        # Activation, Normalization and Dropout
        self.act = torch.nn.Softmax(dim=-1)
        self.norm = torch.nn.BatchNorm2d(out_channels)
        self.dropout = torch.nn.Dropout()

        # To remove
        self.score_style = True

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, adj_hat: torch.Tensor
    ) -> torch.Tensor:
        k = self.k(x)  # B, T', N, (T - t + 1)
        q = self.q(x)  # B, T', N, (T - t + 1)
        v = self.v(x)  # B, T', N, (T - t + 1)

        score = torch.einsum("BTNC, BTnC -> BTNn", k, q).contiguous()  # B, T', N, N

        if adj is not None:
            score = score + adj.unsqueeze(
                1
            )  # Do not use += operator since it mess-up the gradian calculation
        if adj_hat is not None:
            score = score + adj_hat.unsqueeze(
                1
            )  # Do not use += operator since it mess-up the gradian calculation

        score = self.act(score)
        if self.score_style:
            out = torch.einsum(
                "BTnN, BTNC -> BCnT", score, v
            ).contiguous()  # B, (T - t + 1), N, T'
        else:
            out = torch.einsum(
                "BTNC, BTNn -> BCnT", v, score
            ).contiguous()  # B, (T - t + 1), N, T'

        out = self.fc_out(out)  # B, C', N, T'

        res = self.fc_res(x)  # B, C', N, T'

        out = self.norm((out + res))  # B, C', N, T'

        out = self.dropout(out)  # B, C', N, T'

        if self.show_scores:
            return out, score
        return out  # B, C', N, T'


class MultiHeadGraphAttention(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        timestep_max: int,
        show_scores: bool = False,
    ):
        """
        @params
        in_channels: number of input channels (C)
        out_channels: number of output channels (C')
        timestep_max: number of time steps (T)

        @Inputs
        x: tensor of shape (B, C, N, T)
        a: tensor of shape (B, N, N)
        a_hat: tensor of shape (B, N, N)

        @Outputs
        out: tensor of shape (B, C', N, T')
        """
        super().__init__()

        # heads
        self.ga_2 = GraphAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            timestep_max=timestep_max,
            time_filter=2,
            show_scores=show_scores,
        )
        self.ga_3 = GraphAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            timestep_max=timestep_max,
            time_filter=3,
            show_scores=show_scores,
        )
        self.ga_6 = GraphAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            timestep_max=timestep_max,
            time_filter=6,
            show_scores=show_scores,
        )
        self.ga_7 = GraphAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            timestep_max=timestep_max,
            time_filter=7,
            show_scores=show_scores,
        )

        # additional layers
        self.fc_res = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
        )
        self.fc_out = torch.nn.Conv2d(
            in_channels=out_channels * 4,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
        )

        # Normalization and dropout
        self.norm = torch.nn.BatchNorm2d(out_channels)
        self.dropout = torch.nn.Dropout()

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, adj_hat: torch.Tensor
    ) -> torch.Tensor:
        res = self.fc_res(x)  # B, C', N, T

        x = torch.cat(
            [
                self.ga_2(x, adj, adj_hat),
                self.ga_3(x, adj, adj_hat),
                self.ga_6(x, adj, adj_hat),
                self.ga_7(x, adj, adj_hat),
            ],
            dim=1,
        )  # B, 4*C', N, T'

        x = self.fc_out(x)  # B, C', N, T'

        x = self.norm((x + res))  # B, C', N, T'

        x = self.dropout(x)  # B, C', N, T'

        return x  # B, C', N, T'


class Model(torch.nn.Module):
    def __init__(self, configs):
        """
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.channels_last = configs.channels_last
        self.fc_in = torch.nn.Conv2d(
            in_channels=configs.in_channels,
            out_channels=configs.hidden_channels,
            kernel_size=(1, 1),
            stride=1,
        )
        self.blocks = torch.nn.ModuleList()
        for i in range(configs.nb_blocks):
            self.blocks.append(
                MultiHeadGraphAttention(
                    in_channels=configs.hidden_channels * (2**i),
                    out_channels=configs.hidden_channels * (2 ** (i + 1)),
                    timestep_max=configs.timestep_max,
                    show_scores=configs.show_scores,
                )
            )
        self.fc_out = torch.nn.Conv2d(
            in_channels=configs.hidden_channels * (2**configs.nb_blocks),
            out_channels=configs.out_channels,
            kernel_size=(1, 1),
            stride=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        adj_hat: torch.Tensor,
    ) -> torch.Tensor:

        if self.channels_last:
            x = x.transpose(3, 1)  # B, T, N, C -> B, C', N, T

        x = self.fc_in(x)  # B, C', N, T

        for block in self.blocks:
            x = block(x, adj, adj_hat)  # B, C', N, T'


        x = self.fc_out(x)  # B, C', N, T'


        if self.channels_last:
            x = x.transpose(3, 1)  # B, C', N, T' -> B, T', N, C'

        x = x.squeeze() # B, C', N, T' -> B, N, T'
        # 在最终输出前添加Sigmoid，确保输出在0-1范围内
        # x = torch.sigmoid(x)
        # x = torch.tanh(x) * 0.5 + 0.5  # 映射到[0,1]
        # x = torch.clamp(x, 0, 1)
        return x



if __name__ == "__main__":

    # Assuming your model and classes are already imported

    # 1. **Prepare Synthetic Data**
    batch_size = 32
    num_nodes = 9
    num_features = 5
    time_steps = 96

    # Random node feature tensor: [batch_size, num_features, num_nodes, time_steps]
    x = torch.rand(batch_size, num_features, num_nodes, time_steps)

    # Random index tensor (this might represent time-related information)
    # idx = torch.randint(0, 10, (batch_size, 2, time_steps))  # [batch_size, 2, time_steps]

    # Random adjacency matrix for the graph: [batch_size, num_nodes, num_nodes]
    adj = torch.rand(batch_size, num_nodes, num_nodes)

    # Random adjacency hat matrix (could be estimated causality)
    adj_hat = torch.rand(batch_size, num_nodes, num_nodes)

    # Random degree and node_id (node features)
    degrees = torch.randint(1, 5, (num_nodes,))  # Random degrees for nodes
    node_ids = torch.arange(num_nodes)  # Node IDs: 0, 1, 2, 3, 4
    edge_index = torch.randint(0, num_nodes, (2, 10))  # Random edge index

    # 2. **Initialize Model**
    embedding_dict = {
        "time": time_steps,
        "day": 7,
        "node": num_nodes,
        "degree": num_nodes
    }


    class Config:
        def __init__(self):
            self.in_channels = num_features
            self.hidden_channels = 16
            self.out_channels = 1
            self.timestep_max = time_steps
            self.nb_blocks = 2
            self.channels_last = False
            self.show_scores = False
            self.task_name = 'synthetic'
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96


    args = Config()
    # Initialize the model with random parameters
    model = Model(args)

    # 3. **Run the Model**
    output = model(x, adj, adj_hat)

    # 4. **Inspect Outputs**
    print("Output shape:", output.shape)  # Should be [batch_size, num_nodes, time_steps']

    # Optionally, print the output for inspection
    print("Output Tensor:", output)


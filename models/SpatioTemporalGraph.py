import logging
from pathlib import Path
import torch
import numpy as np





class BaseModel(torch.nn.Module):
    def __init__(
        self,
        embedding_dict: dict,
        channels_last: bool = True,
        name: str = "",
        degrees: None = None,
        use_super_node: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.name = name
        self.channels_last = channels_last
        self.logger = logging.getLogger(name)
        self.embedding_dict = embedding_dict
        self.degrees = (
            None
            if degrees is None
            else torch.tensor(degrees, dtype=torch.int, device=device)
        )
        self.node_ids = (
            None
            if self.embedding_dict.get("node", None) is None
            else torch.arange(
                self.embedding_dict["node"], dtype=torch.int, device=device
            )
        )


    def load(self, path: Path = Path.cwd() / "logs" / "model_weights.pth"):
        if path.is_file():
            self.load_state_dict(torch.load(path))
            self.logger.info("Model succesfully loaded")

    def save(self, path: Path = Path.cwd() / "logs" / "model_weights.pth"):
        torch.save(self.state_dict(), path)
        self.logger.info("Model succesfully saved")

    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_onnx(self):
        pass





class GraphAttention(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        timestep_max: int,
        time_filter: int,
        device: str,
        show_scores: bool = False,
    ):
        """
        @params
        in_channels: number of input channels (C)
        out_channels: number of output channels (C')
        timestep_max: number of time steps (T)
        time_filter: size of the time filter (t)
        device: torch.Device (cpu, cuda, tpu)

        @Inputs
        x: tensor of shape (B, C, N, T) [Traffic Data]
        a: tensor of shape (B, N, N) [Distance Information]
        a_hat: tensor of shape (B, N, N) [Estimated Causality Information]

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
            device=device,
        )
        self.q = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=timestep_max,
            kernel_size=(1, time_filter),
            stride=1,
            device=device,
        )
        self.v = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=timestep_max,
            kernel_size=(1, time_filter),
            stride=1,
            device=device,
        )

        # Additional layers
        self.fc_res = torch.nn.Conv2d(
            in_channels=in_channels,  # C
            out_channels=out_channels,  # C'
            kernel_size=(1, 1),
            stride=1,
            device=device,
        )
        self.fc_out = torch.nn.Conv2d(
            in_channels=(timestep_max - time_filter) + 1,
            out_channels=out_channels,  # C'
            kernel_size=(1, 1),
            stride=1,
            device=device,
        )

        # Activation, Normalization and Dropout
        self.act = torch.nn.Softmax(dim=-1)
        self.norm = torch.nn.BatchNorm2d(out_channels, device=device)
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
        device: str,
        show_scores: bool = False,
    ):
        """
        @params
        in_channels: number of input channels (C)
        out_channels: number of output channels (C')
        timestep_max: number of time steps (T)
        device: torch.Device (cpu, cuda, tpu)

        @Inputs
        x: tensor of shape (B, C, N, T) [Traffic Data]
        a: tensor of shape (B, N, N) [Distance Information]
        a_hat: tensor of shape (B, N, N) [Estimated Causality Information]

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
            device=device,
            show_scores=show_scores,
        )
        self.ga_3 = GraphAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            timestep_max=timestep_max,
            time_filter=3,
            device=device,
            show_scores=show_scores,
        )
        self.ga_6 = GraphAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            timestep_max=timestep_max,
            time_filter=6,
            device=device,
            show_scores=show_scores,
        )
        self.ga_7 = GraphAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            timestep_max=timestep_max,
            time_filter=7,
            device=device,
            show_scores=show_scores,
        )

        # additional layers
        self.fc_res = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            device=device,
        )
        self.fc_out = torch.nn.Conv2d(
            in_channels=out_channels * 4,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            device=device,
        )

        # Normalization and dropout
        self.norm = torch.nn.BatchNorm2d(out_channels, device=device)
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


class Model(BaseModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        timestep_max: int,
        embedding_dict: dict,
        nb_blocks: int = 1,
        channels_last: bool = True,
        use_super_node: bool = True,
        degrees: None = None,
        name="STGM_FULL",
        device: str = "cpu",
        show_scores: bool = False,
        *args,
        **kwargs
    ):
        """
        @params
        in_channels: number of input channels (C)
        out_channels: number of output channels (C')
        timestep_max: number of time steps (T)
        nb_blocks: number of internal blocks
        channels_last: if the channels are situated in the last dim
        device: torch.Device (cpu, cuda, tpu)

        @Inputs
        x: tensor of shape (B, T, N, C) [Traffic Data]
        a: tensor of shape (B, N, N) [Distance Information]
        a_hat: tensor of shape (B, N, N) [Estimated Causality Information]

        @Outputs
        out: tensor of shape (B, T', N, C')
        """
        super().__init__(
            name=name,
            channels_last=channels_last,
            degrees=degrees,
            use_super_node=use_super_node,
            embedding_dict=embedding_dict,
            device=device,
        )
        self.fc_in = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
            stride=1,
            device=device,
        )
        self.blocks = torch.nn.ModuleList()
        for i in range(nb_blocks):
            self.blocks.append(
                MultiHeadGraphAttention(
                    in_channels=hidden_channels * (2**i),
                    out_channels=hidden_channels * (2 ** (i + 1)),
                    timestep_max=timestep_max,
                    device=device,
                    show_scores=show_scores,
                )
            )
        self.fc_out = torch.nn.Conv2d(
            in_channels=hidden_channels * (2**nb_blocks),
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        adj: torch.Tensor,
        adj_hat: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        self.logger.debug(
            "[INPUTS] x: %s, idx: %s, adj: %s, adj_hat: %s",
            x.shape,
            idx.shape,
            adj.shape if adj is not None else None,
            adj_hat.shape if adj_hat is not None else None,
        )

        if self.channels_last:
            x = x.transpose(3, 1)  # B, T, N, C -> B, C', N, T
            self.logger.debug("[IN_TRANSPOSE] x: %s", x.shape)

        x = self.fc_in(x)  # B, C', N, T
        self.logger.debug("[FC_IN] x: %s", x.shape)

        for block in self.blocks:
            x = block(x, adj, adj_hat)  # B, C', N, T'
            self.logger.debug("[BLOCK] x: %s", x.shape)

        x = self.fc_out(x)  # B, C', N, T'
        self.logger.debug("[FC_OUT] x: %s", x.shape)

        if self.channels_last:
            x = x.transpose(3, 1)  # B, C', N, T' -> B, T', N, C'
            self.logger.debug("[OUT_TRANSPOSE] x: %s", x.shape)

        self.logger.debug("[OUT] x: %s", x.shape)
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
    idx = torch.randint(0, 10, (batch_size, 2, time_steps))  # [batch_size, 2, time_steps]

    # Random adjacency matrix for the graph: [batch_size, num_nodes, num_nodes]
    adj = torch.rand(batch_size, num_nodes, num_nodes)

    # Random adjacency hat matrix (could be estimated causality)
    adj_hat = torch.rand(batch_size, num_nodes, num_nodes)

    # Random degree and node_id (node features)
    degrees = torch.randint(1, 5, (num_nodes,))  # Random degrees for nodes
    node_ids = torch.arange(num_nodes)  # Node IDs: 0, 1, 2, 3, 4

    # 2. **Initialize Model**
    embedding_dict = {
        "time": time_steps,
        "day": 7,
        "node": num_nodes,
        "degree": num_nodes
    }

    # Initialize the model with random parameters
    model = Model(
        in_channels=num_features,
        hidden_channels=16,
        out_channels=9,
        timestep_max=time_steps,
        embedding_dict=embedding_dict,
        nb_blocks=2,  # Number of Graph Attention Blocks
        channels_last=False,
        use_super_node=False,
        device='cpu',
    )

    # 3. **Run the Model**
    output = model(x, idx, adj, adj_hat)

    # 4. **Inspect Outputs**
    print("Output shape:", output.shape)  # Should be [batch_size, time_steps', num_nodes, out_channels]

    # Optionally, print the output for inspection
    print("Output Tensor:", output)


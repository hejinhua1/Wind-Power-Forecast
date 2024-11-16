import logging
from pathlib import Path
import torch
import numpy as np


def super_node_pre_hook(_module, args, kwargs):
    # print("super_node_pre_hook", args, kwargs)
    if len(args) > 0:
        args[0] = torch.nn.functional.pad(args[0], (0, 0, 0, 1), value=0)
        for i in range(2, 2 + len(args[2:])):
            args[i] = torch.nn.functional.pad(args[i], (0, 1, 0, 1), value=1)
    if isinstance(kwargs.get("x", None), torch.Tensor):
        kwargs["x"] = torch.nn.functional.pad(kwargs["x"], (0, 0, 0, 1), value=0)
    if isinstance(kwargs.get("adj", None), torch.Tensor):
        kwargs["adj"] = torch.nn.functional.pad(kwargs["adj"], (0, 1, 0, 1), value=1)
    if isinstance(kwargs.get("adj_hat", None), torch.Tensor):
        kwargs["adj_hat"] = torch.nn.functional.pad(
            kwargs["adj_hat"], (0, 1, 0, 1), value=1
        )
    return args, kwargs


def super_node_post_hook(_module, _args, _kwargs, x: torch.Tensor):
    if len(x.shape) > 3:
        return x[..., :-1, :]
    else:
        return x[..., :-1, :-1]


class Embedding(torch.nn.Module):
    def __init__(
        self, embedding_dim: int, embeddings: dict[str, int | None], device: str = "cpu"
    ):
        """
        @params
        embedding_dim: embedding dimension (temporal (T))
        embeddings: dict disribing each embedding depth (keys: time, day, node, degree)
        device: torch.Device (cpu, cuda, tpu)

        @Inputs
        x: tensor of shape (B, C, N, T)
        idx: tensor of shape (B, 2, T)

        @Outputs
        out: tensor of shape (B, C', N, T')
        """
        super().__init__()
        self.logger = logging.getLogger("Embedding Module")
        self.embeddings = embeddings
        if self.embeddings.get("time", None) is not None:
            self.time_embedding = torch.nn.Embedding(
                self.embeddings["time"] + 1, embedding_dim=embedding_dim, device=device
            )
        if self.embeddings.get("day", None) is not None:
            self.day_embedding = torch.nn.Embedding(
                self.embeddings["day"] + 1, embedding_dim=embedding_dim, device=device
            )
        if self.embeddings.get("node", None) is not None:
            self.node_embedding = torch.nn.Embedding(
                self.embeddings["node"] + 1, embedding_dim=embedding_dim, device=device
            )
        if self.embeddings.get("degree", None) is not None:
            self.degree_embedding = torch.nn.Embedding(
                self.embeddings["degree"] + 1,
                embedding_dim=embedding_dim,
                device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        node_ids: torch.Tensor | None = None,
        degrees: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logging.debug(
            "x: %s, idx: %s, node_ids: %s, degrees: %s",
            x.shape,
            idx.shape,
            None if node_ids is None else node_ids.shape,
            None if degrees is None else degrees.shape,
        )
        if self.embeddings.get("time", None) is not None:
            x += self.time_embedding(idx[:, 0]).transpose(1, 2).unsqueeze(2)
            logging.debug("x: %s", x.shape)
        if self.embeddings.get("day", None) is not None:
            x += self.day_embedding(idx[:, 1]).transpose(1, 2).unsqueeze(2)
            logging.debug("x: %s", x.shape)
        if not (self.embeddings.get("node", None) is None or node_ids is None):
            x += (
                self.node_embedding(node_ids)
                .transpose(0, 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(x.shape[0], 1, 1, x.shape[-1])
            )
            logging.debug("x: %s", x.shape)
        if not (self.embeddings.get("degree", None) is None or degrees is None):
            x += (
                self.degree_embedding(degrees)
                .transpose(0, 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(x.shape[0], 1, 1, x.shape[-1])
            )
            logging.debug("x: %s", x.shape)
        return x


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        embedding_dict: dict[str, int | None],
        channels_last: bool = True,
        name: str = "",
        degrees: np.ndarray | None = None,
        use_super_node: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.name = name
        self.channels_last = channels_last
        self.logger = logging.getLogger(name)
        self.embedding_dict = embedding_dict
        self.hook_handlers = []
        if use_super_node:
            self.hook_handlers.append(
                self.register_forward_pre_hook(super_node_pre_hook, with_kwargs=True)
            )
            self.hook_handlers.append(
                self.register_forward_hook(super_node_post_hook, with_kwargs=True)
            )
            self.embedding_dict.get("node", None)
            if self.embedding_dict.get("node", None) is not None:
                self.embedding_dict["node"] += 1
            if self.embedding_dict.get("degree", None) is not None:
                self.embedding_dict["degree"] = self.embedding_dict.get("node", None)
            if degrees is not None:
                degrees += 1
                degrees = np.concatenate(
                    (degrees, np.array((self.embedding_dict["node"],))), axis=-1
                )

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

    def remove_super_node_hooks(self):
        for handle in self.hook_handlers:
            handle.remove()

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
        self, x: torch.Tensor, adj: torch.Tensor | None, adj_hat: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        self, x: torch.Tensor, adj: torch.Tensor | None, adj_hat: torch.Tensor | None
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
        embedding_dict: dict[str, int | None],
        nb_blocks: int = 1,
        channels_last: bool = True,
        use_super_node: bool = True,
        degrees: np.ndarray | None = None,
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

        self.embedding = Embedding(
            embedding_dim=hidden_channels,
            embeddings=self.embedding_dict,
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
        adj: torch.Tensor | None = None,
        adj_hat: torch.Tensor | None = None,
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

        x = self.embedding(
            x, idx, degrees=self.degrees, node_ids=self.node_ids
        )  # B, C', N, T
        self.logger.debug("[EMBEDDING] x: %s", x.shape)

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
    x = torch.rand(2, 12, 207, 2)
    idx = torch.randint(0, 207, (2, 207))
    adj = torch.rand(2, 207, 207)
    adj_hat = torch.rand(2, 207, 207)
    model = Model( 2, 64, 2, 12, {"node": 207}, nb_blocks=1)
    out = model(x, idx, adj, adj_hat)
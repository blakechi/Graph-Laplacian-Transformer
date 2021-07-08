from typing import List
from einops.einops import rearrange

import torch
from torch import nn, einsum
try:
    from typing_extensions import Final
except:
    from torch.jit import Final

from .layer_scale import LayerScale
from .base_block import MLP
from .helpers import name_with_msg


class GraphClassAttention(nn.Module):
    heads: Final[int]
    scale: Final[float]

    def __init__(
        self,
        dim: int,
        heads: int,
        use_bias: float = False,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
    ) -> None:
        super().__init__()

        head_dim = dim // heads

        assert (
            head_dim * heads == dim
        ), name_with_msg(self, f"`head_dim` ({head_dim}) * `heads` ({heads}) != `dim` ({dim})")

        self.heads = heads

        self.Q = nn.Linear(dim, dim, bias=use_bias)
        self.K = nn.Linear(dim, dim, bias=use_bias)
        self.V = nn.Linear(dim, dim, bias=use_bias)
        self.out_linear = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

        self.scale = head_dim ** (-0.5)

    def forward(self, cls_tokens: torch.Tensor, x: torch.Tensor, graph_portion: List[int]) -> torch.Tensor:
        b, _ = cls_tokens.shape
        z = torch.cat([cls_tokens, x], dim=0)  # (p d), p = n + b
        
        q, k, v = self.Q(cls_tokens), self.K(z), self.V(z)
        q = rearrange(q, "b (h d) -> h b d", h=self.heads)
        k = rearrange(k, "p (h d) -> h p d", h=self.heads)*self.scale
        v = rearrange(v, "p (h d) -> h p d", h=self.heads)
        
        k_cls, k_list = k[:, : b, :], k[:, b:, :].split(graph_portion, dim=1)
        v_cls, v_list = v[:, : b, :], v[:, b:, :].split(graph_portion, dim=1)
        
        outs: List[torch.Tensor] = []
        for idx in range(len(k_list)):
            q_graph = q[:, idx, :].unsqueeze(dim=1)
            k_graph = torch.cat([k_cls[:, idx, :].unsqueeze(dim=1), k_list[idx]], dim=0)
            v_graph = torch.cat([v_cls[:, idx, :].unsqueeze(dim=1), v_list[idx]], dim=0)
            attention = einsum("h 1 d, h p d -> h 1 p", q_graph, k_graph).softmax(dim=-1)
            attention = self.attention_dropout(attention)
            out = einsum("h 1 p, h p d, -> h 1 d", attention, v_graph)
            out = rearrange(out, "h 1 d -> 1 (h d)")
            outs.append(out)

        outs = torch.cat(outs, dim=0)
        outs = self.out_linear(outs)
        outs = self.out_dropout(outs)

        return outs
        

class GraphClassAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        alpha: float,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        path_dropout: float = 0.,
        **kwargs
    ):
        super().__init__()

        self.attn_block = LayerScale(
            dim=dim,
            alpha=alpha,
            core_block=GraphClassAttention,
            ff_dropout=ff_dropout,
            path_dropout=path_dropout,
            **kwargs
        )

        self.ff_block = LayerScale(
            dim=dim,
            alpha=alpha,
            core_block=MLP,
            expand_dim=dim*ff_expand_scale,
            ff_dropout=ff_dropout,
            path_dropout=path_dropout,
        )

    def forward(self, cls_tokens, x, graph_portion):
        out = self.attn_block(cls_tokens, x, graph_portion)
        out = self.ff_block(out)

        return out
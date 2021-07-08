import torch
from torch import nn

from .layer_scale import LayerScale
from .base_block import MLP
from .multihead_attention import MultiheadAttention

class ClassAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        use_bias: float = False,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.attn = MultiheadAttention(
            dim,
            heads=heads,
            use_bias=use_bias,
            attention_dropout=attention_dropout,
            ff_dropout=ff_dropout
        )

    def forward(self, cls_token, x) -> torch.Tensor:
        z = torch.cat([cls_token, x], dim=1)

        return self.attn((cls_token, z, z))
        

class ClassAttentionLayer(nn.Module):
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
            core_block=ClassAttention,
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

    def forward(self, cls_token, x):
        out = self.attn_block(cls_token, x)
        out = self.ff_block(out)

        return out
from typing import List

import torch
from torch import nn, einsum
try:
    from typing_extensions import Final
except:
    from torch.jit import Final
from einops import rearrange, repeat

from src.utils import LayerScale, MLP, ClassAttentionLayer, TokenDropout, ProjectionHead
from src.utils.helpers import name_with_msg, config_pop_argument


class GraphLaplacianAttention(nn.Module):
    heads: Final[int]
    scale: Final[float]
    mask_value: Final[float]

    def __init__(
        self,
        dim: int,
        heads: int,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
        use_bias: bool = False,
    ) -> None:

        head_dim = dim // heads

        assert (
            head_dim * heads == dim
        ), name_with_msg(self, f"`head_dim` ({head_dim}) * `heads` ({heads}) != `dim` ({dim})")

        self.heads = heads

        self.QKV = nn.Linear(dim, 3*dim, bias=use_bias)
        self.edge_KV = nn.Linear(dim, 2*dim, bias=use_bias)
        self.out_linear = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

        self.scale = head_dim ** (-0.5)
        self.mask_value = -torch.finfo(torch.float32).max()

    def forward(self, x: torch.Tensor, edges: torch.Tensor, attentino_mask: torch.Tensor) -> torch.Tensor:
        _, n, _ = x.shape

        q, k, v = self.QKV(x).chunk(chunks=3, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d")
        k = rearrange(k, "b n (h d) -> b h n d")*self.scale
        v = rearrange(v, "b n (h d) -> b h n d")

        edge_k, edge_v = self.QKV(edges).chunk(chunks=2, dim=-1)
        edge_k = rearrange(edge_k, "b n (h d) -> b h n d")*self.scale
        edge_v = rearrange(edge_v, "b n (h d) -> b h n d")

        attention = einsum("b h n d, b h m d -> b h n m", q, k + edge_k)
        attention = attention.masked_fill_(attentino_mask, self.mask_value)
        attention = attention.softmax(dim=-1)
        
        attention_degree = repeat(torch.eye(n), "n m -> 1 h n m", h=self.heads)
        attention = attention_degree - attention  # D - A
        attention = self.attention_dropout(attention)

        v = v + edge_v
        out = einsum("b h n m, b h m d -> b h n d", attention, v)
        out = rearrange("b h n d -> b n (h d)")
        out = self.out_linear(out)
        out = self.out_dropout(out)

        return out


class GraphLaplacianTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        alpha: float,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        path_dropout: float = 0.,
        **kwargs
    ) -> None:
        super().__init__()

        self.attn_block = LayerScale(
            core_block=GraphLaplacianAttention,
            dim=dim,
            alpha=alpha,
            ff_dropout=ff_dropout,
            path_dropout=path_dropout,
            **kwargs
        )

        self.ff_block = LayerScale(
            core_block=MLP,
            dim=dim,
            alpha=alpha,
            expand_dim=dim*ff_expand_scale,
            ff_dropout=ff_dropout,
            path_dropout=path_dropout,
        )

    def forward(self, x: torch.Tensor, edges: torch.Tensor, attentino_mask: torch.Tensor) -> torch.Tensor:
        x = self.attn_block(x, edges, attentino_mask)
        x = self.ff_block(x)

        return x


class GraphLaplacianTransformerBackbone(nn.Module):
    dim: Final[int]

    def __init__(
        self,
        num_token_layer: int,
        num_cls_layer: int,
        dim: int,
        heads: int,
        alpha: float,
        use_bias: bool = False,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,
        token_dropout: float = 0.,
    ) -> None:
        
        self.dim = dim
        
        self.token_dropout = TokenDropout(token_dropout)
        self.token_layers = nn.ModuleList([
            GraphLaplacianTransformerLayer(
                dim=dim,
                heads=heads,
                alpha=alpha,
                use_bias=use_bias,
                ff_expand_scale=ff_expand_scale,
                ff_dropout=ff_dropout,
                attention_dropout=attention_dropout,
                path_dropout=path_dropout,
            ) for _ in range(num_token_layer)
        ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self.cls_layers = nn.ModuleList([
            ClassAttentionLayer(
                dim=dim,
                heads=heads,
                alpha=alpha,
                use_bias=use_bias,
                ff_expand_scale=ff_expand_scale,
                ff_dropout=ff_dropout,
                attention_dropout=attention_dropout,
                path_dropout=path_dropout,
            ) for _ in range(num_cls_layer)
        ])

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]

        x = self.token_dropout(x)
        for layer in self.token_layers:
            x = layer(x, edges)

        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        for layer in self.cls_layers:
            cls_token = layer(cls_token, x)

        return cls_token

    def parse_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(x)

    def parse_edges(self, x: torch.Tensor) -> torch.Tensor:
        return self.edge_embedding(x)

    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            # Reference from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L110
            nn.init.trunc_normal_(m.weight, std=0.02)

        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.uniform_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        # embedding as well
        return ["cls_token", "LayerNorm.weight", "AffineTransform.alpha"]  
    

class GraphLaplacianTransformerWithLinearClassifier(GraphLaplacianTransformerBackbone):
    def __init__(self, config = None) -> None:
        num_classes = config_pop_argument(config, "num_classes")
        pred_act_fnc_name = config_pop_argument(config, "pred_act_fnc_name")
        super().__init__(**config.__dict__)

        self.proj_head = ProjectionHead(
            self.dim,
            num_classes,
            pred_act_fnc_name,
        )

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        x = super().forward(x, edges)
        x = rearrange(x, "b 1 d -> b d")

        return self.proj_head(x)
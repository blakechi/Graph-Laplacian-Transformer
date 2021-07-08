from typing import List, Optional

import torch
from torch import nn, einsum
from torch._C import dtype
try:
    from typing_extensions import Final
except:
    from torch.jit import Final
from einops import rearrange, repeat

from src.utils import LayerScale, MLP, ClassAttentionLayer, TokenDropout, ProjectionHead
from src.utils.helpers import name_with_msg, config_pop_argument
from .config import GraphLaplacianTransformerConfig


# class GraphLaplacianAttention(nn.Module):
#     heads: Final[int]
#     expanded_heads: Final[int]
#     scale: Final[float]
#     mask_value: Final[float]

#     def __init__(
#         self,
#         dim: int,
#         heads: int,
#         head_expand_scale: float = 1.,
#         attention_dropout: float = 0.,
#         ff_dropout: float = 0.,
#         use_bias: bool = False,
#         use_edge_bias: bool = False,
#         use_attn_expand_bias: bool = False,
#     ) -> None:
#         super().__init__()

#         head_dim = dim // heads

#         assert (
#             head_dim * heads == dim
#         ), name_with_msg(self, f"`head_dim` ({head_dim}) * `heads` ({heads}) != `dim` ({dim})")

#         self.heads = heads
#         self.expanded_heads = int(head_expand_scale*self.heads)  # ceiling

#         self.QK = nn.Linear(dim, 2*dim, bias=use_bias)
#         self.V = nn.Linear(dim, self.expanded_heads*head_dim, bias=use_bias)
#         self.edge_K = nn.Linear(dim, dim, bias=use_edge_bias)
#         self.edge_V = nn.Linear(dim, self.expanded_heads*head_dim, bias=use_edge_bias)
#         self.depth_wise_conv = nn.Conv2d(
#             self.heads,
#             self.expanded_heads,
#             kernel_size=1,
#             groups=self.heads,
#             bias=use_attn_expand_bias
#         )
#         self.out_linear = nn.Linear(self.expanded_heads*head_dim, dim)

#         self.attention_dropout = nn.Dropout(attention_dropout)
#         self.out_dropout = nn.Dropout(ff_dropout)

#         self.scale = head_dim ** (-0.5)
#         self.mask_value = -torch.finfo(torch.float32).max

#     def forward(self, x: torch.Tensor, edges: torch.Tensor, attentino_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         _, n, _ = x.shape

#         #
#         q, k = self.QK(x).chunk(chunks=2, dim=-1)
#         v = self.V(x)
#         q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
#         k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)*self.scale
#         v = rearrange(v, "b n (h d) -> b h n d", h=self.expanded_heads)

#         edge_k = self.edge_K(edges)
#         edge_v = self.edge_V(edges)
#         edge_k = rearrange(edge_k, "b n (h d) -> b h n d", h=self.heads)*self.scale
#         edge_v = rearrange(edge_v, "b n (h d) -> b h n d", h=self.expanded_heads)

#         #
#         attention = einsum("b h n d, b h m d -> b h n m", q, k + edge_k)
#         attention = self.depth_wise_conv(attention)
        
#         if attentino_mask is not None:
#             attention = attention.masked_fill_(attentino_mask, self.mask_value)

#         attention = attention.softmax(dim=-1)
        
#         attention_degree = repeat(torch.eye(n), "n m -> 1 h n m", h=self.expanded_heads)
#         attention = attention_degree - attention  # D - A
#         attention = self.attention_dropout(attention)

#         #
#         v = v + edge_v
#         out = einsum("b h n m, b h m d -> b h n d", attention, v)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         out = self.out_linear(out)
#         out = self.out_dropout(out)

#         return out


class GraphLaplacianAttention(nn.Module):
    heads: Final[int]
    expanded_heads: Final[int]
    scale: Final[float]
    mask_value: Final[float]

    def __init__(
        self,
        dim: int,
        heads: int,
        head_expand_scale: float = 1.,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
        use_bias: bool = False,
        use_edge_bias: bool = False,
        use_attn_expand_bias: bool = False,
    ) -> None:
        super().__init__()

        head_dim = dim // heads

        assert (
            head_dim * heads == dim
        ), name_with_msg(self, f"`head_dim` ({head_dim}) * `heads` ({heads}) != `dim` ({dim})")

        self.heads = heads
        self.expanded_heads = int(head_expand_scale*self.heads)  # ceiling

        self.Q = nn.Linear(dim, dim, bias=use_bias)
        self.K = nn.Linear(dim, dim, bias=use_bias)
        self.V = nn.Linear(dim, self.expanded_heads*head_dim, bias=use_bias)
        self.edge_K = nn.Linear(dim, dim, bias=use_edge_bias)
        self.edge_V = nn.Linear(dim, self.expanded_heads*head_dim, bias=use_edge_bias)
        self.attn_expand_proj = nn.Linear(
            self.heads,
            self.expanded_heads,
            bias=use_attn_expand_bias
        )
        self.out_linear = nn.Linear(self.expanded_heads*head_dim, dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

        self.scale = head_dim ** (-0.5)
        self.mask_value = -torch.finfo(torch.float32).max

    def forward(self, x: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n, e = x.shape

        #
        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = rearrange(q, "n (h d) -> h n d", h=self.heads)
        k = rearrange(k, "n (h d) -> h n d", h=self.heads)*self.scale
        v = rearrange(v, "n (h d) -> h n d", h=self.expanded_heads)

        edge_k, edge_v = self.edge_K(edges), self.edge_V(edges)
        edge_k = rearrange(edge_k, "n (h d) -> h n d", h=self.heads)*self.scale
        edge_v = rearrange(edge_v, "n (h d) -> h n d", h=self.expanded_heads)

        # all edge pairs
        q = q[: edge_index[0], :]
        k = k[: edge_index[1], :]
        # element-wsie attention
        element_wise_attn = einsum("h p d, h p d -> h p", q, k)
        # scatter to attention map
        attention = torch.full(
            (self.heads, n, n),
            fill_value=self.mask_value,
            requires_grad=True,
            device=x.device,
            dtype=x.dtype
        ).scatter_(src=element_wise_attn)
        
        attention = rearrange(attention, "h n m -> n m h")
        attention = self.attn_expand_proj(attention)
        attention = rearrange(attention, "n m h -> h n m")
        attention = attention.softmax(dim=-1)
        
        attention_degree = repeat(torch.eye(n), "n m -> h n m", h=self.expanded_heads)
        attention = attention_degree - attention  # D - A
        attention = self.attention_dropout(attention)

        #
        v = v + edge_v
        out = einsum("h n m, h m d -> h n d", attention, v)
        out = rearrange(out, "h n d -> n (h d)")
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

    def forward(self, x: torch.Tensor, edges: torch.Tensor, attentino_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        use_edge_bias: bool = False,
        use_attn_expand_bias: bool = False,
        head_expand_scale: float = 1.,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,
        token_dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.dim = dim
        
        self.token_dropout = TokenDropout(token_dropout)
        self.token_layers = nn.ModuleList([
            GraphLaplacianTransformerLayer(
                dim=dim,
                heads=heads,
                alpha=alpha,
                use_bias=use_bias,
                use_edge_bias=use_edge_bias,
                use_attn_expand_bias=use_attn_expand_bias,
                head_expand_scale=head_expand_scale,
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
    def __init__(self, config: GraphLaplacianTransformerConfig = None) -> None:
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
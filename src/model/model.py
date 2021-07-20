from typing import List, OrderedDict

import torch
from torch import nn, einsum
try:
    from typing_extensions import Final
except:
    from torch.jit import Final
from torch_geometric.utils import softmax as sparse_softmax
from torch_scatter import scatter
from einops import rearrange, repeat
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from src.utils import LayerScale, MLP, ProjectionHead
from src.utils.helpers import name_with_msg, config_pop_argument
from .config import GraphLaplacianTransformerConfig


class GraphLaplacianAttention(nn.Module):
    heads: Final[int]
    expanded_heads: Final[int]
    scale: Final[float]

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
        self.attn_expand_proj = nn.Linear(self.heads, self.expanded_heads, bias=use_attn_expand_bias)
        self.out_linear = nn.Linear(self.expanded_heads*head_dim, dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

        self.scale = head_dim ** (-0.5)
        
    def forward(self, x: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n, _ = x.shape
        e, _ = edges.shape

        #
        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = rearrange(q, "n (h d) -> h n d", h=self.heads)
        k = rearrange(k, "n (h d) -> h n d", h=self.heads)
        v_non_scatter = rearrange(v, "n (h d) -> h n d", h=self.expanded_heads)
        # all edge pairs
        q = q.index_select(dim=1, index=edge_index[0])  # q[:, edge_index[0], :] (h e d)
        k = k.index_select(dim=1, index=edge_index[1])  # k[:, edge_index[1], :] (h e d)
        v = v_non_scatter.index_select(dim=1, index=edge_index[1])  # v[:, edge_index[1], :] (h e d)

        edge_k, edge_v = self.edge_K(edges), self.edge_V(edges)
        edge_k = rearrange(edge_k, "e (h d) -> h e d", h=self.heads)
        edge_v = rearrange(edge_v, "e (h d) -> h e d", h=self.expanded_heads)

        #
        attention = einsum("h e d, h e d -> h e", q, (k + edge_k)*self.scale)  # element-wsie attention
        attention = rearrange(attention, "h e -> e h")
        attention = self.attn_expand_proj(attention)  # expand attention heads
        attention_list = attention.split(1, dim=1)
        # softmax
        attention_list = [
            sparse_softmax(
                attention_list[head_idx],
                index=edge_index[0],
                num_nodes=n
            ) for head_idx in range(self.expanded_heads)
        ]
        attention = torch.cat(attention_list, dim=1)
        attention = rearrange(attention, "e h -> h e")
        attention = self.attention_dropout(attention)  

        #
        out = einsum("h e, h e d -> h e d", attention, v + edge_v)
        out_list = out.split(1, dim=0)
        out_list = [
            scatter(
                out_list[head_idx],
                index=edge_index[0],
                dim=1,
                reduce="sum"
            ) for head_idx in range(self.expanded_heads)
        ]
        out = torch.cat(out_list, dim=0)
        out = v_non_scatter - out  # D - A
        out = rearrange(out, "h n d -> n (h d)")
        out = self.out_linear(out)
        out = self.out_dropout(out)

        return out

    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        return [] 


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

    def forward(self, cls_tokens: torch.Tensor, x: torch.Tensor, graph_portion: torch.Tensor) -> torch.Tensor:
        b, _ = cls_tokens.shape
        z = torch.cat([cls_tokens, x], dim=0)  # (p d), p = n + b
        
        q, k, v = self.Q(cls_tokens), self.K(z), self.V(z)
        q = rearrange(q, "b (h d) -> h b d", h=self.heads)
        k = rearrange(k, "p (h d) -> h p d", h=self.heads)*self.scale
        v = rearrange(v, "p (h d) -> h p d", h=self.heads)
        
        k_cls, k_list = k[:, : b, :], k[:, b:, :].split(graph_portion.tolist(), dim=1)
        v_cls, v_list = v[:, : b, :], v[:, b:, :].split(graph_portion.tolist(), dim=1)
        
        outs: List[torch.Tensor] = []
        for idx in range(len(k_list)):
            q_graph = q[:, idx, :].unsqueeze(dim=1)
            k_graph = torch.cat([k_cls[:, idx, :].unsqueeze(dim=1), k_list[idx]], dim=1)
            v_graph = torch.cat([v_cls[:, idx, :].unsqueeze(dim=1), v_list[idx]], dim=1)
            attention = einsum("h l d, h p d -> h l p", q_graph, k_graph).softmax(dim=-1)  # l == 1
            attention = self.attention_dropout(attention)
            out = einsum("h l p, h p d -> h l d", attention, v_graph)
            out = rearrange(out, "h 1 d -> 1 (h d)")
            outs.append(out)

        outs = torch.cat(outs, dim=0)
        outs = self.out_linear(outs)
        outs = self.out_dropout(outs)

        return outs

    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        return []  


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

        self.pre_norm_edge = nn.LayerNorm(dim)
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

    def forward(self, x: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.attn_block(x, self.pre_norm_edge(edges), edge_index)
        x = self.ff_block(x)
        
        return x
        

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

    def forward(self, cls_tokens: torch.Tensor, x: torch.Tensor, graph_portion: torch.Tensor) -> torch.Tensor:
        cls_tokens = self.attn_block(cls_tokens, x, graph_portion)
        cls_tokens = self.ff_block(cls_tokens)

        return cls_tokens


class GraphLaplacianTransformerBackbone(nn.Module):
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
    ) -> None:
        super().__init__()
        
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

        self.cls_token = nn.Parameter(torch.randn(1, dim), requires_grad=True)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.cls_layers = nn.ModuleList([
            GraphClassAttentionLayer(
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

    def forward(self, x: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor, graph_portion: torch.Tensor) -> torch.Tensor:
        b = graph_portion.shape[0]

        for layer in self.token_layers:
            x = layer(x, edges, edge_index)

        cls_tokens = repeat(self.cls_token, "1 d -> b d", b=b)
        for layer in self.cls_layers:
            cls_tokens = layer(cls_tokens, x, graph_portion)

        return cls_tokens

        # Mean pooling
        # graphs = x.split(graph_portion.tolist(), dim=0)
        # graphs = [graph.mean(dim=0, keepdim=True) for graph in graphs]

        # graphs = torch.cat(graphs, dim=0)
        
        # return graphs

    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        return ["cls_token"]  


class GraphLaplacianTransformerWithLinearClassifier(nn.Module):
    def __init__(self, config: GraphLaplacianTransformerConfig = None) -> None:
        num_classes = config_pop_argument(config, "num_classes")
        pred_act_fnc_name = config_pop_argument(config, "pred_act_fnc_name")
        grad_clip_value = config_pop_argument(config, "grad_clip_value")
        super().__init__()

        self.atom_embedding = AtomEncoder(config.dim)
        self.edge_embedding = BondEncoder(config.dim)
        self.edge_proj = nn.Sequential(OrderedDict([
            ("proj", nn.Linear(config.dim, config.dim)),
            ("norm", nn.LayerNorm(config.dim))
        ]))

        self.backbone = GraphLaplacianTransformerBackbone(**config.__dict__)
        self.proj_head = ProjectionHead(
            config.dim,
            num_classes,
            pred_act_fnc_name,
        )
        
        self.no_weight_decays = set()
        self.grad_clip_value = grad_clip_value
        self.apply(self._aggregate_no_weight_decays)
        self.apply(self._init_weights)
        self.apply(self._register_grad_clip)

    def forward(self, x: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor, graph_portion: torch.Tensor) -> torch.Tensor:
        x = self.atom_embedding(x)
        edges = self.edge_embedding(edges)
        edges = self.edge_proj(edges)

        x = self.backbone(x, edges, edge_index, graph_portion)

        return self.proj_head(x)

    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        return ["atom_embedding", "edge_embedding", "edge_proj.norm"]

    @torch.jit.ignore
    def _aggregate_no_weight_decays(self, m):
        if hasattr(m, "no_weight_decay") and callable(getattr(m, "no_weight_decay")):
            self.no_weight_decays |= set(m.no_weight_decay())
    
    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)

            if m.bias is not None:
                nn.init.uniform_(m.bias)

    @torch.jit.ignore
    def _register_grad_clip(self, m):
        m.register_full_backward_hook(lambda m, grad_in, grad_out: nn.utils.clip_grad_value_(m.parameters(), self.grad_clip_value))

    @torch.jit.ignore
    def num_parameters(self) -> int:
        return sum(torch.numel(params) for params in self.parameters())
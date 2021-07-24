from typing import List, OrderedDict, Set, Tuple, Dict

import torch
from torch import nn, einsum
try:
    from typing_extensions import Final
except:
    from torch.jit import Final
from torch_geometric.utils import softmax as sparse_softmax
from torch_scatter import scatter
from einops import rearrange, repeat

from src.dataset import AtomEncoder, BondEncoder
from src.utils import LayerScale, MLP, ProjectionHead
from src.utils.helpers import name_with_msg, config_pop_argument
from .config import GraphLaplacianTransformerConfig


class GraphEdgeFusionAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        edge_dim: int,
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
        self.V = nn.Linear(dim, dim, bias=use_bias)
        # self.edge_Q = nn.Linear(edge_dim, dim, bias=use_edge_bias)
        self.edge_K = nn.Linear(edge_dim, dim, bias=use_edge_bias)
        self.edge_V = nn.Linear(dim, dim, bias=use_edge_bias)
        self.attn_expand_proj = nn.Linear(self.heads, self.expanded_heads, bias=use_attn_expand_bias)
        self.attn_squeeze_proj = nn.Linear(self.expanded_heads, self.heads, bias=False)
        self.out_linear = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

        self.scale = head_dim ** (-0.5)
        
    def forward(self, x: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor]:
        n, _ = x.shape  # no batch size since graphs are coalessed into one
        # e, _ = edges.shape

        #
        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = rearrange(q, "n (h d) -> h n d", h=self.heads)
        k = rearrange(k, "n (h d) -> h n d", h=self.heads)
        v_non_scatter = rearrange(v, "n (h d) -> h n d", h=self.heads)
        # all edge pairs
        q = q.index_select(dim=1, index=edge_index[0])  # q[:, edge_index[0], :] (h e d)
        k = k.index_select(dim=1, index=edge_index[1])  # k[:, edge_index[1], :] (h e d)
        v = v_non_scatter.index_select(dim=1, index=edge_index[1])  # v[:, edge_index[1], :] (h e d)

        # edge_q = self.edge_Q(edges)
        edge_k = self.edge_K(edges)
        edge_v = self.edge_V(edges)
        # edge_q, edge_k, edge_v = self.edge_Q(edges), self.edge_K(edges), self.edge_V(edges)
        # edge_q = rearrange(edge_q, "e (h d) -> h e d", h=self.heads)
        edge_k = rearrange(edge_k, "e (h d) -> h e d", h=self.heads)
        edge_v = rearrange(edge_v, "e (h d) -> h e d", h=self.heads)

        #
        k = (k + edge_k)*self.scale
        attention = einsum("h e d, h e d -> h e", q, k)  # element-wsie attention
        attention = rearrange(attention, "h e -> e h")
        attention = self.attn_expand_proj(attention)
        attention = nn.functional.gelu(attention)
        attention = self.attn_squeeze_proj(attention)
        # softmax
        attention_list = attention.split(1, dim=1)
        attention = torch.cat([
            sparse_softmax(
                attention_list[head_idx],
                index=edge_index[0],
                dim=0,
                num_nodes=n
            ) for head_idx in range(self.heads)
        ], dim=1)
        attention = self.attention_dropout(attention)  
        attention = rearrange(attention, "e h -> h e")

        #
        v = v + edge_v
        out = einsum("h e, h e d -> h e d", attention, v)
        out_list = out.split(1, dim=0)
        out = torch.cat([
            scatter(
                out_list[head_idx],
                index=edge_index[0],
                dim=1,
                reduce="sum"
            ) for head_idx in range(self.heads)
        ], dim=0)
        out = rearrange(out, "h n d -> n (h d)")
        out = self.out_linear(out)
        out = self.out_dropout(out)

        return out, attention

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return set()


class GraphLaplacianAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int,
        head_expand_scale: float = 1.,
        attention_dropout: float = 0.,
        ff_dropout: float = 0.,
        use_bias: bool = False,
        use_attn_expand_bias: bool = False,
        dtype = torch.float32,
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
        self.V = nn.Linear(dim, dim, bias=use_bias)
        self.attn_expand_proj = nn.Conv2d(self.heads, self.expanded_heads, kernel_size=1, bias=use_attn_expand_bias)
        self.attn_squeeze_proj = nn.Conv2d(self.expanded_heads, self.heads, kernel_size=1, bias=False)
        self.out_linear = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out_dropout = nn.Dropout(ff_dropout)

        self.scale = head_dim ** (-0.5)
        self.mask_value = -torch.finfo(dtype).max
        
    def forward(self, x: torch.Tensor, graph_portion: List[int], graph_edge_index: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        b = len(graph_portion)
        n, _ = x.shape  # no batch size since graphs are coalessed into one

        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = rearrange(q, "p (h d) -> h p d", h=self.heads)
        k = rearrange(k, "p (h d) -> h p d", h=self.heads)*self.scale
        v = rearrange(v, "p (h d) -> h p d", h=self.heads)
        
        q_list = q.split(graph_portion, dim=1)
        k_list = k.split(graph_portion, dim=1)
        v_list = v.split(graph_portion, dim=1)
        
        outs: List[torch.Tensor] = []
        attentions: List[torch.Tensor] = []
        for idx in range(b):
            q_graph = q_list[idx]
            k_graph = k_list[idx]
            v_graph = v_list[idx]

            attention = einsum("h n d, h m d -> h n m", q_graph, k_graph)
            attention = self.attn_expand_proj(attention.unsqueeze(dim=0))  # (h, n, m) -> (1, h, n, m)
            attention = nn.functional.gelu(attention)
            attention = self.attn_squeeze_proj(attention).squeeze(dim=0)  # (1, h, n, m) -> (h, n, m)
            attention = attention.softmax(dim=-1)
            attention = self.attention_dropout(attention)

            out = einsum("h n m, h m d -> h n d", attention, v_graph)
            out = v_graph - out  # D - A
            out = rearrange(out, "h n d -> n (h d)")
            outs.append(out)

            edge_attention = attention[:, graph_edge_index[idx][0], graph_edge_index[idx][1]]
            attentions.append(edge_attention)

        attentions = torch.cat(attentions, dim=1)
        outs = torch.cat(outs, dim=0)
        outs = self.out_linear(outs)
        outs = self.out_dropout(outs)

        return outs, attentions

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return set()


class GraphClassAttention(nn.Module):

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
        for idx in range(b):
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
    def no_weight_decay(self) -> Set[str]:
        return set()


class GraphEdgeFusionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        **kwargs
    ) -> None:
        super().__init__()

        self.node_norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(dim)
        self.attn_block = GraphEdgeFusionAttention(
            dim=dim,
            **kwargs
        )

    def forward(self, x: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.node_norm(x)
        edges = self.node_norm(edges)
        x, attention = self.attn_block(x, edges, edge_index)

        return x, attention


class GraphLaplacianLayer(nn.Module):
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

    def forward(self, x: torch.Tensor, graph_portion: List[int], graph_edge_index: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        x, attention = self.attn_block(x, graph_portion, graph_edge_index)
        x, _ = self.ff_block(x)
        
        return x, attention
        

class GraphClassPoolLayer(nn.Module):
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

    def forward(self, cls_tokens: torch.Tensor, x: torch.Tensor, graph_portion: List[int]) -> torch.Tensor:
        cls_tokens, _ = self.attn_block(cls_tokens, x, graph_portion)
        cls_tokens, _ = self.ff_block(cls_tokens)

        return cls_tokens


class GraphLaplacianTransformerBackbone(nn.Module):
    def __init__(
        self,
        num_token_layer: int,
        num_cls_layer: int,
        dim: int,
        edge_dim: int,
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
        
        self.edge_fusion = GraphEdgeFusionLayer(
            dim=dim,
            edge_dim=edge_dim,
            heads=heads,
            # alpha=alpha,
            use_bias=use_bias,
            use_edge_bias=use_edge_bias,
            use_attn_expand_bias=use_attn_expand_bias,
            head_expand_scale=head_expand_scale,
            # ff_expand_scale=ff_expand_scale,
            ff_dropout=ff_dropout,
            attention_dropout=attention_dropout,
            # path_dropout=path_dropout,
        )

        self.token_layers = nn.ModuleList([
            GraphLaplacianLayer(
                dim=dim,
                heads=heads,
                alpha=alpha,
                use_bias=use_bias,
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
            GraphClassPoolLayer(
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

    def forward(self, x: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor, graph_portion: torch.Tensor, graph_edge_index: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        b = graph_portion.shape[0]
        graph_portion = graph_portion.tolist()
        attentions = []

        #
        x, attention = self.edge_fusion(x, edges, edge_index)
        attentions.append(attention)

        #
        for layer in self.token_layers:
            x, attention = layer(x, graph_portion, graph_edge_index)
            attentions.append(attention)

        #
        cls_tokens = repeat(self.cls_token, "1 d -> b d", b=b)
        for layer in self.cls_layers:
            cls_tokens = layer(cls_tokens, x, graph_portion)

        return cls_tokens, attentions

        # Mean pooling
        # graphs = x.split(graph_portion.tolist(), dim=0)
        # graphs = [graph.mean(dim=0, keepdim=True) for graph in graphs]

        # graphs = torch.cat(graphs, dim=0)
        
        # return graphs

    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        return { "cls_token" }


class GraphLaplacianTransformerWithLinearClassifier(nn.Module):
    def __init__(self, config: GraphLaplacianTransformerConfig = None) -> None:
        edge_dim = config_pop_argument(config, "edge_dim")
        num_classes = config_pop_argument(config, "num_classes")
        pred_act_fnc_name = config_pop_argument(config, "pred_act_fnc_name")
        grad_clip_value = config_pop_argument(config, "grad_clip_value")
        super().__init__()

        edge_dim = edge_dim if edge_dim is not None else config.dim // config.heads

        self.atom_embedding = AtomEncoder(config.dim)
        self.edge_embedding = BondEncoder(edge_dim)
        self.edge_proj = nn.Sequential(OrderedDict([
            ("proj", nn.Linear(edge_dim, edge_dim)),
            ("norm", nn.LayerNorm(edge_dim))
        ]))

        self.backbone = GraphLaplacianTransformerBackbone(**config.__dict__, edge_dim=edge_dim)
        self.proj_head = ProjectionHead(
            config.dim,
            num_classes,
            pred_act_fnc_name,
        )
        
        self.no_weight_decays = set()
        self.grad_clip_value = grad_clip_value
        
        self.apply(self._aggregate_no_weight_decays)
        self.apply(self._init_weights)
        if self.grad_clip_value is not None:
            self.apply(self._register_grad_clip)

    def forward(self, x: torch.Tensor, edges: torch.Tensor, edge_index: torch.Tensor, graph_portion: torch.Tensor, graph_edge_index: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        x = self.atom_embedding(x)
        edges = self.edge_embedding(edges)
        edges = self.edge_proj(edges)

        x, attentions = self.backbone(x, edges, edge_index, graph_portion, graph_edge_index)
        
        attention_kldiv_loss = self.get_attention_kldiv_loss(attentions, edge_index[0])

        return self.proj_head(x), attention_kldiv_loss

    def get_attention_kldiv_loss(self, attentions: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        num_heads = attentions[0].shape[0]
        kldiv_losses = []
        for idx in range(1, len(attentions)):
            log_p = torch.log(attentions[idx - 1] + 1e-12)
            log_q = torch.log(attentions[idx] + 1e-12)
            kldiv_loss = attentions[idx - 1]*(log_p - log_q)
            kldiv_loss_list = kldiv_loss.split(1, dim=0)  # split heads
            kldiv_loss = torch.cat([
                scatter(
                    kldiv_loss_list[head_idx],
                    index=edge_index,
                    dim=1,
                    reduce="sum"
                ) for head_idx in range(num_heads)
            ], dim=0)
            
            kldiv_losses.append(kldiv_loss.mean())

        return torch.stack(kldiv_losses, dim=0).mean()

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {"atom_embedding", "edge_embedding", "bias"}

    @torch.jit.ignore
    def _aggregate_no_weight_decays(self, m):
        r"""gather all `no_weight_decay` in the submodules"""
        
        if hasattr(m, "no_weight_decay") and callable(getattr(m, "no_weight_decay")):
            self.no_weight_decays |= set(m.no_weight_decay())
    
    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def _register_grad_clip(self, m):
        r"""register gradient clipping in the backward

        Warning: Won't work when scripting.
        """
        m.register_full_backward_hook(lambda m, grad_in, grad_out: nn.utils.clip_grad_value_(m.parameters(), self.grad_clip_value))

    @torch.jit.ignore
    def num_parameters(self) -> int:
        return sum(torch.numel(params) for params in self.parameters())
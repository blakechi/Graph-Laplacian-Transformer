class GraphLaplacianTransformerConfig(object):
    def __init__(
        self,
        num_token_layer: int,
        num_cls_layer: int,
        dim: int,
        heads: int,
        alpha: float,
        num_classes: int,
        use_bias: bool = False,
        use_edge_bias: bool = False,
        use_attn_expand_bias: bool = False,
        head_expand_scale: float = 1.,
        ff_expand_scale: int = 4,
        ff_dropout: float = 0.,
        attention_dropout: float = 0.,
        path_dropout: float = 0.,
        token_dropout: float = 0.,
        pred_act_fnc_name: str = "ReLU",
        grad_clip_value: float = 1e-2,
        **rest,
    ) -> None:
        super().__init__()

        self.num_token_layer = num_token_layer
        self.num_cls_layer = num_cls_layer
        self.dim = dim
        self.heads = heads
        self.alpha = alpha
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.use_edge_bias = use_edge_bias
        self.use_attn_expand_bias = use_attn_expand_bias
        self.head_expand_scale = head_expand_scale
        self.ff_expand_scale = ff_expand_scale
        self.ff_dropout = ff_dropout
        self.attention_dropout = attention_dropout
        self.path_dropout = path_dropout
        self.token_dropout = token_dropout
        self.pred_act_fnc_name = pred_act_fnc_name
        self.grad_clip_value = grad_clip_value
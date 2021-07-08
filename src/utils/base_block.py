from typing import Optional
from functools import partial

import torch
from torch import nn

from .helpers import name_with_msg, get_act_fnc, get_conv_layer


class FeedForward(nn.Module):
    r"""
    Feed-Forward Layer (Alias: MLP)
    Support 1x1 convolution for 1, 2, and 3D data
    """
    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        expand_dim: Optional[int] = None,
        ff_expand_scale: Optional[int] = None,
        ff_dropout: float = 0.0,
        act_fnc_name: str = "GELU",
        use_convXd: Optional[int] = None,
        **rest
    ) -> None:
        super().__init__()
        
        expand_dim = expand_dim or ff_expand_scale*dim if (expand_dim is not None) and (ff_expand_scale is not None) else dim
        out_dim = out_dim or dim

        if use_convXd:
            assert (
                0 < use_convXd and use_convXd < 4
            ), name_with_msg(f"`use_convXd` must be 1, 2, or 3 for valid `ConvXd` supported by PyTorch. But got: {use_convXd}")

            core = partial(get_conv_layer(f"Conv{use_convXd}d"), kernel_size=1)
        else:
            core = nn.Linear

        self.ff_0 = core(dim, expand_dim)
        self.act_fnc = get_act_fnc(act_fnc_name)()
        self.dropout = nn.Dropout(ff_dropout)
        self.ff_1 = core(expand_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_0(x)
        x = self.act_fnc(x)
        x = self.dropout(x)
        x = self.ff_1(x)

        return x

MLP = FeedForward


# Reference from: https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L642
class ProjectionHead(nn.Module):
    def __init__(self, dim, out_dim, act_fnc_name="ReLU", ff_dropout: float = 0.1):
        super().__init__()

        self.head = nn.Sequential(  
            nn.Linear(dim, dim),
            get_act_fnc(act_fnc_name)(),
            nn.Dropout(ff_dropout),
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        return self.head(x)
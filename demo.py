import torch

from src.model import GraphLaplacianTransformerWithLinearClassifier
from src.model import GraphLaplacianTransformerConfig


if __name__ == "__main__":

    glt_config = GraphLaplacianTransformerConfig(
        6,
        2,
        128,
        4,
        1e-4,
        2,
        head_expand_scale=2,
        ff_dropout=0.1,
        attention_dropout=0.1,
        path_dropout=0.1,
    )

    glt = GraphLaplacianTransformerWithLinearClassifier(glt_config)
    
    x = torch.rand(1, 32, 128)
    edges = torch.rand(1, 32, 128)

    print(glt)
    print(glt(x, edges).shape)
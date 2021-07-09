import torch
from torch_geometric.data import DataLoader
from torch_geometric.transforms import RemoveIsolatedNodes
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from src.model import GraphLaplacianTransformerWithLinearClassifier
from src.model import GraphLaplacianTransformerConfig


if __name__ == "__main__":

    glt_config = GraphLaplacianTransformerConfig(
        6,
        2,
        128,
        4,
        1.,
        128,
        use_bias=True,
        use_edge_bias=True,
        use_attn_expand_bias=True,
        head_expand_scale=2,
        ff_dropout=0.1,
        attention_dropout=0.1,
        path_dropout=0.05,
    )
    glt = GraphLaplacianTransformerWithLinearClassifier(glt_config)
    print(glt)
    print(glt.num_parameters())

    #
    atom_encoder = AtomEncoder(emb_dim = 128)
    bond_encoder = BondEncoder(emb_dim = 128)

    #
    batch_size = 32
    num_workers = 1
    pin_memory = True

    dataset = PygGraphPropPredDataset(name="ogbg-molpcba") 

    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    criterian = torch.nn.BCEWithLogitsLoss(reduction="sum")
    torch.autograd.set_detect_anomaly(True)
    for data in train_loader:
        x, edges, edge_index, batch, y = data.x, data.edge_attr, data.edge_index.to(torch.long), data.batch, data.y
        x = atom_encoder(x)
        edges = atom_encoder(edges)
        graph_portion = batch.bincount()
        mask = ~torch.isnan(y)

        out = glt(x, edges, edge_index, graph_portion)

        loss = criterian(out[mask], y[mask])
        loss.backward()

        for name, param in glt.named_parameters():
            if name.startswith("token_layers"):
                print(name, param.grad)

        assert False
        


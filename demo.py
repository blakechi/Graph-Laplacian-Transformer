import numpy as np
import torch
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from src.model import GraphLaplacianTransformerConfig, GraphLaplacianTransformerWithLinearClassifier


if __name__ == "__main__":
    dataset_name = "ogbg-molpcba"
    batch_size = 32
    num_workers = 1
    pin_memory = False

    #
    glt_config = GraphLaplacianTransformerConfig(
        6,
        2,
        128,
        4,
        1e-4,
        128,
        use_bias=False,
        use_edge_bias=False,
        use_attn_expand_bias=False,
        head_expand_scale=1,
        ff_dropout=0.1,
        attention_dropout=0.1,
        path_dropout=0.05,
    )
    glt = GraphLaplacianTransformerWithLinearClassifier(glt_config)
    # print(glt)

    #
    atom_encoder = AtomEncoder(emb_dim = 128)
    bond_encoder = BondEncoder(emb_dim = 128)
    evaluator = Evaluator(name=dataset_name)
    dataset = PygGraphPropPredDataset(name=dataset_name) 

    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    criterian = torch.nn.BCEWithLogitsLoss()

    torch.autograd.set_detect_anomaly(True)
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in train_loader:
            x, edges, edge_index, batch, y = data.x, data.edge_attr, data.edge_index.to(torch.long), data.batch, data.y.to(torch.float)
            graph_portion = batch.bincount()
            mask = ~torch.isnan(y)
            y = y.numpy()
            print(y)
            y[~mask] = np.nan
            print(y)
            assert False
            out = glt(x, edges, edge_index, graph_portion)

            loss = criterian(out[mask], y[mask])
            # loss.backward()

            # for name, param in glt.named_parameters():
            #     if "token_layers" in name:
            #         print(name, param.grad.norm())

            # eval_key = evaluator.eval_metric
            # evaluation_score = evaluator.eval({
            #     'y_pred': out,
            #     'y_true': y,
            # })[eval_key]
            # print("evaluation_score: ", evaluation_score)
            y_true.append(y)
            y_pred.append(out)

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    eval_key = evaluator.eval_metric
    evaluation_score = evaluator.eval({
        'y_pred': y_pred,
        'y_true': y_true,
    })[eval_key]
    print("evaluation_score: ", evaluation_score)
        


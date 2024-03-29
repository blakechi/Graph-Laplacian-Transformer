from tqdm import tqdm
import torch
from torch_geometric.data import DataLoader
# from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from src.model import GraphLaplacianTransformerConfig, GraphLaplacianTransformerWithLinearClassifier
from src.utils import RemoveIsolatedNodes


if __name__ == "__main__":
    torch.cuda.set_device(2)
    dataset_name = "ogbg-molhiv"
    batch_size = 32
    num_workers = 1
    pin_memory = False

    #
    config_path = "/media/storage0/pwchi/Graph_Laplacian_Transformer/ogbg-molhiv/run_2021-07-21-03-05-26_d_10_e_256_h_8_he_4_a_1e-2/config.json"
    glt_config = GraphLaplacianTransformerConfig.from_json(config_path)
    # glt_config = GraphLaplacianTransformerConfig(
    #     6,
    #     2,
    #     128,
    #     4,
    #     1,
    #     1,
    #     use_bias=False,
    #     use_edge_bias=False,
    #     use_attn_expand_bias=False,
    #     head_expand_scale=1,
    #     ff_dropout=0.1,
    #     attention_dropout=0.1,
    #     path_dropout=0.05,
    # )
    glt = GraphLaplacianTransformerWithLinearClassifier(glt_config)
    print(glt.num_parameters())
    assert False
    
    #
    # atom_encoder = AtomEncoder(emb_dim = 128)
    # bond_encoder = BondEncoder(emb_dim = 128)
    # evaluator = Evaluator(name=dataset_name)
    dataset = PygGraphPropPredDataset(name=dataset_name, pre_transform=RemoveIsolatedNodes()) 

    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    criterian = torch.nn.BCEWithLogitsLoss()

    # torch.autograd.set_detect_anomaly(True)
    # y_true, y_pred = [], []

    for data in train_loader:
        x, edges, edge_index, batch, y = data.x, data.edge_attr, data.edge_index.to(torch.long), data.batch, data.y.to(torch.float)
        graph_portion = batch.bincount()
        mask = ~torch.isnan(y)

        out = glt(x, edges, edge_index, graph_portion)
        
        loss = criterian(out, y)
        loss.backward()

        for name, params in glt.named_parameters():
            if params.grad is not None:
                print(name, params.grad.norm())

        assert False
    
        # out = glt(x, edges, edge_index, graph_portion)
        # print(out.requires_grad)
        # print(out.shape, y.shape)
        # loss = criterian(out[mask], y[mask])
        # loss.backward()

        # for name, param in glt.named_parameters():
        #     if "token_layers" in name:
        #         print(name, param.grad.norm())

        # assert False
        # eval_key = evaluator.eval_metric
        # evaluation_score = evaluator.eval({
        #     'y_pred': out,
        #     'y_true': y,
        # })[eval_key]
        # print("evaluation_score: ", evaluation_score)
        # y_true.append(y)
        # y_pred.append(out)

    # y_true = torch.cat(y_true)
    # y_pred = torch.cat(y_pred)
    # eval_key = evaluator.eval_metric
    # evaluation_score = evaluator.eval({
    #     'y_pred': y_pred,
    #     'y_true': y_true,
    # })[eval_key]
    # print("evaluation_score: ", evaluation_score)
        


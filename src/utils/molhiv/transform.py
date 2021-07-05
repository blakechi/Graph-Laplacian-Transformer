import os
import sys
import copy
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(2, os.getcwd())
import torch
import torch.nn.functional as F
import torch_geometric
from einops import rearrange
from ogb.utils.features import get_bond_feature_dims
full_bond_feature_dims = get_bond_feature_dims()

from utils.preprocessing import get_transition_matrix, get_k_step_transition_matrix_stack


def pre_transform(graph, max_num_nodes, max_num_edges, k_hops=2, aggregate_neighbor=False, num_eig_vector=16):
    num_features, num_nodes = graph.x.shape[-1], graph.num_nodes
    num_node_pads = max_num_nodes - num_nodes

    # Pad with 0 and shift 1 for every features to right
    graph.x = graph.x + 1
    graph.x = torch.cat(
        [graph.x, torch.zeros([num_node_pads, num_features])],
        dim=0,
    ).to(torch.long).unsqueeze(0)


    # Add Padding at 0 and shift 1 for every edge features to right
    graph.edge_attr = graph.edge_attr + 1

    # Remove self-looping
    if graph.contains_self_loops():
        edge_index, edge_attr = torch_geometric.utils.remove_self_loops(graph.edge_index, graph.edge_attr)
        graph.edge_index = edge_index
        graph.edge_attr = edge_attr

    # # Get Graph Laplacian
    # lap_index, lap_weight = torch_geometric.utils.get_laplacian(edge_index=graph.edge_index)
    # graph.lap_index = lap_index
    # graph.lap_weight = lap_weight

    # Get eigen vectors of Laplacian
    # lap = torch_geometric.utils.to_dense_adj(
    #     edge_index=graph.lap_index, 
    #     edge_attr=graph.lap_weight, 
    #     max_num_nodes=num_nodes
    # )
    # eigen_value, eigen_vector = lap[0].symeig(eigenvectors=True)

    # # Filter out DC term
    # eigen_value = eigen_value[1:]
    # eigen_vector = eigen_vector[:, 1:]

    # # Filter out high frequency term or pad with 0 if not enough
    # if eigen_vector.shape[1] < num_eig_vector:
    #     eigen_vector = F.pad(eigen_vector, (0, num_eig_vector - num_nodes + 1))
    #     eigen_value = F.pad(eigen_value, (0, num_eig_vector - num_nodes + 1))
    # else:
    #     eigen_value = eigen_value[:num_eig_vector]
    #     eigen_vector = eigen_vector[:, :num_eig_vector]

    # graph.eig_value = eigen_value
    # graph.eig_vector = eigen_vector

    # Get Dense Adjacency Matrix without edge information
    adj = torch_geometric.utils.to_dense_adj(
        edge_index=graph.edge_index, 
        max_num_nodes=max_num_nodes
    )

    # k-step neighbors' masks
    transition_matrix = get_transition_matrix(adj[0])
    stack = get_k_step_transition_matrix_stack(
        transition_matrix,
        k_hops
    )

    # for k_hop in range(1, k_hops + 1):
    #     stack[k_hop] += stack[k_hop - 1]

    # stack = stack > 0
    # if not aggregate_neighbor:
    #     for k_hop in reversed(range(1, k_hops + 1)):
    #         overlapping = torch.logical_and(stack[k_hop], stack[k_hop - 1])
    #         stack[k_hop] = torch.logical_xor(stack[k_hop], overlapping)

    graph.k_hops_neighbor_mask = stack > 0

    # padding_mask: The padded part would be True
    graph.padding_mask = torch.cat(
        [torch.zeros([graph.num_nodes]), torch.ones([max_num_nodes - graph.num_nodes])],
        dim=0,
    ).bool().unsqueeze(0)

    # edge_attr
    feature_count = 0
    for idx, num_features in enumerate(full_bond_feature_dims):
        graph.edge_attr[:, idx] = graph.edge_attr[:, idx] + feature_count
        feature_count = feature_count + num_features + 1

    return graph


def transform(graph, k_hops=2, num_eig_vector=16):
    max_num_nodes = graph.padding_mask.shape[0]
    num_node_pads = max_num_nodes - graph.num_nodes

    # Get adjacent matrix with edge information
    # graph.adj = torch_geometric.utils.to_dense_adj(
    #     edge_index=graph.edge_index, 
    #     edge_attr=graph.edge_attr, 
    #     max_num_nodes=max_num_nodes
    # )

    # Get dense laplacian matrix
    # graph.lap = torch_geometric.utils.to_dense_adj(
    #     edge_index=graph.lap_index, 
    #     edge_attr=graph.lap_weight, 
    #     max_num_nodes=max_num_nodes
    # )

    # Pad eigen vector
    # graph.eig_vector = torch.cat(
    #     [graph.eig_vector, torch.zeros([num_node_pads, num_eig_vector])],
    #     dim=0,
    # )

    # Get relative position indice
    stack = graph.k_hops_neighbor_mask
    pos_indice = torch.zeros_like(stack[0], dtype=torch.long).fill_(k_hops + 1)  # means out of range

    for k in range(k_hops + 1):
        pos_indice.masked_fill_(stack[k], k)

    graph.pos_indice = pos_indice.unsqueeze(dim=0)

    #
    # uniform_adj = (graph.adj.sum(dim=-1) > 0)
    # graph.degree = uniform_adj.sum(dim=-1).view(-1, 1)

    return graph


if __name__ == "__main__":
    from dataset import MolHivDataset
    dataset = MolHivDataset(name="ogbg-molhiv", root='dataset/', transform=transform, pre_transform=pre_transform)
    split_idx = dataset.get_idx_split() 

    # from embedding import AtomEncoder, BondEncoder
    # atom_encoder = AtomEncoder(emb_dim=64)
    # bond_encoder = BondEncoder(emb_dim=64)
    # atom_emb = atom_encoder(dataset[0].x) # x is input atom feature
    # bond_emb = bond_encoder(dataset[0].edge_attr) # x is input atom feature
    # print(atom_encoder)
    # print(bond_encoder)
    # assert False

    from torch_geometric.data import DataLoader
    # train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    # valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

    # max_degree = 0
    for idx, batch in enumerate(test_loader):
        data_list = batch.to_data_list()
        
        # graph = data_list[0]
        # print((graph.adj > 0).sum(dim=-1).diagonal(dim1=-2, dim2=-1).sum())

        # adjs = batch.adj
        # adjs = rearrange(adjs, "(b c) n m d -> b c n m d", b=32)
        # print((adjs[0].sum(dim=-1) > 0).sum())
        x = batch.x.reshape(32, 222, -1)[0]
        print(x.shape)
        padding_mask = batch.padding_mask.reshape(32, -1)[0]
        print(padding_mask.sum())
        print((x == 0).sum().float()/9.0)
        print("-----------------------")
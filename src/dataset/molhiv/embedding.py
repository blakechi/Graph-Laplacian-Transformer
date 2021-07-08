import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim + 1, emb_dim)  # reserve space for padding
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0

        for i in range(x.shape[-1]):
            # x_embedding += self.atom_embedding_list[i](x[:, i])
            # x_embedding += self.atom_embedding_list[i](x[:, :, i])
            x_embedding += self.atom_embedding_list[i](x[..., i])  # For both cases listed above

        return x_embedding.squeeze(dim=-2)


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim + 1, emb_dim)  # reserve space for padding
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0

        for i in range(edge_attr.shape[-1]):
            # bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])
            # bond_embedding += self.bond_embedding_list[i](edge_attr[:, :, i])
            bond_embedding += self.bond_embedding_list[i](edge_attr[..., i])  # For both cases listed above

        return bond_embedding.squeeze(dim=-2)
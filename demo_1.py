from torch_geometric.data import DataLoader
from torch_geometric.transforms import RemoveIsolatedNodes
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


batch_size = 32
nworkers = 1
pin_memory = True

dataset = PygGraphPropPredDataset(name="ogbg-molpcba") 

split_idx = dataset.get_idx_split() 
train_loader = DataLoader(
    dataset[split_idx["train"]],
    batch_size=batch_size,
    shuffle=True,
    num_workers=nworkers,
    pin_memory=pin_memory
)

for data in train_loader:
    d = data.to_data_list()
    print(len(d))
    print(d[0])

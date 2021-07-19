import os
import argparse

import torch
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from src.model import GraphLaplacianTransformerConfig, GraphLaplacianTransformerWithLinearClassifier
from parser import set_parser


# Parser
#
parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser = set_parser(parser)


if __name__ == "__main__":
    # Prepare args and creat loggings
    args = parser.parse_args()
    args.device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        args.device = f"cuda:{args.cuda_device}"

     # Dataset
    dataset = PygGraphPropPredDataset(
        name=args.dataset_name,
        root=args.dataset_dir,
        # pre_transform=RemoveIsolatedNodes()
    )

    split_idx = dataset.get_idx_split() 
    # train_loader = DataLoader(
    #     dataset[split_idx["train"]],
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_memory
    # )
    # valid_loader = DataLoader(
    #     dataset[split_idx["valid"]],
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_memory
    # )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    # Model
    config = GraphLaplacianTransformerConfig(**args.__dict__)
    model = GraphLaplacianTransformerWithLinearClassifier(config)
    # for name, params in model.named_parameters():
    #     print(name, params.norm())
    checkpoint_path = os.path.join(args.log_dir, args.dataset_name, args.checkpoint_folder, args.checkpoint_name)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    # Optimizer
    no_weight_decays = model.no_weight_decays
    optimizer_grouped_parameters = [
        {
            'params': [
                params for name, params in model.named_parameters()
                if not any(nd in name for nd in no_weight_decays)
            ],
            'weight_decay': args.weight_decay
        },
        {
            'params': [
                params for name, params in model.named_parameters()
                if any(nd in name for nd in no_weight_decays)
            ],
            'weight_decay': 0.0
        }
    ]
    optimizer = torch.optim.Adam(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=args.betas,
    )
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # print(optimizer.state_dict())
    # Loss & Evaluator
    loss_fn = torch.nn.BCEWithLogitsLoss().to(args.device)
    evaluator = Evaluator(name=args.dataset_name)


    for name, params in model.named_parameters():
        print(name, params.norm())
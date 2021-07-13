#!/usr/bin/env python3
"""
Reference from: https://github.com/rwightman/pytorch-image-models/blob/master/train.py
"""

import argparse
import os
import logging
import json
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.transforms import RemoveIsolatedNodes
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from warmup_scheduler import GradualWarmupScheduler

from src.model import GraphLaplacianTransformerConfig, GraphLaplacianTransformerWithLinearClassifier
from parser import set_parser
from train_and_evaluate import train_one_epoch, evaluate_or_test


# Loggere
#
FORMAT = '%(asctime)s - %(levelname)s - file: %(pathname)s - line: %(lineno)d - %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Parser
#
parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser = set_parser(parser)


def main():
    # Prepare args and creat loggings
    args = parser.parse_args()
    args.device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        args.device = f"cuda:{args.cuda_device}"
        
    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")
    task_folder = os.path.join(args.log_dir, args.dataset_name)
    run_folder = os.path.join(task_folder, f"run_{now}_{args.log_msg}")
    if not os.path.exists(task_folder):
        os.mkdir(task_folder)
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)

    file_handler = logging.FileHandler(
        os.path.join(run_folder, "logging.log"),
        mode='w'
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    writer = SummaryWriter(log_dir=os.path.join(task_folder, "runs"), filename_suffix=f"{now}_{args.log_msg}")

    with open(os.path.join(run_folder, "config.txt"), 'w') as json_file:
        json.dump(args.__dict__, json_file, indent=4)


    # Dataset
    logger.info(f"Reading datasets...")
    dataset = PygGraphPropPredDataset(
        name=args.dataset_name,
        root=args.dataset_dir,
        # pre_transform=RemoveIsolatedNodes()
    )

    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )


    # Model
    logger.info(f"Initializing the model...")
    config = GraphLaplacianTransformerConfig(**args.__dict__)
    model = GraphLaplacianTransformerWithLinearClassifier(config)


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


    # Scheduler
    lr_scheduler_tail = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs*len(train_loader),
        eta_min=args.min_lr
    )
    lr_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1e1,
        total_epoch=2*len(train_loader),
        after_scheduler=lr_scheduler_tail
    )


    # Loss & Evaluator
    loss_fn = torch.nn.BCEWithLogitsLoss().to(args.device)
    evaluator = Evaluator(name=args.dataset_name)


    # Train Loop
    logger.info(f"Starting to train... (Epoch: {args.epochs})")
    model.to(args.device)
    for epoch in range(args.epochs):
        train_metric = train_one_epoch(
            epoch,
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            loss_fn,
            evaluator,
            writer,
            logger,
            args
        )

        valid_metric = evaluate_or_test(
            epoch,
            valid_loader,
            model,
            loss_fn,
            evaluator,
            writer,
            logger,
            args,
            step=(epoch + 1)*len(train_loader) - 1
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metric': train_metric,
            'valid_metric': valid_metric,
            },
            os.path.join(run_folder, f"checkpoint_{now}_{epoch}_{(epoch + 1)*len(train_loader) - 1}.pt")
        )

    test_metric = evaluate_or_test(
        epoch,
        test_loader,
        model,
        loss_fn,
        evaluator,
        writer,
        args,
        step=(epoch + 1)*len(train_loader) - 1,
        mode='test'
    )

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
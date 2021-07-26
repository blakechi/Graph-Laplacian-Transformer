#!/usr/bin/env python3
"""
Reference from: https://github.com/rwightman/pytorch-image-models/blob/master/train.py
"""

import os
import json
import logging
import argparse
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
# from torch_geometric.transforms import RemoveIsolatedNodes
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from warmup_scheduler import GradualWarmupScheduler

from src.model import GraphLaplacianTransformerConfig, GraphLaplacianTransformerWithLinearClassifier
from src.utils import RemoveIsolatedNodes
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
        torch.cuda.set_device(f"cuda:{args.cuda_device}")
        args.device = f"cuda:{args.cuda_device}"
        
    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")
    task_folder = os.path.join(args.log_dir, args.dataset_name)
    new_run_folder = os.path.join(task_folder, f"run_{now}_{args.log_msg}")
    if not os.path.exists(task_folder):
        os.mkdir(task_folder)
    if not os.path.exists(new_run_folder):
        os.mkdir(new_run_folder)

    file_handler = logging.FileHandler(
        os.path.join(new_run_folder, "logging.log"),
        mode='w'
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info(f"PID: {os.getpid()}")
    
    writer = SummaryWriter(log_dir=os.path.join(task_folder, "runs"), filename_suffix=f"{now}_{args.log_msg}")

    with open(os.path.join(new_run_folder, "config.json"), 'w') as json_file:
        json.dump(args.__dict__, json_file, indent=4)


    # Dataset
    logger.info(f"Reading datasets...")
    dataset = PygGraphPropPredDataset(
        name=args.dataset_name,
        root=args.dataset_dir,
        pre_transform=RemoveIsolatedNodes()
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
    logger.info("Initializing the model...")
    old_run_folder = os.path.join(args.log_dir, args.dataset_name, args.run_folder)
    if args.run_folder and args.config_name:
        logger.info("Configuring the model by the given configuration file...")
        config_path = os.path.join(old_run_folder, args.config_name)
        config = GraphLaplacianTransformerConfig.from_json(config_path)
    else:
        config = GraphLaplacianTransformerConfig(**args.__dict__)

    model = GraphLaplacianTransformerWithLinearClassifier(config)
    if args.run_folder and args.checkpoint_name:
        logger.info("Loading parameters from the given checkpoint...")
        checkpoint_path = os.path.join(old_run_folder, args.checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs*len(train_loader),
        eta_min=args.min_lr
    )
    # lr_scheduler = GradualWarmupScheduler(
    #     optimizer,
    #     multiplier=args.max_lr/args.lr,
    #     total_epoch=1*len(train_loader),
    #     after_scheduler=lr_scheduler_tail
    # )


    # Loss & Evaluator
    loss_fn = torch.nn.BCEWithLogitsLoss().to(args.device)
    evaluator = Evaluator(name=args.dataset_name)
    eval_key = evaluator.eval_metric

    # Train Loop
    logger.info(f"Starting to train... (Total Epoch: {args.epochs})")
    best_valid_score = 0.
    model.to(args.device)

    nan_list = []
    for name, params in model.named_parameters():
        if torch.any(params.isnan()):
            nan_list.append((name, params))
    
    assert len(nan_list) == 0, f"Find nan in the model:\n{nan_list}"

    
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
        
        # Storing the model when it gets record high for valid metric, or every 10 epochs, or at the last epoch 
        if valid_metric[eval_key] > best_valid_score or (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            test_metric = evaluate_or_test(
                epoch,
                test_loader,
                model,
                loss_fn,
                evaluator,
                writer,
                logger,
                args,
                step=(epoch + 1)*len(train_loader) - 1,
                mode='test'
            )

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metric': train_metric,
                'valid_metric': valid_metric,
                'test_metric': test_metric,
                },
                os.path.join(new_run_folder, f"checkpoint_{now}_{epoch}_{(epoch + 1)*len(train_loader) - 1}.pt")
            )
            best_valid_score = valid_metric[eval_key]

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
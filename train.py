#!/usr/bin/env python3
"""
Reference from: https://github.com/rwightman/pytorch-image-models/blob/master/train.py
"""

import argparse
import time
import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.transforms import RemoveIsolatedNodes
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from timm.utils import AverageMeter

from src.model import GraphLaplacianTransformerConfig, GraphLaplacianTransformerWithLinearClassifier
# from src.dataset import CorrelationDataset

# Logger
#
FORMAT = '%(asctime)s - %(levelname)s - line: %(lineno)d - %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Parser
#
parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# Path
# parser.add_argument('--dir', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--log_dir', metavar='LOGDIR',
                    help='path for the logging file')

# Dataset
parser.add_argument('--dataset_name', default="", type=str,
                    metavar='N', help='Graph dataset name in OGB')
parser.add_argument('--pin_memory', default=False, action='store_true',
                    help='Whether to copy batches into CUDA pinned memory')

# Model
parser.add_argument('--depth', default=2, type=int,
                    metavar='N', help='The number of conv layers')
parser.add_argument('--base_num_channel', default=4, type=int,
                    metavar='N', help='The number of channels after the first conv layer')

# Optimizer - Adam
parser.add_argument('--betas', type=float, default=[0.9, 0.999], nargs=2,
                    metavar='N', help='betas')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')

# Procedure
parser.add_argument('--epochs', type=int, default=200, metavar='E',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='MINLR',
                    help='Minimum learning rate (default: 1e-6)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--logging_interval', type=int, default=32, metavar='N',
                    help='logging per N batches')


def main():
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    file_handler = logging.FileHandler(
        os.path.join(args.log_dir, "logging.log"),
        mode='w'
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    
    writer = SummaryWriter(comment=f"_{args.batch_size}_{args.lr}_{args.min_lr}")

    # Model
    logger.info(f"Initializing the model...")
    config = GraphLaplacianTransformerConfig.MOLPCBA()
    model = GraphLaplacianTransformerWithLinearClassifier(config)

    # Optimizer
    no_decay = model.no_weight_decay()
    optimizer_grouped_parameters = [
        {
            'params': [
                params for name, params in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': args.weight_decay
        },
        {
            'params': [
                params for name, params in model.named_parameters()
                if any(nd in name for nd in no_decay)
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
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    # Dataset
    logger.info(f"Reading datasets...")

    dataset = PygGraphPropPredDataset(name = args.dataset_name) 

    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.nworkers,
        pin_memory=args.pin_memory
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.nworkers,
        pin_memory=args.pin_memory
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.nworkers,
        pin_memory=args.pin_memory
    )

    # Loss
    loss_fn = nn.MSELoss().to(args.device)


    # Train Loop
    logger.info(f"Starting to train... (Epoch: {args.epochs})")
    model.to(args.device)
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            epoch,
            train_loader,
            model,
            optimizer,
            loss_fn,
            writer,
            args
        )

        test_loss = evaluate(epoch, test_loader, model, loss_fn, args)
        writer.add_scalar("Loss/test", test_loss, (epoch + 1)*(len(train_loader) - 1))  # Steps

        lr_scheduler.step()

    writer.flush()
    writer.close()


def train_one_epoch(epoch, loader, model, optimizer, loss_fn, writer, args):    
    last_idx = len(loader) - 1
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()

        x = batch['image'].to(args.device)
        y = batch['corr'].to(args.device)

        y_pred = model(x)
        
        loss = loss_fn(y_pred, y)
        losses.update(loss.item(), x.shape[0])

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start_time)
        writer.add_scalar("Loss/train", loss.item(), epoch*len(loader) + batch_idx)  # Steps

        if batch_idx != 0 and batch_idx % args.logging_interval == 0:
            logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'Time: {batch_time.val:.3f}s  '
                'LR: {lr:.3e}  '.format(
                    epoch, batch_idx, len(loader), 100. * batch_idx / last_idx,
                    loss=losses,
                    batch_time=batch_time,
                    lr=optimizer.param_groups[0]['lr'],
                )
            )

    return losses.avg


def evaluate(epoch, loader, model, loss_fn, args):
    last_idx = len(loader) - 1
    losses = AverageMeter()

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch['image'].to(args.device)
            y = batch['corr'].to(args.device)

            y_pred = model(x)
            
            loss = loss_fn(y_pred, y)
            losses.update(loss.item(), x.shape[0])

    end_time = time.time()

    logger.info(
        'Evaluate: {} [{:>4d}/{} ({:>3.0f}%)]  '
        'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
        'Time: {time:.3f}s  '.format(
            epoch, batch_idx, len(loader), 100. * batch_idx / last_idx,
            loss=losses,
            time=end_time - start_time,
        )
    )

    return losses.avg


if __name__ == "__main__":
    main()
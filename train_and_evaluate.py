"""
Reference from: https://github.com/rwightman/pytorch-image-models/blob/master/train.py
"""

import time

import torch
from timm.utils import AverageMeter


def train_one_epoch(epoch, loader, model, optimizer, loss_fn, evaluator, writer, logger, args):    
    last_idx = len(loader) - 1
    batch_time = AverageMeter()
    losses = AverageMeter()
    y_pred_list, y_true_list = [], []

    model.train()
    start_time = time.time()
    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad()

        x, edges, edge_index, batch, y = data.x, data.edge_attr, data.edge_index.to(torch.long), data.batch, data.y
        x = x.to(args.device)
        edges = edges.to(args.device)
        edge_index = edge_index.to(args.device)
        graph_portion = batch.bincount().to(args.device)
        mask = ~torch.isnan(y)

        logit = model(x, edges, edge_index, graph_portion)
        logit = logit.cpu()

        loss = loss_fn(logit[mask], y[mask])
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start_time)
        y_true_list.append(y)
        y_pred_list.append(logit)
        losses.update(loss.item(), args.batch_size)
        writer.add_scalar("Loss/train", losses.val, epoch*len(loader) + batch_idx)  # Steps

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

    y_pred, y_true = torch.cat(y_pred_list), torch.cat(y_true_list)
    eval_key = evaluator.eval_metric
    evaluation_score = evaluator.eval({
        'y_pred': y_pred,
        'y_true': y_true,
    })[eval_key]
    metrics = {
        "loss": losses.avg,
        eval_key: evaluation_score
    }
    writer.add_scalar(f"Evaluation/train_{eval_key}", evaluation_score, (epoch + 1)*len(loader) - 1)  # zero-based, so "- 1"

    logger.info(
        f'Train/Evaluation[{eval_key}]: {metrics[eval_key]}'
    )

    return metrics


@torch.no_grad()
def evaluate_or_test(epoch, loader, model, loss_fn, evaluator, writer, logger, args, step=0, mode='eval'):
    losses = AverageMeter()
    y_pred_list, y_true_list = [], []

    model.eval()
    start_time = time.time()
    for batch_idx, data in enumerate(loader):
        x, edges, edge_index, batch, y = data.x, data.edge_attr, data.edge_index.to(torch.long), data.batch, data.y
        x = x.to(args.device)
        edges = edges.to(args.device)
        edge_index = edge_index.to(args.device)
        graph_portion = batch.bincount().to(args.device)
        mask = ~torch.isnan(y)

        logit = model(x, edges, edge_index, graph_portion)
        logit = logit.cpu()

        loss = loss_fn(logit[mask], y[mask])

        y_true_list.append(y)
        y_pred_list.append(logit)
        losses.update(loss.item(), args.batch_size)

    end_time = time.time()
    writer.add_scalar(f"Loss/{mode}", losses.val, step)  # Steps

    logger.info(
        '{}: {}  '
        'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
        'Time: {time:.3f}s  '.format(
            mode.capitalize(), epoch,
            loss=losses,
            time=end_time - start_time,
        )
    )

    y_pred, y_true = torch.cat(y_pred_list), torch.cat(y_true_list)
    eval_key = evaluator.eval_metric
    evaluation_score = evaluator.eval({
        'y_pred': y_pred,
        'y_true': y_true,
    })[eval_key]
    metrics = {
        "loss": losses.avg,
        eval_key: evaluation_score
    }
    writer.add_scalar(f"Evaluation/{mode}_{eval_key}", evaluation_score, (epoch + 1)*len(loader) - 1)  # zero-based, so "- 1"

    logger.info(
        f'{mode.capitalize()}/Evaluation[{eval_key}]: {metrics[eval_key]}'
    )

    return metrics
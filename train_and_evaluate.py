import time

import torch
from timm.utils import AverageMeter


def train_one_epoch(epoch, data_loader, model, optimizer, lr_scheduler, loss_fn, evaluator, writer, logger, args):    
    num_steps = len(data_loader)
    last_step = num_steps - 1
    batch_time = AverageMeter()
    losses = AverageMeter()
    y_pred_list, y_true_list = [], []

    model.train()
    start_time = time.time()
    for batch_idx, data in enumerate(data_loader):
        optimizer.zero_grad()

        graph_edge_index = [graph.edge_index.to(torch.long).to(args.device) for graph in data.to_data_list()]
        required_data = data.x, data.edge_attr, data.edge_index.to(torch.long), data.batch.bincount(), data.y.to(torch.float)
        x, edges, edge_index, graph_portion, y = map(lambda ele: ele.to(args.device), required_data)
        mask = ~torch.isnan(y)

        logit, attn_kldiv_loss = model(x, edges, edge_index, graph_portion, graph_edge_index)
        
        loss = loss_fn(logit[mask], y[mask])
        loss += args.gamma*attn_kldiv_loss
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start_time)
        y_pred_list.append(logit.detach().cpu())
        y_true_list.append(y.detach().cpu())
        losses.update(loss.item(), args.batch_size)
        writer.add_scalar("Loss/train", losses.val, epoch*num_steps + batch_idx)  # Steps
        
        if batch_idx != 0 and batch_idx % args.logging_interval == 0:
            logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'Time: {batch_time.val:.3f}s  '
                'LR: {lr:.3e}  '.format(
                    epoch, batch_idx, num_steps, 100. * batch_idx / last_step,
                    loss=losses,
                    batch_time=batch_time,
                    lr=optimizer.param_groups[0]['lr'],
                )
            )

        lr_scheduler.step()

    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)
    eval_key = evaluator.eval_metric
    evaluation_score = evaluator.eval({
        'y_pred': y_pred,
        'y_true': y_true,
    })[eval_key]

    writer.add_scalar(f"Evaluation/train_{eval_key}_epoch", evaluation_score, (epoch + 1)*num_steps - 1)  # zero-based, so "- 1"
    logger.info(
        f'Train/Evaluation[{eval_key}]: {evaluation_score}'
    )

    metrics = {
        "loss": losses.avg,
        eval_key: evaluation_score,
    }

    return metrics


@torch.no_grad()
def evaluate_or_test(epoch, data_loader, model, loss_fn, evaluator, writer, logger, args, step=0, mode='eval'):
    num_steps = len(data_loader)
    losses = AverageMeter()
    y_pred_list, y_true_list = [], []

    model.eval()
    start_time = time.time()
    for batch_idx, data in enumerate(data_loader):
        graph_edge_index = [graph.edge_index.to(torch.long).to(args.device) for graph in data.to_data_list()]
        required_data = data.x, data.edge_attr, data.edge_index.to(torch.long), data.batch.bincount(), data.y.to(torch.float)
        x, edges, edge_index, graph_portion, y = map(lambda ele: ele.to(args.device), required_data)
        mask = ~torch.isnan(y)

        logit, _ = model(x, edges, edge_index, graph_portion, graph_edge_index)

        loss = loss_fn(logit[mask], y[mask])

        y_pred_list.append(logit.detach().cpu())
        y_true_list.append(y.detach().cpu())
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

    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)
    eval_key = evaluator.eval_metric
    evaluation_score = evaluator.eval({
        'y_pred': y_pred,
        'y_true': y_true,
    })[eval_key]

    writer.add_scalar(f"Evaluation/{mode}_{eval_key}_epoch", evaluation_score, (epoch + 1)*num_steps - 1)  # zero-based, so "- 1"
    logger.info(
        f'{mode.capitalize()}/Evaluation[{eval_key}]: {evaluation_score}'
    )

    metrics = {
        "loss": losses.avg,
        eval_key: evaluation_score
    }

    return metrics
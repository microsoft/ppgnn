import argparse

import torch
import torch.nn.functional as F


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.utils.data,
) -> float:
    """Train step for a single batch of data"""

    model.train()
    optimizer.zero_grad()
    out = model(data)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    del out
    return loss.item()


def eval_step(
    model: torch.nn.Module, data: torch.utils.data
) -> tuple([list, list, list]):
    """Evaluate the model on a single batch of data"""

    model.eval()
    logits, accs, losses, preds = model(data), [], [], []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        loss = F.nll_loss(model(data)[mask], data.y[mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())
    return accs, preds, losses


def get_optimizer(args: argparse, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Return the optimizer for the given model"""

    if args.net in ["APPNP", "GPRGNN"]:
        optimizer = torch.optim.Adam(
            [
                {
                    "params": model.lin1.parameters(),
                    "weight_decay": args.weight_decay,
                    "lr": args.lr,
                },
                {
                    "params": model.lin2.parameters(),
                    "weight_decay": args.weight_decay,
                    "lr": args.lr,
                },
                {
                    "params": model.prop1.parameters(),
                    "weight_decay": 0.0,
                    "lr": args.lr,
                },
            ],
            lr=args.lr,
        )

    elif args.net in ["PPGNN"]:
        trainable_vars = []

        if args.beta >= 0 and args.beta < 1:
            GPR_vars = [
                {
                    "params": model.gprnn_model.lin1.parameters(),
                    "weight_decay": args.weight_decay,
                    "lr": args.lr,
                },
                {
                    "params": model.gprnn_model.lin2.parameters(),
                    "weight_decay": args.weight_decay,
                    "lr": args.lr,
                },
                {
                    "params": model.gprnn_model.prop1.parameters(),
                    "weight_decay": 0.0,
                    "lr": args.lr,
                },
            ]

            trainable_vars += GPR_vars

        if args.beta > 0 and args.beta <= 1:
            GPRGNN_pro_vars = [
                {"params": model.piecewise_model.polynomials.parameters()},
            ]

            trainable_vars += GPRGNN_pro_vars

        optimizer = torch.optim.Adam(trainable_vars, lr=args.lr)

    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    return optimizer


def train_model(
    args: argparse,
    dataset: torch.utils.data,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple([float, float]):
    """Train the model on the given dataset"""

    data = dataset.data
    model, data = model(dataset, args).to(device), data.to(device)

    best_val_acc, best_test_acc = 0, 0
    val_loss_history, val_acc_history, predictions = [], [], []

    optimizer = get_optimizer(args, model)

    for epoch in range(args.epochs):
        _ = train_step(model, optimizer, data)

        (
            [train_acc, val_acc, test_acc],
            _,
            [_, val_loss, _],
        ) = eval_step(model, data)

        print(
            "Epoch: {} | Train Acc: {:.4f} | Val Loss: {:.4f} | Test Acc: {:.4f}".format(
                epoch, train_acc, val_loss, test_acc
            )
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(val_loss_history[-(args.early_stopping + 1) : -1])
                if val_loss > tmp.mean().item():
                    break

    return best_test_acc, best_val_acc

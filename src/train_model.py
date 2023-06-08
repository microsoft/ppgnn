#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.
# An adaption of GPR-GNN source code

import argparse
import os
import pickle as pkl
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset_utils import DataLoaderNew
# from utils import random_planetoid_splits
from GNN_models import *

np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)


def RunExp(args, dataset, data, Net, percls_trn, val_lb):
    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
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

    appnp_net = Net(dataset, args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, data = appnp_net.to(device), data.to(device)

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
                {
                    "params": model.en_model.lin1.parameters(),
                    "weight_decay": args.weight_decay,
                    "lr": args.lr,
                },
                {
                    "params": model.en_model.lin2.parameters(),
                    "weight_decay": args.weight_decay,
                    "lr": args.lr,
                },
                {"params": model.en_model.polynomials.parameters()},
            ]

            trainable_vars += GPRGNN_pro_vars

        optimizer = torch.optim.Adam(trainable_vars, lr=args.lr)

    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    best_val_acc = test_acc = 0
    best_val_loss = float("inf")
    val_loss_history = []
    val_acc_history = []
    predictions = []

    learned_gammas = []

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        (
            [train_acc, val_acc, tmp_test_acc],
            preds,
            [train_loss, val_loss, tmp_test_loss],
        ) = test(model, data)

        print(
            "Epoch: [{}]	Train Acc: {:.4f}\tVal Loss: {:.4f}\tTest Acc: {:.4f}".format(
                epoch, train_acc, val_loss, tmp_test_acc
            )
        )

        poly_vals_epoch = []
        for idx, poly in enumerate(model.en_model.polynomials):
            poly_vals_epoch.append(poly.detach().cpu().numpy())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            predictions = preds
            if args.net == "GPRGNN":
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(val_loss_history[-(args.early_stopping + 1) : -1])
                if val_loss > tmp.mean().item():
                    break

    print(test_acc)
    return (
        test_acc,
        best_val_acc,
        Gamma_0,
        predictions[2].detach().cpu().numpy(),
    )


if __name__ == "__main__":
    adjType = "sym_adj"
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--early_stopping", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train_rate", type=float, default=0.025)
    parser.add_argument("--val_rate", type=float, default=0.025)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--dprate", type=float, default=0.5)
    parser.add_argument("--C", type=int)
    parser.add_argument(
        "--Init",
        type=str,
        choices=["SGC", "PPR", "NPPR", "Random", "WS", "Null"],
        default="PPR",
    )
    parser.add_argument("--Gamma", default=None)
    parser.add_argument("--ppnp", default="GPR_prop", choices=["PPNP", "GPR_prop"])
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--output_heads", default=1, type=int)

    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--RPMAX", type=int, default=1)
    parser.add_argument(
        "--net",
        type=str,
        choices=["GCN", "GAT", "APPNP", "ChebNet", "JKNet", "GPRGNN", "PPGNN"],
        default="GPRGNN",
    )
    parser.add_argument("--split", type=int, default=0, required=True)
    parser.add_argument("--norm", type=str, default="no")

    # PPGNN Specific Hyperparameters
    parser.add_argument("--total_buckets", type=int, default=5)
    parser.add_argument("--alphas", type=str)
    parser.add_argument("--evd_dims", type=int, default=64)
    parser.add_argument("--beta", type=str)

    args = parser.parse_args()

    gnn_name = args.net
    if gnn_name == "GCN":
        Net = GCN_Net
    elif gnn_name == "GAT":
        Net = GAT_Net
    elif gnn_name == "APPNP":
        Net = APPNP_Net
    elif gnn_name == "ChebNet":
        Net = ChebNet
    elif gnn_name == "JKNet":
        Net = GCN_JKNet
    elif gnn_name == "GPRGNN":
        Net = GPRGNN
    elif gnn_name == "PPGNN":
        Net = PPGNN

    args.feat_dims = 2048
    args.beta = float(args.beta)
    args.alphas = list(map(float, args.alphas.split(",")))
    args.norm = "sym"

    dname = args.dataset
    dataset = DataLoaderNew(dname, args, split=args.split, adj_type=adjType)
    data = dataset.data

    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None
    alpha = args.alpha
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    Results0 = []

    test_acc, best_val_acc, Gamma_0, predictions = RunExp(
        args, dataset, data, Net, percls_trn, val_lb
    )
    Results0.append([test_acc, best_val_acc, Gamma_0])

    test_acc_mean, val_acc_mean, _ = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    print(f"{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment:")
    print(
        f"test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}"
    )

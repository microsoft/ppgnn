import argparse

import numpy as np
import torch

from baselines import GPRGNN, APPNP_Net, ChebNet, GAT_Net, GCN_JKNet, GCN_Net
from dataset_utils import Dataset
from ppgnn import PPGNN
from train import train_model


def get_model(
    model_name: str,
) -> torch.nn.Module:
    """Obtains the model from the given model name"""

    if model_name == "GCN":
        model = GCN_Net
    elif model_name == "GAT":
        model = GAT_Net
    elif model_name == "APPNP":
        model = APPNP_Net
    elif model_name == "ChebNet":
        model = ChebNet
    elif model_name == "JKNet":
        model = GCN_JKNet
    elif model_name == "GPRGNN":
        model = GPRGNN
    elif model_name == "PPGNN":
        model = PPGNN

    return model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Common Arguments
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs to train for"
    )
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="Weight decay"
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=200,
        help="Number of epochs to wait before early stopping",
    )
    parser.add_argument("--hidden", type=int, default=64, help="Number of hidden units")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--dataset", default="Cora", help="Dataset name")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--net",
        type=str,
        choices=["GCN", "GAT", "APPNP", "ChebNet", "JKNet", "GPRGNN", "PPGNN"],
        default="PPGNN",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        help="Split number for the dataset",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="sym",
        help="Normalization method for the adjacency matrix",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../graph_datasets",
        help="Path to the dataset directory",
    )

    # GPRGNN and PPGNN hyper-params
    parser.add_argument("--K", type=int, default=10, help="Polynomial order")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Alpha value for PPR Initialization",
    )
    parser.add_argument(
        "--dprate",
        type=float,
        default=0.5,
        help="Dropout rate specific for Graphs",
    )
    parser.add_argument(
        "--Init",
        type=str,
        choices=["SGC", "PPR", "NPPR", "Random", "WS", "Null"],
        default="Random",
        help="Initialization method",
    )
    parser.add_argument("--ppnp", default="GPR_prop", choices=["PPNP", "GPR_prop"])

    # GAT Specific hyper-params
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--output_heads", default=1, type=int)

    # PPGNN specific hyper-params
    parser.add_argument(
        "--total_buckets",
        type=int,
        default=5,
        help="Total Number of Eigenvalue Buckets",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        help="Comma separated list of alphas for PPGNN Initialization",
    )
    parser.add_argument(
        "--evd_dims",
        type=int,
        default=64,
        help="Number of top and bottom eigenvalues to use for PPGNN Initialization",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Beta value for the tradeoff between general polynomial and the specific eigenvalue polynomials",
    )

    args = parser.parse_args()

    args.feat_dims = 2048
    args.beta = float(args.beta)
    args.alphas = list(map(float, args.alphas.split(","))) if args.alphas else None
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    model = get_model(args.net)
    dataset = Dataset(
        dataset_name=args.dataset,
        split=args.split,
        adj_type=args.norm + "_adj",
        evd_dims=args.evd_dims,
        total_buckets=args.total_buckets,
        data_dir=args.data_dir,
    )

    best_test_acc, best_val_acc = train_model(args, dataset, model, device)
    print(
        f"\nModel: {args.net} | Dataset: {args.dataset}, Split: {args.split} | Test Accuracy: {best_test_acc * 100:.2f} | Validation Accuracy: {best_val_acc * 100:.2f}"
    )

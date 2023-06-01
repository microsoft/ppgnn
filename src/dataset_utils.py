#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from typing import DefaultDict
import torch
import math
import os
import pickle as pkl
import h5py

import pandas as pd
from sklearn.utils.extmath import randomized_svd
import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.decomposition import PCA
from numpy import linalg as LA

from cSBM_dataset import dataset_ContextualSBM
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Amazon
from torch_geometric.nn import APPNP
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.io import read_npz


class dataset_heterophily(InMemoryDataset):
    def __init__(
        self,
        root="data/",
        name=None,
        p2raw=None,
        train_percent=0.01,
        transform=None,
        pre_transform=None,
    ):
        existing_dataset = ["chameleon", "film", "squirrel"]
        if name not in existing_dataset:
            raise ValueError(
                f"name of hypergraph dataset must be one of: {existing_dataset}"
            )
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!'
            )

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent.item()

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, "rb") as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)


class WebKB(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = (
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/"
        "master/new_data"
    )

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ["cornell", "texas", "washington", "wisconsin"]

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return ["out1_node_feature_label.txt", "out1_graph_edges.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        for name in self.raw_file_names:
            download_url(f"{self.url}/{self.name}/{name}", self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], "r") as f:
            data = f.read().split("\n")[1:-1]
            x = [[float(v) for v in r.split("\t")[1].split(",")] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split("\t")[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], "r") as f:
            data = f.read().split("\n")[1:-1]
            data = [[int(v) for v in r.split("\t")] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.name)


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


class Dataset:
    def __init__(
        self,
        name,
        all_args,
        split=0,
        adj_type="sym",
        transform=None,
        pre_transform=None,
    ):
        self.name = name.lower()
        self.split = split
        self.pre_transform = pre_transform

        self.data_dir = "../datasets"
        self.dataset_data = pkl.load(
            open(osp.join(self.data_dir, self.name + ".pkl"), "rb")
        )
        self.edges = torch.tensor(
            self.dataset_data[adj_type].nonzero(), dtype=torch.long
        )

        self.features = torch.tensor(self.dataset_data["X"], dtype=torch.float)
        self.labels = torch.tensor(
            self.dataset_data["labels"].argmax(1), dtype=torch.long
        )
        self.data = Data(x=self.features, edge_index=self.edges, y=self.labels)

        self.num_features = self.dataset_data["X"].shape[1]
        self.num_classes = self.dataset_data["labels"].shape[1]

        self.data = (
            self.data if self.pre_transform is None else self.pre_transform(self.data)
        )
        self.test_ids = self.dataset_data["split_" + str(split)]["test_ids"]

        # using all nodes for training
        self.data.train_mask = index_to_mask(
            self.dataset_data["split_" + str(split)]["train_ids"],
            size=self.dataset_data["X"].shape[0],
        )
        self.data.val_mask = index_to_mask(
            self.dataset_data["split_" + str(split)]["val_ids"],
            size=self.dataset_data["X"].shape[0],
        )
        self.data.test_mask = index_to_mask(
            self.dataset_data["split_" + str(split)]["test_ids"],
            size=self.dataset_data["X"].shape[0],
        )

        sym_norm_graph = self.dataset_data[adj_type].astype(np.float32)
        ADJ = normalize_sparse_graph(sym_norm_graph, -0.5, -0.5)

        # EVD
        if self.name in ["ogbn_arxiv", "flickr"]:
            Ux, Sx, _ = compute_or_load_svd(
                self.name, self.dataset_data["X"], dims=2048, key="X"
            )
            Ua_L = np.array(
                h5py.File(
                    osp.join(self.data_dir, "EVD_Sym_SymNrml_{}.mat".format(self.name))
                )["UE_L"]
            ).T
            Sa_L = np.diag(
                h5py.File(
                    osp.join(self.data_dir, "EVD_Sym_SymNrml_{}.mat".format(self.name))
                )["SE_L"]
            )

            Ua_H = np.array(
                h5py.File(
                    osp.join(self.data_dir, "EVD_Sym_SymNrml_{}.mat".format(self.name))
                )["UE_H"]
            ).T
            Sa_H = np.diag(
                h5py.File(
                    osp.join(self.data_dir, "EVD_Sym_SymNrml_{}.mat".format(self.name))
                )["SE_H"]
            )

            C_L = Ua_L.T @ Ux
            C_H = Ua_H.T @ Ux

            U = np.concatenate((Ua_L, Ua_H), 1)
            C = np.concatenate((C_L, C_H), 0)

            Sa_L = Sa_L.reshape(-1, 1).astype(np.float32)
            Sa_H = Sa_H.reshape(-1, 1).astype(np.float32)
            Sx = Sx.reshape(-1, 1).astype(np.float32)
            SA = np.concatenate((Sa_L, Sa_H), 0)

        else:
            Ua, Sa, _ = compute_or_load_ev(self.name, ADJ.todense(), key="A")
            Ux, Sx, _ = compute_or_load_svd(
                self.name, self.dataset_data["X"], dims=2048, key="X"
            )
            Va = Ua

            C = Va.T @ Ux

            idx = Sa.argsort()[::-1]
            Ua = Ua[:, idx]
            Sa = Sa[idx]

            K1 = min(all_args.evd_dims, int(self.features.shape[0] / 2))
            Ua_L = Ua[:, :K1]
            Sa_L = Sa[:K1]

            Ua_H = Ua[:, -K1:]
            Sa_H = Sa[-K1:]

            C_L = Ua_L.T @ Ux
            C_H = Ua_H.T @ Ux

            U = np.concatenate((Ua_L, Ua_H), 1)
            C = np.concatenate((C_L, C_H), 0)

            Sa_L = Sa_L.reshape(-1, 1).astype(np.float32)
            Sa_H = Sa_H.reshape(-1, 1).astype(np.float32)
            Sx = Sx.reshape(-1, 1).astype(np.float32)
            SA = np.concatenate((Sa_L, Sa_H), 0)

            Sa_L_tensor = torch.tensor(Sa_L, dtype=torch.float)
            Sa_H_tensor = torch.tensor(Sa_H, dtype=torch.float)

            self.data.Sa_L = Sa_L_tensor
            self.data.Sa_H = Sa_H_tensor

        U_tensor = torch.tensor(U, dtype=torch.float)
        Sx_tensor = torch.tensor(Sx, dtype=torch.float)
        C_tensor = torch.tensor(C, dtype=torch.float)
        SA_tensor = torch.tensor(SA, dtype=torch.float)

        self.data.Ua = U_tensor
        self.data.Sx = Sx_tensor
        self.data.C = C_tensor
        self.data.SA = SA_tensor

        # Create bucketed Sa
        total_buckets = 0
        step_size = max(1, int(math.floor(U.shape[1] / all_args.total_buckets)))
        tileVals = []
        left_indices = []
        right_indices = []
        bucketed_SA = []
        for i in range(0, U.shape[1], step_size):
            total_buckets += 1
            tileVals.append(len(SA[i : i + step_size]))
            bucketed_SA.append(
                torch.tensor(
                    np.array(SA[i : i + step_size]).astype(np.float32),
                    dtype=torch.float,
                )
            )

            left_indices.append(i + step_size - 1)
            right_indices.append(i + step_size)

        bucketed_Sa_tile = torch.tensor(
            np.array(tileVals).astype(np.int32), dtype=torch.long
        )
        left_indices = torch.tensor(
            np.array(left_indices).astype(np.int32), dtype=torch.long
        )
        right_indices = torch.tensor(
            np.array(right_indices).astype(np.int32), dtype=torch.long
        )
        self.bucketed_SA = bucketed_SA
        self.U = U

        self.data.bucketed_Sa_tile = bucketed_Sa_tile
        self.data.bucketed_Sa_vals = self.bucketed_SA


def DataLoader(name):
    if "cSBM_data" in name:
        path = "../data/"
        dataset = dataset_ContextualSBM(path, name=name)
    else:
        name = name.lower()

    if name in ["cora", "citeseer", "pubmed"]:
        root_path = "../"
        path = osp.join(root_path, "data", name)
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
    elif name in ["computers", "photo"]:
        root_path = "../"
        path = osp.join(root_path, "data", name)
        dataset = Amazon(path, name, T.NormalizeFeatures())
    elif name in ["chameleon", "film", "squirrel"]:
        dataset = dataset_heterophily(
            root="../data/", name=name, transform=T.NormalizeFeatures()
        )
    elif name in ["texas", "cornell"]:
        dataset = WebKB(root="../data/", name=name, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f"dataset {name} not supported in dataloader")

    return dataset


def DataLoaderNew(name, all_args, split=0, adj_type="sym"):
    dataset = Dataset(
        name=name,
        all_args=all_args,
        split=split,
        adj_type=adj_type,
        transform=T.NormalizeFeatures(),
    )

    return dataset


def normalize_sparse_graph(bipartite_graph, gamma, beta):
    b_graph = bipartite_graph.tocsr().copy()
    r_graph = b_graph.copy()
    c_graph = b_graph.copy()

    row_sums = []
    for i in range(bipartite_graph.shape[0]):
        row_sum = r_graph.data[r_graph.indptr[i] : r_graph.indptr[i + 1]].sum()
        if row_sum == 0:
            row_sums.append(0.0)
        else:
            row_sums.append(row_sum**gamma)

    c_graph = c_graph.tocsc()
    col_sums = []
    for i in range(bipartite_graph.shape[1]):
        col_sum = c_graph.data[c_graph.indptr[i] : c_graph.indptr[i + 1]].sum()

        if col_sum == 0:
            col_sums.append(0.0)
        else:
            col_sums.append(col_sum**beta)

    for i in range(bipartite_graph.shape[0]):
        if row_sums[i] != 0:
            b_graph.data[r_graph.indptr[i] : r_graph.indptr[i + 1]] *= row_sums[i]

    b_graph = b_graph.tocsc()
    for i in range(bipartite_graph.shape[1]):
        if col_sums[i] != 0:
            b_graph.data[c_graph.indptr[i] : c_graph.indptr[i + 1]] *= col_sums[i]
    return b_graph


def compute_or_load_svd(dataset_str, data, dims=2048, key="A"):
    """
    Computes or Loads SVD if cached.
    By default computes SVD with max of 2048 dimension
    """

    cache_path = "_cache/svd/%s/" % dataset_str
    cache = cache_path + "%s_%d.pkl" % (key, dims)

    if os.path.exists(cache):
        dump = pd.read_pickle(cache)
        U, S, VT = dump["U"], dump["S"], dump["VT"]

    else:
        U, S, VT = randomized_svd(data, dims, random_state=123)

        dump = {"U": U, "S": S, "VT": VT}

    os.makedirs(cache_path, exist_ok=True)
    pkl.dump(dump, open(cache, "wb"))

    return U, S, VT


def compute_or_load_ev(dataset_str, data, key="A"):
    """
    Computes or Loads SVD if cached.
    By default computes SVD with max of 2048 dimension
    """
    cache_path = "_cache/ev/%s/" % dataset_str
    cache = cache_path + "%s.pkl" % key

    if os.path.exists(cache):
        dump = pd.read_pickle(cache)
        U, S = dump["eigenvectors"], dump["eigenvalues"]
    else:
        S, U = LA.eigh(data)
        dump = {"eigenvectors": U, "eigenvalues": S}
        os.makedirs(cache_path, exist_ok=True)
        pkl.dump(dump, open(cache, "wb"))
    return U, S, None

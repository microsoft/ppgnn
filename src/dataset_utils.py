#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import math
import os
import os.path as osp
import pickle as pkl
from typing import DefaultDict

import h5py
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from torch_geometric.io import read_npz
from torch_geometric.nn import APPNP
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_sparse import coalesce


def index_to_mask(index, size):
    """Converts index to mask"""
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def compute_or_load_ev(
    dataset_name: str, data: np.ndarray, key: str = "A"
) -> tuple([np.ndarray, np.ndarray]):
    """Computes or Loads EVD if cached"""

    cache_path = "_cache/ev/%s/" % dataset_name
    cache = cache_path + "%s.pkl" % key

    if os.path.exists(cache):
        dump = pd.read_pickle(cache)
        U, S = dump["eigenvectors"], dump["eigenvalues"]
    else:
        S, U = LA.eigh(data)
        dump = {"eigenvectors": U, "eigenvalues": S}
        os.makedirs(cache_path, exist_ok=True)
        pkl.dump(dump, open(cache, "wb"))
    return U, S


def normalize_sparse_graph(
    graph: scipy.sparse.csr_matrix, gamma: float = -0.5, beta: float = -0.5
) -> scipy.sparse.csr_matrix:
    """Normalize a sparse graph using via the gamma and beta coefficients."""

    b_graph = graph.tocsr().copy()
    r_graph = b_graph.copy()
    c_graph = b_graph.copy()

    row_sums = []
    for i in range(graph.shape[0]):
        row_sum = r_graph.data[r_graph.indptr[i] : r_graph.indptr[i + 1]].sum()
        if row_sum == 0:
            row_sums.append(0.0)
        else:
            row_sums.append(row_sum**gamma)

    c_graph = c_graph.tocsc()
    col_sums = []
    for i in range(graph.shape[1]):
        col_sum = c_graph.data[c_graph.indptr[i] : c_graph.indptr[i + 1]].sum()

        if col_sum == 0:
            col_sums.append(0.0)
        else:
            col_sums.append(col_sum**beta)

    for i in range(graph.shape[0]):
        if row_sums[i] != 0:
            b_graph.data[r_graph.indptr[i] : r_graph.indptr[i + 1]] *= row_sums[i]

    b_graph = b_graph.tocsc()
    for i in range(graph.shape[1]):
        if col_sums[i] != 0:
            b_graph.data[c_graph.indptr[i] : c_graph.indptr[i + 1]] *= col_sums[i]
    return b_graph


class Dataset:
    """This class is used to load the dataset and also obtain the train, val
    and test splits, along with the eigenvalues for the PPGNN model."""

    def __init__(
        self,
        dataset_name: str,
        split: int = 0,
        adj_type: str = "sym",
        evd_dims: int = 2048,
        total_buckets: int = 10,
        data_dir="../datasets",
    ) -> None:

        self.name = dataset_name.lower()
        self.split = split
        self.dataset_data = pkl.load(open(osp.join(data_dir, self.name + ".pkl"), "rb"))

        # Loading the Dataset
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

        self.test_ids = self.dataset_data["split_" + str(split)]["test_ids"]

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
        A = normalize_sparse_graph(sym_norm_graph, -0.5, -0.5)

        # Obtain the top and bottom K eigenvalues and eigenvectors of the
        # normalized graph. Note that U denotes the eigenvector, eig denotes
        # the eigenvalues.
        if self.name in ["ogbn_arxiv", "flickr"]:
            U_L = np.array(
                h5py.File(
                    osp.join(
                        self.data_dir,
                        "EVD_Sym_SymNrml_{}.mat".format(self.name),
                    )
                )["UE_L"]
            ).T
            eig_L = np.diag(
                h5py.File(
                    osp.join(
                        self.data_dir,
                        "EVD_Sym_SymNrml_{}.mat".format(self.name),
                    )
                )["SE_L"]
            )

            U_H = np.array(
                h5py.File(
                    osp.join(
                        self.data_dir,
                        "EVD_Sym_SymNrml_{}.mat".format(self.name),
                    )
                )["UE_H"]
            ).T
            eig_H = np.diag(
                h5py.File(
                    osp.join(
                        self.data_dir,
                        "EVD_Sym_SymNrml_{}.mat".format(self.name),
                    )
                )["SE_H"]
            )

        else:
            U, eig = compute_or_load_ev(self.name, A.todense(), key="A")
            idx = eig.argsort()[::-1]
            U = U[:, idx]
            eig = eig[idx]

            K1 = min(evd_dims, int(self.features.shape[0] / 2))
            U_L = U[:, :K1]
            eig_L = eig[:K1]

            U_H = U[:, -K1:]
            eig_H = eig[-K1:]

        # Collect the top and bottom K eigenvectors and eigenvalues
        U_LH = np.concatenate((U_L, U_H), 1)
        eig_L = eig_L.reshape(-1, 1).astype(np.float32)
        eig_H = eig_H.reshape(-1, 1).astype(np.float32)
        eig_LH = np.concatenate((eig_L, eig_H), 0)

        # Create bucketed eig
        step_size = max(1, int(math.floor(U.shape[1] / total_buckets)))
        bucketed_eig = []
        for i in range(0, U.shape[1], step_size):
            total_buckets += 1
            bucketed_eig.append(
                torch.tensor(
                    np.array(eig_LH[i : i + step_size]).astype(np.float32),
                    dtype=torch.float,
                )
            )

        self.data.bucketed_eig_vals = bucketed_eig
        self.data.U_LH = torch.tensor(U_LH, dtype=torch.float)
        self.data.eig_LH = torch.tensor(eig_LH, dtype=torch.float)
        self.data.U = torch.tensor(U, dtype=torch.float)
        self.data.eig = torch.tensor(eig, dtype=torch.float)

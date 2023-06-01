#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from collections import Counter

import torch
import math
import numpy as np


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    dataset_str, split_number = "texas", 9
    split_data = np.load(
        "/VILI/Attention/NeurIPS/Codes/splits/{}_split_0.6_0.2_{}.npz".format(
            dataset_str, split_number
        )
    )

    if Flag is 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        # data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        # data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        # data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)

        data.train_mask = torch.BoolTensor(
            split_data["train_mask"]
        )  # index_to_mask(train_nodes, size=data.num_nodes)
        data.val_mask = torch.BoolTensor(
            split_data["val_mask"]
        )  # index_to_mask(val_nodes, size=data.num_nodes)
        data.test_mask = torch.BoolTensor(
            split_data["test_mask"]
        )  # index_to_mask(test_nodes, size=data.num_nodes)

    else:
        val_index = torch.cat(
            [i[percls_trn : percls_trn + val_lb] for i in indices], dim=0
        )
        rest_index = torch.cat([i[percls_trn + val_lb :] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

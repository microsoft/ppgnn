"""Code borrowed from the GPRGNN repository: 
    https://github.com/jianhao2016/GPRGNN.git. All credits go to the original 
    authors."""


import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from numpy.core.numeric import True_
from torch.nn import Linear, Parameter
from torch_geometric.nn import (
    APPNP,
    ChebConv,
    GATConv,
    GCNConv,
    JumpingKnowledge,
    MessagePassing,
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, mul, sum


def gcn_norm(
    edge_index: torch.Tensor,
    norm: str = "sym",
    edge_weight: torch.Tensor = None,
    num_nodes: int = None,
    improved: bool = False,
    add_self_loops: bool = True,
    dtype: str = None,
) -> tuple([torch.Tensor, torch.Tensor]):
    """GCN normalization"""

    fill_value = 2.0 if improved else 1.0
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0, dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes
            )
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

        if norm == "no":
            return edge_index, edge_weight
        elif norm == "row":
            deg_inv_sqrt = deg.pow_(-1.0)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
            normA = deg_inv_sqrt[row] * edge_weight
            return edge_index, normA
        elif norm == "sym":
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
            normA = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
            return edge_index, normA


def get_init(Init, alpha, K):
    assert Init in ["SGC", "PPR", "NPPR", "Random", "WS"]
    if Init == "SGC":
        # SGC-like
        TEMP = 0.0 * np.ones(K + 1)
        TEMP[alpha] = 1.0
    elif Init == "PPR":
        # PPR-like
        TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
        TEMP[-1] = (1 - alpha) ** K
    elif Init == "NPPR":
        # Negative PPR
        TEMP = (alpha) ** np.arange(K + 1)
        TEMP = TEMP / np.sum(np.abs(TEMP))
    elif Init == "Random":
        # Random
        bound = np.sqrt(3 / (K + 1))
        TEMP = np.random.uniform(-bound, bound, K + 1)
        TEMP = TEMP / np.sum(np.abs(TEMP))
    return TEMP


class GPR_prop(MessagePassing):
    """Propagation class for GPR_GNN"""

    def __init__(self, K, alpha, Init, norm, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.norm = norm

        assert Init in ["SGC", "PPR", "NPPR", "Random", "WS"]
        if Init == "SGC":
            # SGC-like
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == "PPR":
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == "NPPR":
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == "Random":
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == "WS":
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index,
            self.norm,
            edge_weight,
            num_nodes=x.size(0),
            dtype=x.dtype,
        )

        hidden = x * (self.temp[0])

        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return "{}(K={}, temp={})".format(self.__class__.__name__, self.K, self.temp)


class GPRGNN(torch.nn.Module):
    """Note that this implementation does not contain the log_softmax in the end.
        This is because it is being used in PPGNN. If you want to use it as 
        a standalone model, please add log_softmax in the end."""
        
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        if args.ppnp == "PPNP":
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == "GPR_prop":
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.norm)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x


class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 32, K=2)
        self.conv2 = ChebConv(32, dataset.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout,
        )
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout,
        )
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN_JKNet(torch.nn.Module):
    def __init__(self, dataset, args):
        in_channels = dataset.num_features
        out_channels = dataset.num_classes

        super(GCN_JKNet, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin1 = torch.nn.Linear(16, out_channels)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode="lstm", channels=16, num_layers=4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)

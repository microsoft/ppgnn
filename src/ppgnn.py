import torch
import argparse
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from baselines import GPRGNN, get_init
from dataset_utils import Dataset


class PieceWisePoly(torch.nn.Module):
    """Class that computes the polynomial on the lower and higher end of the
    eigenvalue spectrum. Note that the polynomial is computed on the
    reconstructed graph from the eigenvalues of the lower and higher
    eigenvalues. Note that the features are transformed using the same
    initial transformation as the GPRGNN model."""

    def __init__(
        self,
        K: int,
        alphas: list,
        Init: str,
        dataset: Dataset,
        args: argparse,
        gpr: GPRGNN,
    ) -> None:
        super(PieceWisePoly, self).__init__()
        self.K = K
        self.Init = Init
        self.dataset = dataset
        self.dropout = args.dropout
        self.gpr = gpr

        self.polynomials = torch.nn.ParameterList()

        for _ in range(len(dataset.data.bucketed_eig_vals)):
            self.polynomials.append(Parameter(torch.tensor(get_init(Init, alphas, K))))

    def forward(self, data: Dataset) -> torch.Tensor:

        # Computes the polynomial on the first bucket
        bucketed_poly = self.polynomials[0][1] * data.bucketed_eig_vals[0]
        for k in range(1, self.K):
            bucketed_poly += self.polynomials[0][k + 1] * torch.pow(
                data.bucketed_eig_vals[0], k + 1
            )
        bucketed_poly += self.polynomials[0][0]  # Add the constant term
        adapted_eigs = bucketed_poly

        # Computes the polynomial on the other buckets
        for poly in range(1, len(self.polynomials)):
            bucketed_poly = self.polynomials[poly][1] * data.bucketed_eig_vals[poly]
            for k in range(1, self.K):
                bucketed_poly += self.polynomials[poly][k + 1] * torch.pow(
                    data.bucketed_eig_vals[poly], k + 1
                )
            bucketed_poly += self.polynomials[poly][0]  # Add the constant term

            adapted_eigs = torch.cat((adapted_eigs, bucketed_poly), 0)

        # Reconstruction of the graph from the adapted eigenvalues
        diag_eigs = torch.diag(torch.squeeze(adapted_eigs))
        constructed_A = torch.mm(torch.mm(data.U_LH, diag_eigs), data.U_LH.T)

        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = F.relu(self.gpr.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gpr.lin2(x)

        x = torch.mm(constructed_A, x)

        return x


class PPGNN(torch.nn.Module):
    """Class that implements the PPGNN model. This is done by combining the
    GPRGNN model and the PieceWisePoly model. Note that this codebase
    was built on top of the codebase of the GPRGNN model."""

    def __init__(self, dataset: Dataset, args: argparse) -> None:
        super(PPGNN, self).__init__()
        self.args = args
        self.gprnn_model = GPRGNN(dataset, args)
        self.piecewise_model = PieceWisePoly(
            args.K, args.alphas, args.Init, dataset, args, self.gprnn_model
        )

    def forward(self, data: Dataset) -> torch.Tensor:
        if self.args.beta == 0:
            gpr_out = self.gprnn_model(data)
            return F.log_softmax(gpr_out, dim=1)
        elif self.args.beta == 1:
            piecewise_out = self.piecewise_model(data)
            return F.log_softmax(piecewise_out, dim=1)
        else:
            gpr_out = self.gprnn_model(data)
            piecewise_out = self.piecewise_model(data)
            return F.log_softmax(
                self.args.beta * piecewise_out + (1 - self.args.beta) * gpr_out, dim=1
            )

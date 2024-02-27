import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sn_sage_cov import SN_SAGEConv, SAGEConv
from torch.nn.utils import spectral_norm
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class GraphSAGE(nn.Module):
    """GraphSAGE"""

    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        # SN_SAGEConv implement the spectral normalization (SN), while SAEGConv is the one without SN.
        self.sage1 = SN_SAGEConv(dim_in, dim_h, aggr='max', normalize=True)
        self.sage2 = SN_SAGEConv(dim_h, dim_h, aggr='max', normalize=True)  # Pay attention to the dim_out when connecting to GP

    def forward(self, x, edge_index):
        h = F.relu(self.sage1(x, edge_index))
        h = F.dropout(h, 0.1, training=self.training)
        h = F.relu(self.sage2(h, edge_index))
        h = F.dropout(h, 0.1, training=self.training)

        return h


class GPR_1(ApproximateGP):
    def __init__(self, num_inducing, dim_latent):
        inducing_points = torch.rand(num_inducing, dim_latent) #Added
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPR_1, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPR_0(ApproximateGP):
    def __init__(self, num_inducing, dim_latent):
        inducing_points = torch.rand(num_inducing, dim_latent) #Added
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPR_0, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GraphDKL(gpytorch.Module):  # Need to be changed to gpytorch.Module if GP is used, otherwise nn.Module
    def __init__(self, feature_extractor, num_inducing, latent_dim):
        super(GraphDKL, self).__init__()
        self.gnn = feature_extractor  # This GNN model for feature extraction has been initialized outside

        self.gp_1 = GPR_1(num_inducing, latent_dim)
        self.gp_0 = GPR_0(num_inducing, latent_dim)

        # Neural Net Decoder
        self.fc_y1_pred = nn.Sequential(
                                        spectral_norm(nn.Linear(100, 100)), nn.ReLU(),  # nn.Dropout(0.1),
                                        spectral_norm(nn.Linear(100, latent_dim)), nn.ReLU()
#                                        nn.Linear(100,100), nn.ReLU(),
#                                        nn.Linear(100, latent_dim), nn.ReLU(),
                                        )

        self.fc_y0_pred = nn.Sequential(
                                        spectral_norm(nn.Linear(100, 100)), nn.ReLU(),  # nn.Dropout(0.1),
                                        spectral_norm(nn.Linear(100, latent_dim)), nn.ReLU()
#                                        nn.Linear(100,100), nn.ReLU(),
#                                        nn.Linear(100, latent_dim), nn.ReLU(),
                                        )

    def forward(self, x, edge_index, t):
        latent = self.gnn(x, edge_index)  # latent representation learning

        latent_1 = self.fc_y1_pred(latent)
        latent_0 = self.fc_y0_pred(latent)

        # Gaussian Process Regressors
        pred_1 = self.gp_1(latent_1)   # GP output before warpping with the likelihood function
        pred_0 = self.gp_0(latent_0)

        return pred_1, pred_0, latent
 

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import gpytorch
import torch
import torch.optim as optim
from models.graphdkl import GraphSAGE, GraphDKL
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.FloatTensor
LongTensor = torch.LongTensor


def load_data(path, name='BlogCatalog',exp_id='0',original_X = False, extra_str=""):

	data = sio.loadmat(path+name+extra_str+'/'+name+exp_id+'.mat')
	A = data['Network'] #csr matrix


	if not original_X:
		X = data['X_100']
	else:
		X = data['Attributes']

	Y1 = data['Y1']
	Y0 = data['Y0']
	T = data['T']

	return X, A, T, Y1, Y0


def wasserstein(x,y,p=0.5,lam=10,its=10,sq=False,backpropT=False,cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]
    
    x = x.squeeze()
    y = y.squeeze()
    
#    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x,y) #distance_matrix(x,y,p=2)
    
    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M,10.0/(nx*ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta*torch.ones(M[0:1,:].shape)
    col = torch.cat([delta*torch.ones(M[:,0:1].shape),torch.zeros((1,1))],0)
    if cuda:
        row = row.cuda()
        col = col.cuda()
    Mt = torch.cat([M,row],0)
    Mt = torch.cat([Mt,col],1)

    '''compute marginal'''
    a = torch.cat([p*torch.ones((nx,1))/nx,(1-p)*torch.ones((1,1))],0)
    b = torch.cat([(1-p)*torch.ones((ny,1))/ny, p*torch.ones((1,1))],0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1)*1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K/a

    u = a

    for i in range(its):
        u = 1.0/(ainvK.matmul(b/torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.cuda()
    v = b/(torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.cuda()

    upper_t = u*(torch.t(v)*K).detach()

    E = upper_t*Mt
    D = 2*torch.sum(E)

    if cuda:
        D = D.cuda()

    return D, Mlam


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def trt_ctr(treatment):
    list1, list0 = [], []
    for index, i in enumerate(treatment):
        if i == 1:
            list1.append(index)
        elif i == 0:
            list0.append(index)
        else:
            pass
    return list1, list0


def initialization(experiment, data_path, data_name, k, train_size, learning_rate, weight_decay, seed, status):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Load data and init models
    # X, A, T, Y1, Y0 = utils.load_data(args.path, name=args.dataset, original_X=False, exp_id=str(experiment), extra_str=args.extrastr)
    X, A, T, Y1, Y0 = load_data(data_path, name=data_name, original_X=False, exp_id=str(experiment), extra_str=k)

    n = X.shape[0]
    n_train = int(n * (train_size + 0.2))
    n_test = int(n * 0.2)

    idx = np.random.permutation(n)
    idx_train, idx_test = idx[:n_train], idx[n_train:]

    idx_ = np.random.permutation(n_train)
    n_training = int(n_train * 0.75)
    idx_training, idx_val = idx_[:n_training], idx_[n_training:]

    X = X.todense()
    X = Tensor(X)

    X = scaler.fit_transform(X)  # Do standardization instead of the default row_norm
    X = Tensor(X)  # Transfer back to tensor to get aligned

    Y1 = Tensor(np.squeeze(Y1))
    Y0 = Tensor(np.squeeze(Y0))
    T = LongTensor(np.squeeze(T))

    idx_train = LongTensor(idx_train)
    idx_val = LongTensor(idx_val)
    idx_test = LongTensor(idx_test)

    # Create GraphSAGE instance
    dim_in = X.shape[1]
    dim_h = 100
    dim_out = 20
    graphsage = GraphSAGE(dim_in, dim_h, dim_out, status)

    num_inducing = int(0.95 * 0.6 * X.shape[0])
    model = GraphDKL(graphsage, num_inducing, dim_out, status)

    likelihood_1 = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_0 = gpytorch.likelihoods.GaussianLikelihood()

    optimizer = optim.Adam([{'params': model.gnn.parameters()},
                            {'params': model.fc_y1_pred.parameters()},
                            {'params': model.fc_y0_pred.parameters()},  # previously not added for optimize (25th June)
                            {'params': model.gp_1.hyperparameters(), 'lr': learning_rate * 0.5},
                            {'params': model.gp_1.variational_parameters()},
                            {'params': model.gp_0.hyperparameters(), 'lr': learning_rate * 0.5},
                            {'params': model.gp_0.variational_parameters()},
                            {'params': likelihood_1.parameters()},
                            {'params': likelihood_0.parameters()}], lr=learning_rate, weight_decay=weight_decay)

    return X.to(device), A, T.to(device), Y1.to(device), Y0.to(
        device), idx_train, idx_training, idx_val, idx_test, model.to(device), optimizer, likelihood_1.to(
        device), likelihood_0.to(device)

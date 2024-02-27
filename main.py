import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import utils
import csv
import gpytorch


# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='BlogCatalog') #Flickr

parser.add_argument('--extrastr', type=str, default='0.5')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')

parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')

parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')

parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units.')

parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')

parser.add_argument('--tr', type=float, default=0.6)

parser.add_argument('--path', type=str, default='./datasets/')

parser.add_argument('--normy', type=int, default=1)

parser.add_argument('--gamma', type=float, default=0.1)

parser.add_argument('--mode', type=str, default='WithOSN')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
loss = torch.nn.MSELoss()


def train(epoch, X, A_train, A_training, T, Y1, Y0, idx_training, idx_val):
    t = time.time()
    
    model.train()
    likelihood_1.train()
    likelihood_0.train()

    optimizer.zero_grad()

    pred_1, pred_0, rep = model(X[idx_training], A_training, T[idx_training])

    YF = torch.where(T>0,Y1,Y0)

    if args.normy:
        # recover the normalized outcomes
        ym, ys = torch.mean(YF), torch.std(YF)
        YFtr, YFva = (YF[idx_training] - ym) / ys, (YF[idx_val] - ym) / ys
    else:
        YFtr = YF[idx_train]
        YFva = YF[idx_val]

    #Loss for GP regressors
    T_train = T[idx_training]
    YFtr_1 = YFtr[T_train==1]
    YFtr_0 = YFtr[T_train==0]

    list_1, list_0 = utils.trt_ctr(T_train)

    loss_train = - mll_1(pred_1[list_1], YFtr_1) - mll_0(pred_0[list_0], YFtr_0) #+ 1 * dist
    loss_train.backward()
    optimizer.step()

    if epoch%10==0:
        
        # validation for GP regressors
        T_val = T[idx_val]
        YFva_1 = YFva[T_val == 1]
        YFva_0 = YFva[T_val == 0]

        list_1_val, list_0_val = utils.trt_ctr(T_val)

        pred_1, pred_0, rep = model(X, A_train, T)
        loss_val = - mll_1(pred_1[idx_val][list_1_val], YFva_1) - mll_0(pred_0[idx_val][list_0_val], YFva_0) #+ 1 * dist

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


def evaluation(X, A, T, Y1, Y0, idx_train, idx_test):
    model.eval()
    likelihood_1.eval()
    likelihood_0.eval()

    pred_1, pred_0, rep = model(X, A, T)
    pred_1_cf, pred_0_cf, _ = model(X, A, 1 - T)

    yf_pred = torch.where(T > 0, pred_1.mean, pred_0.mean)
    ycf_pred = torch.where(1-T > 0, pred_1_cf.mean, pred_0_cf.mean)

    YF = torch.where(T>0,Y1,Y0)
    YCF = torch.where(T>0,Y0,Y1)

    ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])

    y1_pred, y0_pred = torch.where(T>0,yf_pred,ycf_pred), torch.where(T>0,ycf_pred,yf_pred)

    if args.normy:
        y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

    pehe_ts = torch.sqrt(loss((y1_pred - y0_pred)[idx_test],(Y1 - Y0)[idx_test]))
    mae_ate_ts = torch.abs(torch.mean((y1_pred - y0_pred)[idx_test])-torch.mean((Y1 - Y0)[idx_test]))
    
    return pehe_ts.item(), mae_ate_ts.item()


if __name__ == '__main__':

    pehes, maes = [], []

    for experiment in range(0, 10):
        print("Experiment: {}".format(experiment))
        # Train model

        (X, A, T, Y1, Y0, idx_train, idx_training, idx_val, idx_test,
         model, optimizer, likelihood_1, likelihood_0) = utils.initialization(experiment=experiment,
                                                                              data_path=args.path,
                                                                              data_name=args.dataset,
                                                                              k=args.extrastr,
                                                                              train_size=args.tr,
                                                                              learning_rate=args.lr,
                                                                              weight_decay=args.weight_decay)

        t_total = time.time()

        indices = idx_train
        A_train_ = A[indices][:, indices]  # Add _ to indicate matrix  form
        A_matrix_train = A_train_.tocoo()
        row_train, col_train, edge_attr_train = A_matrix_train.row, A_matrix_train.col, A_matrix_train.data
        edge_index_train = torch.stack([torch.from_numpy(row_train), torch.from_numpy(col_train)], dim=0).to(device)
        A_train = edge_index_train.long()

        indices_ = idx_training
        A_training_ = A_train_[indices_][:, indices_] # Should inherit matrix form
        A_matrix_training = A_training_.tocoo()
        row_training, col_training, edge_attr_training = A_matrix_training.row, A_matrix_training.col, A_matrix_training.data
        edge_index_training = torch.stack([torch.from_numpy(row_training), torch.from_numpy(col_training)], dim=0).to(device)
        A_training = edge_index_training.long()

        A_matrix = A.tocoo()
        row, col, edge_attr = A_matrix.row, A_matrix.col, A_matrix.data
        edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)], dim=0).to(device)
        A = edge_index.long()

        scheduler = MultiStepLR(optimizer, milestones=[0.5 * args.epochs, 0.75 * args.epochs], gamma=args.gamma)
        mll_1 = gpytorch.mlls.VariationalELBO(likelihood_1, model.gp_1, X[idx_train].shape[0])
        mll_0 = gpytorch.mlls.VariationalELBO(likelihood_0, model.gp_0, X[idx_train].shape[0])
        
        for epoch in range(args.epochs):
            train(epoch, X=X[idx_train], A_train=A_train, A_training=A_training, T=T[idx_train], Y1=Y1[idx_train], Y0=Y0[idx_train], idx_training=idx_training, idx_val = idx_val) # model, optimizer, likelihood_1, likelihood_0)
            scheduler.step()

        print("Total time elapsed: {:.2f} mins".format(round((time.time() - t_total)/60, 2)))

        # Testing
        # Note, that during testing, we can have access to the training information as a complete graph is present, but during traning
        # we have no access to the test info, that means, we split the training data from the whole graph (original dataset).
        pehe, mae = evaluation(X, A, T, Y1, Y0, idx_train, idx_test) 
        pehes.append(pehe)
        maes.append(mae)

        print('PEHE:', pehe)

        # Saving trained model into local
        state_dict = model.state_dict()
        likelihood_1_state_dict = likelihood_1.state_dict()
        likelihood_0_state_dict = likelihood_0.state_dict()

        torch.save({'model': state_dict, 'likelihood_1': likelihood_1_state_dict, 'likelihood_0': likelihood_0_state_dict},
                    'trained_models/inductive/{}/GraphDKL_{}{}_{}.dat'.format(args.mode, args.dataset, args.extrastr, experiment))
        

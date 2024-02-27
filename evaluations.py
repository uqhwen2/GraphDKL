
import time
import argparse
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from models.graphdkl import GraphSAGE, GraphDKL
import utils
import csv
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import gpytorch
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


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

parser.add_argument('--threshold', type=float, default=0.95)

parser.add_argument('--mode', type=str, default='WithOSN')

parser.add_argument('--count', type=int, default=1)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)

loss = torch.nn.MSELoss()

def eva(X, A, T, Y1, Y0, idx_train, idx_test):
    model.eval()

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

    yf_pred_variance = torch.where(T > 0, pred_1.variance, pred_0.variance)
    ycf_pred_variance = torch.where(1 - T > 0, pred_1_cf.variance, pred_0_cf.variance)
    y1_pred_variance, y0_pred_variance = torch.where(T > 0, yf_pred_variance, ycf_pred_variance), torch.where(T > 0,
                                                                                                              ycf_pred_variance,
                                                                                                              yf_pred_variance)
    uncertainty = ((y1_pred_variance.sqrt() + y0_pred_variance.sqrt())[idx_test]) * 0.5 * ys
    draw_dist = uncertainty.cpu().detach().numpy()
    # Get the x-th percentile of the array
    quantile_threshold = np.quantile(draw_dist, args.threshold)

    lower = []
    errors = ((y1_pred - y0_pred)[idx_test] - (Y1 - Y0)[idx_test]) ** 2
    for idx, i in enumerate(errors):
        # print(round(i.item(),2), round(uncertainty[idx].item(),2), round(uncertainty[idx].item()/uncertainty_mean.item(),2))
        if uncertainty[idx].item() <= quantile_threshold:
            lower.append(idx)

    pehe_ts = torch.sqrt(loss((y1_pred - y0_pred)[idx_test],(Y1 - Y0)[idx_test]))
    pruned_pehe_ts = torch.sqrt(loss((y1_pred - y0_pred)[idx_test][lower], (Y1 - Y0)[idx_test][lower]))
    mae_ate_ts = torch.abs(torch.mean((y1_pred - y0_pred)[idx_test])-torch.mean((Y1 - Y0)[idx_test]))

    # CATE_pred
    cate_pred = torch.round(y1_pred - y0_pred)
    # CATE_true
    cate_true = Y1 - Y0

    recommendation_pred = cate_pred[idx_test][lower] >= 0.0
    #print('Recommendation for train:', torch.sum(recommendation_pred), len(recommendation_pred))
    recommendation_true = cate_true[idx_test][lower] >= 0.0
    rec_errors = recommendation_pred != recommendation_true
    
    return pehe_ts.item(), mae_ate_ts.item(), pruned_pehe_ts.item(), torch.sum(rec_errors).cpu().detach().numpy()


if __name__ == '__main__':

    pehes, maes, pruned_pehes, error_props = [], [], [], []

    for experiment in range(0, 10): # simulation 0 is different from the rest, can cause some nan problem when estimating.

        # Train model
        (X, A, T, Y1, Y0, idx_train, _, _, idx_test,
         model, _, _, _) = utils.initialization(experiment=experiment,
                                                data_path=args.path,
                                                data_name=args.dataset,
                                                k=args.extrastr,
                                                train_size=args.tr,
                                                learning_rate=args.lr,
                                                weight_decay=args.weight_decay,
                                                seed=args.seed,
                                                status=args.mode)
        
        t_total = time.time()
       
        A_matrix = A.tocoo() 
        row, col, edge_attr = A_matrix.row, A_matrix.col, A_matrix.data
        edge_index = torch.stack([torch.from_numpy(row), torch.from_numpy(col)], dim=0).to(device)
        A = edge_index.long()

        # Define the path to the saved model
        PATH = "trained_models/inductive/{}/GraphDKL_{}{}_{}.dat".format(args.mode, args.dataset, args.extrastr, experiment)

        # Load the saved model
        checkpoint = torch.load(PATH)

        # Extract the model and likelihood state dictionaries
        model_state_dict = checkpoint['model']
        likelihood_1_state_dict = checkpoint['likelihood_1']
        likelihood_0_state_dict = checkpoint['likelihood_0']

        # Load the saved model state dictionary
        model.load_state_dict(model_state_dict)

        # Testing
        pehe, mae, pruned_pehe, error_prop = eva(X, A, T, Y1, Y0, idx_train, idx_test)
        pehes.append(pehe)
        maes.append(mae)
        pruned_pehes.append(pruned_pehe)
        error_props.append(error_prop)

    #print('mean Error Rec: ', round(sum(error_props) / len(error_props), 2))
    print('Reject {:2f}:'.format(1-args.threshold), 'mean Pruned PEHE: ', round(sum(pruned_pehes) / len(pruned_pehes),2), 'mean PEHE: ', round(sum(pehes) / len(pehes),2), 'mean MAE: ', round(sum(maes) / len(maes),2))

    # Open the CSV file for reading
    with open('{}_{}_test.csv'.format(args.dataset, args.extrastr), 'r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Update the values in the second row with zero up to the column under 0.95
    if args.mode == 'WithOSN':
        rows[1][args.count] = f"{sum(pruned_pehes) / len(pruned_pehes):.2f}"
    else:
        rows[2][args.count] = f"{sum(pruned_pehes) / len(pruned_pehes):.2f}"

    # Write the updated data back to the CSV file
    with open('{}_{}_test.csv'.format(args.dataset, args.extrastr), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

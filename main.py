from KGGraph.Dataset.molecule_dataset import MoleculeDataset
import pandas as pd
from KGGraph.Dataset.scaffold_split import scaffold_split
from torch_geometric.data import DataLoader
from KGGraph.Model.Architecture.class_gin import GINNet
from KGGraph.Model.Train.train_utils import train, evaluate, train_reg, eval_reg
from KGGraph.Model.Train.visualize import plot_metrics
import torch
import argparse
import numpy as np
from torch import optim
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--dataset', type=str, default = 'tox21', 
                        help='[bbbp, bace, sider, clintox, sider,tox21, toxcast, esol,freesolv,lipophilicity, alk]')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if args.dataset in ['tox21', 'hiv', 'pcba', 'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox', 'mutag', 'alk']:
        task_type = 'cls'
    else:
        task_type = 'reg'

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "alk":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == 'esol':
        num_tasks = 1
    elif args.dataset == 'freesolv':
        num_tasks = 1
    elif args.dataset == 'lipophilicity':
        num_tasks = 1
    elif args.dataset == 'qm7':
        num_tasks = 1
    elif args.dataset == 'qm8':
        num_tasks = 12
    elif args.dataset == 'qm9':
        num_tasks = 12
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, _ = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    # elif args.split == "random":
    #     train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #     print("random")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    if args.dataset == 'freesolv':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GINNet(num_layer=args.num_layer, out_channels=num_tasks, dropout = args.dropout_ratio)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.decay)
    print(optimizer)

    finetune_model_save_path = './model/' + args.dataset + '.pth'

    # training based on task type
    if task_type == 'cls':
        train_auc_list, test_auc_list = [], []
        for epoch in range(1, args.epochs+1):
            print('====epoch:',epoch)
            
            train(model, device, train_loader, optimizer)

            print('====Evaluation')
            if args.eval_train:
                train_auc, train_loss = evaluate(model, device, train_loader)
            else:
                print('omit the training accuracy computation')
                train_auc = 0
            val_auc, val_loss = evaluate(model, device, val_loader)
            test_auc, test_loss = evaluate(model, device, test_loader)
            test_auc_list.append(float('{:.4f}'.format(test_auc)))
            train_auc_list.append(float('{:.4f}'.format(train_auc)))

            torch.save(model.state_dict(), finetune_model_save_path)
            
            print("train_auc: %f val_auc: %f test_auc: %f" %(train_auc, val_auc, test_auc))
            print("train_loss: %f val_loss: %f test_loss: %f" %(train_loss, val_loss, test_loss))


    elif task_type == 'reg':
        train_list, test_list = [], []
        for epoch in range(1, args.epochs+1):
            print('====epoch:',epoch)
            
            train_reg(args, model, device, train_loader, optimizer)

            print('====Evaluation')
            if args.eval_train:
                train_mse, train_mae, train_rmse = eval_reg(args, model, device, train_loader)
            else:
                print('omit the training accuracy computation')
                train_mse, train_mae, train_rmse = 0, 0, 0
            val_mse, val_mae, val_rmse = eval_reg(args, model, device, val_loader)
            test_mse, test_mae, test_rmse = eval_reg(args, model, device, test_loader)
            
            if args.dataset in ['esol', 'freesolv', 'lipophilicity']:
                test_list.append(float('{:.6f}'.format(test_rmse)))
                train_list.append(float('{:.6f}'.format(train_rmse)))
                torch.save(model.state_dict(), finetune_model_save_path)

            elif args.dataset in ['qm7', 'qm8', 'qm9']:
                test_list.append(float('{:.6f}'.format(test_mae)))
                train_list.append(float('{:.6f}'.format(train_mae)))
                torch.save(model.state_dict(), finetune_model_save_path)
                
            print("train_mse: %f val_mse: %f test_mse: %f" %(train_mse, val_mse, test_mse))
            print("train_mae: %f val_mae: %f test_mae: %f" %(train_mae, val_mae, test_mae))
            print("train_rmse: %f val_rmse: %f test_rmse: %f" %(train_rmse, val_rmse, test_rmse))



if __name__ == "__main__":
    main()
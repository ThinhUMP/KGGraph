from KGGraph.Dataset.molecule_dataset import MoleculeDataset
import pandas as pd
from KGGraph.Dataset.split import scaffold_split, random_split
from torch_geometric.data import DataLoader
from KGGraph.GnnModel.Architecture import GINNet, gin
from KGGraph.GnnModel.Train.train_utils import train_epoch_cls, train_epoch_reg
from KGGraph.GnnModel.Train.visualize import plot_metrics
import torch
import argparse
import numpy as np
from torch import optim
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=0.0,
                        help='weight decay (default: 0)')
    parser.add_argument('--hidden_channels', type=int, default=2048,
                        help='number of hidden nodes in the GNN network (default: 512).')
    parser.add_argument('--num_layer', type=int, default=1,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--dataset', type=str, default = 'tox21', 
                        help='[bbbp, bace, sider, clintox, sider,tox21, toxcast, esol,freesolv,lipophilicity, alk]')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--save_fig_path', type=str, default = 'dataset/', help='path for saving training images')
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
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
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
    # model = GINNet(num_layer=args.num_layer, out_channels=num_tasks, dropout = args.dropout_ratio)
    model = gin(dim_h=args.hidden_channels, out_channels=num_tasks, dropout=args.dropout_ratio)
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.decay)
    # optimizer = optim.SGD(model.parameters(), lr= args.lr, weight_decay=args.decay)
    print(optimizer)

    model_save_path = './model/' + args.dataset + '.pth'

    # training based on task type
    if task_type == 'cls':
        metrics_training = train_epoch_cls(args, model, device, train_loader, val_loader, test_loader, optimizer, model_save_path)

    elif task_type == 'reg':
        test_mae_list = train_epoch_reg(args, model, device, train_loader, val_loader, test_loader, optimizer, model_save_path) 

    plot_metrics(args, metrics_training)
    


if __name__ == "__main__":
    main()
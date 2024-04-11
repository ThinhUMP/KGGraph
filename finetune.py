from KGGraph.KGGProcessor.molecule_dataset import MoleculeDataset
import pandas as pd
from KGGraph.KGGProcessor.split import scaffold_split, random_split
from torch_geometric.data import DataLoader
from KGGraph.KGGModel.Architecture import GINTrain
from KGGraph.KGGModel.Train.train_utils import train_epoch_cls, train_epoch_reg
from KGGraph.KGGModel.Train.visualize import plot_metrics
from KGGraph.KGGModel.Train.get_task_type_num_tasks import get_num_task, get_task_type
from KGGraph.KGGModel.Train.crawl_metrics import average_test_metrics, average_train_metrics
import torch
import torch.nn as nn
import argparse
from torch import optim
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    #set up parameters
    parser = argparse.ArgumentParser(description='PyTorch implementation of training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--training_rounds', type=int, default=3,
                        help='number of rounds to train to get the average test auc (default: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr_feat', type=float, default=0.001,
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--lr_pred', type=float, default=0.001,
                        help='learning rate for the prediction layer (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0.0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, 
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='embedding dimensions (default: 512)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.3)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin",
                        help='gnn_type (gin, gcn)')
    parser.add_argument('--decompose_type', type=str, default="motif",
                        help='decompose_type (brics, jin, motif, smotif) (default: motif).')
    parser.add_argument('--dataset', type=str, default = 'bace',
                        help='[bbbp, bace, sider, clintox, tox21, toxcast, esol, freesolv, lipophilicity]')
    parser.add_argument('--input_model_file', type=str, default = './saved_model/pretrain.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=42, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--save_path', type=str, default = 'Data/', help='path for saving training images, test_metrics csv, model')
    parser.add_argument('--GNN_different_lr', type=bool, default = True, help='if the learning rate of GNN backbone is different from the learning rate of prediction layers')
    args = parser.parse_args()
    
    for i in range(1, args.training_rounds+1):
        print("====Round ", i)
        
        #set up seeds
        torch.manual_seed(args.runseed)
        np.random.seed(args.runseed)
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.runseed)
        
        #set up device
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print('device', device)
        
        #set up task type
        task_type = get_task_type(args)

        #set up number of tasks
        num_tasks = get_num_task(args)

        #set up dataset
        dataset = MoleculeDataset("Data/" + task_type + "/" + args.dataset, dataset=args.dataset, decompose_type=args.decompose_type)
        print(dataset)
        
        #data split
        if args.split == "scaffold":
            smiles_list = pd.read_csv('Data/' + task_type + '/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset, _ = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
            print("scaffold")
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
            print("random")
        else:
            raise ValueError("Invalid split option.")

        print(train_dataset[0])

        #data loader
        if args.dataset == 'freesolv':
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

        #set up model
        model = GINTrain(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
        if not args.input_model_file == "":
            model.from_pretrained(args.input_model_file)
        
        model.to(device)

        #set up optimizer
        #different learning rate for different part of GNN
        model_param_group = []
        if args.GNN_different_lr:
            print('GNN update')
            model_param_group.append({"params": model.gnn.parameters(), "lr":args.lr_feat})
        else:
            print('No GNN update')
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr_pred})
        # optimizer = optim.SGD(model_param_group, weight_decay=args.decay)
        optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
        print(optimizer)
        
        #set up criterion
        if task_type == 'classification':
            criterion = nn.BCEWithLogitsLoss(reduction = "none")
        else:
            pass
        
        # training based on task type
        if task_type == 'classification':
            metrics_training = train_epoch_cls(args, model, device, train_loader, val_loader, test_loader, optimizer, criterion, task_type, training_round = i)

        elif task_type == 'regression':
            test_mae_list = train_epoch_reg(args, model, device, train_loader, val_loader, test_loader, optimizer, args.save_path) 

    #craw metrics
    average_test_metrics(args, task_type)
    df_train = average_train_metrics(args, task_type, remove = True)
    
    # plot training metrics
    plot_metrics(args, df_train, task_type)
    


if __name__ == "__main__":
    main()
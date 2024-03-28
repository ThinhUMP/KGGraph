from KGGraph.Dataset.molecule_dataset import MoleculeDataset
import pandas as pd
from KGGraph.Dataset.split import scaffold_split, random_split
from torch_geometric.data import DataLoader
from KGGraph.GnnModel.Architecture import GINNet, gin, GINGenerate
from KGGraph.GnnModel.Train.train_utils import train_epoch_cls, train_epoch_reg
from KGGraph.GnnModel.Train.visualize import plot_metrics
from KGGraph.GnnModel.Train.get_task_type_num_tasks import get_num_task, get_task_type
import torch
import argparse
from torch import optim
import warnings
warnings.filterwarnings('ignore')

def main():
    #set up parameters
    parser = argparse.ArgumentParser(description='PyTorch implementation of training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0.1,
                        help='weight decay (default: 0)')
    parser.add_argument('--hidden_channels', type=int, default=2048,
                        help='number of hidden nodes in the GNN network (default: 512).')
    # parser.add_argument('--num_layer', type=int, default=3, 
    #                     help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='embedding dimensions (default: 128)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--dataset', type=str, default = 'tox21',
                        help='[bbbp, bace, sider, clintox, sider,tox21, toxcast, esol,freesolv,lipophilicity]')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--num_workers', type=int, default = 20, help='number of workers for dataset loading')
    parser.add_argument('--save_path', type=str, default = 'dataset/', help='path for saving training images, test_metrics csv, model')
    args = parser.parse_args()

    #set up device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    #set up task type
    task_type = get_task_type(args)

    #set up number of tasks
    num_tasks = get_num_task(args)

    #set up dataset
    dataset = MoleculeDataset("dataset/" + task_type + "/" + args.dataset, dataset=args.dataset)
    print(dataset)
    
    #data split
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + task_type + '/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
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
    # model = GINNet(num_layer=args.num_layer, out_channels=num_tasks, dropout = args.dropout_ratio)
    # model = gin(in_channels=dataset[0].x.size(1), dim_h=args.hidden_channels, out_channels=num_tasks, dropout=args.dropout_ratio)
    model = GINGenerate(in_channels=dataset[0].x.size(1), emb_dim = args.emb_dim, dropout=args.dropout_ratio, out_channels=num_tasks)
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.decay)
    # optimizer = optim.SGD(model.parameters(), lr= args.lr, weight_decay=args.decay)
    print(optimizer)

    # training based on task type
    if task_type == 'classification':
        metrics_training = train_epoch_cls(args, model, device, train_loader, val_loader, test_loader, optimizer, task_type)

    elif task_type == 'regression':
        test_mae_list = train_epoch_reg(args, model, device, train_loader, val_loader, test_loader, optimizer, args.save_path) 

    # plot training metrics
    plot_metrics(args, metrics_training, task_type)
    


if __name__ == "__main__":
    main()
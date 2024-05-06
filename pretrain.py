import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import rdkit
import sys
from tqdm import tqdm
import numpy as np
from KGGraph.KGGModel.Architecture.GNN import GNN
from KGGraph.KGGDecode.decoder import Model_decoder
from KGGraph.KGGDecode.data_utils import MoleculeDataset, molgraph_to_graph_data
from KGGraph.KGGModel.Train.visualize import plot_pretrain_loss
import os
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def group_node_rep(node_rep, batch_size, num_part):
    """
Groups the node representations based on the batch size and number of partitions.

Args:
    node_rep (list): The list of node representations.
    batch_size (int): The size of the batch.
    num_part (list): The list of numbers of partitions for each batch.

Returns:
    tuple: A tuple containing two lists - group and super_group.
        - group (list): The grouped node representations.
        - super_group (list): The super group node representations.

Examples:
    >>> node_rep = [1, 2, 3, 4, 5, 6]
    >>> batch_size = 2
    >>> num_part = [[2, 1], [1, 1]]
    >>> group_node_rep(node_rep, batch_size, num_part)
    ([[1, 2], [3]], [4, 6])
"""

    group = []
    super_group = []
    # print('num_part', num_part)
    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        num_motif = num_part[i][1]
        num_all = num_atom + num_motif + 1
        group.append(node_rep[count:count + num_atom])
        super_group.append(node_rep[count + num_all -1])
        count += num_all
    return group, super_group


def train(args, model_list, loader, optimizer_list, device, pretrain_loss, epoch):
    model, model_decoder = model_list

    model.train()
    model_decoder.train()
    if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0, 0
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_size = len(batch)

        graph_batch = molgraph_to_graph_data(batch)
        graph_batch = graph_batch.to(device)
        node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
        num_part = graph_batch.num_part
        node_rep, super_node_rep = group_node_rep(node_rep, batch_size, num_part)

        loss, bond_if_auc, bond_if_ap, atom_type_acc, atom_num_rmse, bond_num_rmse = model_decoder(batch, node_rep, super_node_rep)

        optimizer_list.zero_grad()

        loss.backward()

        optimizer_list.step()

        if_auc += bond_if_auc
        if_ap += bond_if_ap
        a_type_acc += atom_type_acc
        a_num_rmse += atom_num_rmse
        b_num_rmse += bond_num_rmse

        if (step+1) % 20 == 0:
            if_auc = if_auc / 20 
            if_ap = if_ap / 20 
            type_acc = type_acc / 20 
            a_type_acc = a_type_acc / 20
            a_num_rmse = a_num_rmse / 20
            b_num_rmse = b_num_rmse / 20

            print('Batch:',step,'loss:',loss.item())
            if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0, 0
        pretrain_loss['loss'][epoch-1] = loss.item()
    pretrain_loss.to_csv('Data/pretrain_loss.csv')
    return pretrain_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='embedding dimensions (default: 512)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='./Data/zinc/all.txt',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--decompose_type', type=str, default="motif",
                        help='decompose_type (brics, jin, motif, smotif) (default: motif).')
    parser.add_argument('--output_model_file', type=str, default='./saved_model_kgg_nodecay/pretrain.pth',
                        help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument("--hidden_size", type=int, default=512, help='hidden size')
    parser.add_argument('--mask_node_edge', type=bool, default = True, help='Mask node and edge for pretrain and finetune')
    parser.add_argument('--fix_ratio', type=bool, default = True, help='Fixing ratio of removal nodes and edges or not')
    args = parser.parse_args()


    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print('device', device)

    dataset = MoleculeDataset(args.dataset, args.decompose_type, args.mask_node_edge, args.fix_ratio)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x:x, drop_last=True)

    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    
    if not os.path.isdir('./saved_model_kgg_nodecay'):
        os.mkdir('./saved_model_kgg_nodecay')
    if 'pretrain.pth' in os.listdir('saved_model_kgg_nodecay'):
        print('Continue pretraining')
        model.load_state_dict(torch.load(args.output_model_file))
    
    model_decoder = Model_decoder(args.hidden_size, device).to(device)

    model_list = [model, model_decoder]
    optimizer = optim.Adam([{"params":model.parameters()},{"params":model_decoder.parameters()}], lr=args.lr, weight_decay=args.decay)

    pretrain_loss = pd.DataFrame(columns = ['loss'], index = range(args.epochs))
    for epoch in range(1, args.epochs + 1):
        print('====epoch',epoch)
        pretrain_loss = train(args, model_list, loader, optimizer, device, pretrain_loss, epoch)

        if not args.output_model_file == "":
            torch.save(model.state_dict(), args.output_model_file)
    
    plot_pretrain_loss(pretrain_loss)


if __name__ == "__main__":
    main()

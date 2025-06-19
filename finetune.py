from KGGraph.KGGProcessor.finetune_dataset import MoleculeDataset
import pandas as pd
import os
from KGGraph.KGGProcessor.split import scaffold_split, random_split
from torch_geometric.data import DataLoader
from KGGraph.KGGModel.graph_model import GraphModel, GNN
from KGGraph.KGGModel.finetune_utils import (
    train_epoch_cls,
    train_epoch_reg,
    get_num_task,
    get_task_type,
    evaluate,
)
from KGGraph.KGGModel.visualize import plot_metrics, clean_state_dict
from KGGraph.KGGModel.crawl_metrics import average_test_metrics
import torch
import torch.nn as nn
import argparse
from torch import optim
from typing import List
import numpy as np
from pretrain import seed_everything
import warnings
import time

warnings.filterwarnings("ignore")


def main():
    # set up parameters
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of training of graph neural networks"
    )
    parser.add_argument(
        "--device", type=int, default=1, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--training_rounds",
        type=int,
        default=1,
        help="number of rounds to train to get the average test auc (default: 3)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr_feat", type=float, default=0.001, help="learning rate (default: 0.0005)"
    )
    parser.add_argument(
        "--lr_pred",
        type=float,
        default=0.001,
        help="learning rate for the prediction layer (default: 0.001)",
    )
    parser.add_argument(
        "--decay", type=float, default=0, help="weight decay (default: 0)"
        "--decay", type=float, default=0, help="weight decay (default: 0)"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5).",
    )
    parser.add_argument(
        "--emb_dim", type=int, default=512, help="embedding dimensions (default: 512)"
    )
    parser.add_argument(
        "--dropout_ratio", type=float, default=0.7, help="dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="how the node features across layers are combined. last, sum, max or concat",
    )
    parser.add_argument("--gnn_type", type=str, default="gin", help="gnn_type (gin, gat, gin_torch, transformer)")
    parser.add_argument(
        "--decompose_type",
        type=str,
        default="motif",
        help="decompose_type (brics, jin, motif, smotif, tmotif) (default: motif).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="esol",
        default="clintox",
        help="[bbbp, bace, sider, clintox, tox21, toxcast, hiv, muv, esol, freesolv, lipo, qm7, qm8, qm9]",
    )
    parser.add_argument(
        "--input_model_file",
        type=str,
        default="./pretrained_model_chembl29/gin/pretrain.pth",
        help="filename to read the model (if there is any)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for splitting the dataset, minibatch selection, random initialization.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="scaffold",
        help="random or scaffold or random_scaffold",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="number of workers for dataset loading",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="Data/",
        help="path for saving training images, test_metrics csv, model",
    )
    parser.add_argument(
        "--GNN_different_lr",
        type=bool,
        default=True,
        help="if the learning rate of GNN backbone is different from the learning rate of prediction layers",
    )
    parser.add_argument(
        "--mask_node",
        type=bool,
        default=False,
        default=False,
        help="Mask node for pretrain and finetune",
    )
    parser.add_argument(
        "--mask_edge",
        type=bool,
        default=False,
        default=False,
        help="Mask edge for pretrain and finetune",
    )
    parser.add_argument(
        "--mask_node_ratio",
        type=float,
        default=0.25,
        default=0.25,
        help="Ratio of removal nodes",
    )
    parser.add_argument(
        "--mask_edge_ratio",
        type=float,
        default=0.1,
        default=0.1,
        help="Ratio of removal edges",
    )
    parser.add_argument(
        "--fix_ratio",
        type=bool,
        default=False,
        default=False,
        help="Fixing ratio of removal nodes and edges or not at specified ratio",
    )
    args = parser.parse_args()

    # set up time
    # Start timing for finetuning
    round_start_finetune = time.time() 
    
    for i in range(1, args.training_rounds + 1):
        print("====Round ", i)

        # set up seeds
        seed_everything(args.seed)

        # set up device
        device = (
            torch.device("cuda:" + str(args.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print("device", device)

        # set up task type
        task_type = get_task_type(args)

        # set up number of tasks
        num_tasks = get_num_task(args)

        # set up dataset
        dataset = MoleculeDataset(
            "Data/" + task_type + "/" + args.dataset,
            dataset=args.dataset,
            decompose_type=args.decompose_type,
            mask_node=args.mask_node,
            mask_edge=args.mask_edge,
            mask_node_ratio=args.mask_node_ratio,
            mask_edge_ratio=args.mask_edge_ratio,
            fix_ratio=args.fix_ratio,
        )
        print(dataset)

        # data split
        if args.split == "scaffold":
            smiles_list = pd.read_csv(
                "Data/" + task_type + "/" + args.dataset + "/processed/smiles.csv",
                header=None,
            )[0].tolist()
            (
                train_dataset,
                valid_dataset,
                test_dataset,
                (_, _, test_smiles),
            ) = scaffold_split(
                dataset,
                smiles_list,
                null_value=0,
                frac_train=0.8,
                frac_valid=0.1,
                frac_test=0.1,
            )
            print("scaffold")
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(
                dataset,
                null_value=0,
                frac_train=0.8,
                frac_valid=0.1,
                frac_test=0.1,
                seed=args.seed,
            )
            print("random")
        else:
            raise ValueError("Invalid split option.")

        print(train_dataset[0])

        # with open(f"Data/contamination/test_{args.dataset}.txt", "a") as f:
        #         f.writelines("%s\n" % s for s in test_smiles)

        # data loader
        if args.dataset == "freesolv":
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # set up model
        model = GraphModel(
            args.num_layer,
            args.emb_dim,
            num_tasks,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type,
            x_features=dataset[0].x.size(1),
            edge_features=dataset[0].edge_attr.size(1),
        )

        # load pretrained model
        if not args.input_model_file == "":
            model.from_pretrained(args.input_model_file)
        model.to(device)

        # set up optimizer
        # different learning rate for different part of GNN
        model_param_group = []
        if args.GNN_different_lr:
            print("GNN update")
            model_param_group.append(
                {"params": model.gnn.parameters(), "lr": args.lr_feat}
            )
        else:
            print("No GNN update")
        model_param_group.append(
            {"params": model.graph_pred_linear.parameters(), "lr": args.lr_pred}
        )
        # optimizer = optim.SGD(model_param_group, weight_decay=args.decay)
        optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
        print(optimizer)

        # set up criterion
        if task_type == "classification":
            criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            pass

        # training based on task type
        if task_type == "classification":
            train_epoch_cls(
                args,
                model,
                device,
                train_loader,
                val_loader,
                test_loader,
                optimizer,
                criterion,
                task_type,
                training_round=i,
            )

        elif task_type == "regression":
            train_epoch_reg(
                args,
                model,
                device,
                train_loader,
                val_loader,
                test_loader,
                optimizer,
                task_type,
                training_round=i,
            )

    # End timing for finetuning
    round_end_finetune = time.time()
    print("========================")
    print(f"Time taken for finetuning 1 round: {((round_end_finetune - round_start_finetune)/args.training_rounds)/60:.2f} mins")
    print("========================")

    # craw metrics
    average_test_metrics(args, task_type)

    # plot training metrics
    df_train_path = os.path.join(
        args.save_path,
        task_type,
        args.dataset,
        f"train_metrics_round_1.csv",
    )
    df_train = pd.read_csv(df_train_path)
    plot_metrics(args, df_train, task_type)


if __name__ == "__main__":
    main()

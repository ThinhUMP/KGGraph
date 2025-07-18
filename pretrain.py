import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import rdkit
import numpy as np
from KGGraph.KGGModel.graph_model import GNN
from KGGraph.KGGDecode.decoder import Model_decoder
from KGGraph.KGGProcessor.pretrain_dataset import MoleculeDataset
from KGGraph.KGGModel.visualize import plot_pretrain_loss
from KGGraph.KGGModel.pretrain_utils import train
import os
import pandas as pd
from typing import List
import random
import warnings
import time

warnings.filterwarnings("ignore")

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of pre-training of graph neural networks"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 60)",
    )

    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--decay", type=float, default=0, help="weight decay (default: 0)"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5)",
    )
    parser.add_argument(
        "--emb_dim", type=int, default=512, help="embedding dimensions (default: 512)"
    )
    parser.add_argument(
        "--dropout_ratio", type=float, default=0.5, help="dropout ratio (default: 0.2)"
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="how the node features across layers are combined. last, sum, max or concat",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./Data/pretrain_datasets/250kzinc15.txt",
        help="root directory of dataset. For now, only classification.",
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="gin",
        help="gnn_type (gat, gin, gcn, graphsage)",
    )
    parser.add_argument(
        "--decompose_type",
        type=str,
        default="motif",
        help="decompose_type (brics, jin, motif, smotif, tmotif) (default: motif).",
    )
    parser.add_argument(
        "--output_model_directory",
        type=str,
        default="./pretrained_model_zinc15/",
        help="directory contains pre-trained models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for splitting the dataset over 3 rounds.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="number of workers for dataset loading",
    )
    parser.add_argument("--hidden_size", type=int, default=512, help="hidden size")
    parser.add_argument(
        "--mask_node",
        type=bool,
        default=False,
        help="Mask node for pretrain and finetune",
    )
    parser.add_argument(
        "--mask_edge",
        type=bool,
        default=False,
        help="Mask edge for pretrain and finetune",
    )
    parser.add_argument(
        "--mask_node_ratio",
        type=float,
        default=0.15,
        help="Ratio of removal nodes",
    )
    parser.add_argument(
        "--mask_edge_ratio",
        type=float,
        default=0.15,
        help="Ratio of removal edges",
    )
    parser.add_argument(
        "--fix_ratio",
        type=bool,
        default=True,
        help="Fixing ratio of removal nodes and edges at specified ratio or not",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("device", device)
    print("GNNs backbone: ", args.gnn_type)

    torch.multiprocessing.set_sharing_strategy("file_system")

    dataset = MoleculeDataset(
        args.dataset,
        args.decompose_type,
        args.mask_node,
        args.mask_edge,
        args.mask_node_ratio,
        args.mask_edge_ratio,
        args.fix_ratio,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
        drop_last=True,
    )

    model = GNN(
        args.num_layer,
        args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
    ).to(device)

    # if "pretrain.pth" in os.listdir("pretrained_model_chembl29/gin/"):
    #     print("Continue pretraining")
    #     model.load_state_dict(torch.load("pretrained_model_chembl29/gin/pretrain.pth"))

    model_decoder = Model_decoder(args.hidden_size, device).to(device)

    model_list = [model, model_decoder]
    optimizer = optim.Adam(
        [{"params": model.parameters()}, {"params": model_decoder.parameters()}],
        lr=args.lr,
        weight_decay=args.decay,
    )

    if args.output_model_directory == "":
        raise ValueError(
            "You must indicate the directory where pretrained model is saved!"
        )

    pretrain_loss = pd.DataFrame(columns=["loss"], index=range(args.epochs))

    # Start timing for finetuning
    start_pretrain = time.time()

    for epoch in range(1, args.epochs + 1):
        print("====epoch", epoch)
        pretrain_loss = train(
            args, model_list, loader, optimizer, device, pretrain_loss, epoch
        )

        # Save model for every epoch
        base_path = os.path.join(args.output_model_directory, args.gnn_type)
        print(base_path)
        os.makedirs(base_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(base_path, "pretrain.pth"))

        # Save model at specific checkpoint epochs
        checkpoint_epochs = [40, 60, 80, 100]
        if epoch in checkpoint_epochs:
            checkpoint_path = os.path.join(
                args.output_model_directory, f"{args.gnn_type}_e{epoch}"
            )
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(
                model.state_dict(), os.path.join(checkpoint_path, "pretrain.pth")
            )

    # End timing for finetuning
    end_pretrain = time.time()
    print("========================")
    print(
        f"Time taken for pretraining: {((end_pretrain - start_pretrain))/3600:.2f} hours"
    )
    print("========================")

    plot_pretrain_loss(args, pretrain_loss)


if __name__ == "__main__":
    main()

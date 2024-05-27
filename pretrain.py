import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import rdkit
import numpy as np
from KGGraph.KGGModel.Architecture.GNN import GNN
from KGGraph.KGGDecode.decoder import Model_decoder
from KGGraph.KGGProcessor.pretrain_dataset import MoleculeDataset
from KGGraph.KGGModel.TrainUtils.visualize import plot_pretrain_loss
from KGGraph.KGGModel.TrainUtils.pretrain_utils import train
import os
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


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
        default=60,
        help="number of epochs to train (default: 100)",
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
        default="./Data/zinc/all.txt",
        help="root directory of dataset. For now, only classification.",
    )
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument(
        "--decompose_type",
        type=str,
        default="motif",
        help="decompose_type (brics, jin, motif, smotif) (default: motif).",
    )
    parser.add_argument(
        "--output_model_file",
        type=str,
        default="./saved_model_mlp_ce60_1layer/pretrain.pth",
        help="filename to output the pre-trained model",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
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
        default=0.05,
        help="Ratio of removal nodes",
    )
    parser.add_argument(
        "--mask_edge_ratio",
        type=float,
        default=0.05,
        help="Ratio of removal edges",
    )
    parser.add_argument(
        "--fix_ratio",
        type=bool,
        default=True,
        help="Fixing ratio of removal nodes and edges at specified ratio or not",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print("device", device)

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

    if not os.path.isdir("./saved_model_mlp_ce60_1layer"):
        os.mkdir("./saved_model_mlp_ce60_1layer")
    if "pretrain.pth" in os.listdir("saved_model_mlp_ce60_1layer"):
        print("Continue pretraining")
        model.load_state_dict(torch.load(args.output_model_file))

    model_decoder = Model_decoder(args.hidden_size, device).to(device)

    model_list = [model, model_decoder]
    optimizer = optim.Adam(
        [{"params": model.parameters()}, {"params": model_decoder.parameters()}],
        lr=args.lr,
        weight_decay=args.decay,
    )

    pretrain_loss = pd.DataFrame(columns=["loss"], index=range(args.epochs))
    for epoch in range(1, args.epochs + 1):
        print("====epoch", epoch)
        pretrain_loss = train(
            args, model_list, loader, optimizer, device, pretrain_loss, epoch
        )

        if not args.output_model_file == "":
            torch.save(model.state_dict(), args.output_model_file)

    plot_pretrain_loss(pretrain_loss)


if __name__ == "__main__":
    main()

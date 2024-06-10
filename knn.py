from KGGraph.KGGProcessor.finetune_dataset import MoleculeDataset
import pandas as pd
from torch_geometric.data import DataLoader
from KGGraph.KGGModel.visualize import extract_embeddings
from KGGraph.KGGModel.graph_model import GNN
from KGGraph.KGGModel.finetune_utils import get_task_type
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import torch
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def main():
    # set up parameters
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of training of graph neural networks"
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
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5).",
    )
    parser.add_argument(
        "--emb_dim", type=int, default=512, help="embedding dimensions (default: 512)"
    )
    parser.add_argument(
        "--dropout_ratio", type=float, default=0.4, help="dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="how the node features across layers are combined. last, sum, max or concat",
    )
    parser.add_argument("--gnn_type", type=str, default="gin", help="gnn_type (gin)")
    parser.add_argument(
        "--decompose_type",
        type=str,
        default="motif",
        help="decompose_type (brics, jin, motif, smotif) (default: motif).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bbbp",
        help="[bbbp, bace, sider, clintox, tox21, toxcast, hiv, muv, esol, freesolv, lipo, qm7, qm8, qm9]",
    )
    parser.add_argument(
        "--input_model_file",
        type=str,
        default="saved_model_mlp_ce60_1layer_x/pretrain.pth",
        help="filename to read the model (if there is any)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for splitting the dataset."
    )
    parser.add_argument(
        "--runseed",
        type=int,
        default=42,
        help="Seed for minibatch selection, random initialization.",
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
        default=16,
        help="number of workers for dataset loading",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="Data/",
        help="path for saving training images, test_metrics csv, model",
    )
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
        default=0.25,
        help="Ratio of removal nodes",
    )
    parser.add_argument(
        "--mask_edge_ratio",
        type=float,
        default=0.1,
        help="Ratio of removal edges",
    )
    parser.add_argument(
        "--fix_ratio",
        type=bool,
        default=False,
        help="Fixing ratio of removal nodes and edges or not at specified ratio",
    )
    args = parser.parse_args()

    # set up seeds
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # set up device
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("device", device)

    # set up task type
    task_type = get_task_type(args)

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

    # data loader
    if args.dataset == "freesolv":
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    state_dict = torch.load(args.input_model_file)
        
    model = GNN(
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
        x_features=dataset[0].x.size(1),
        edge_features=dataset[0].edge_attr.size(1),
        )
    model.load_state_dict(state_dict)
    print("Load model done")
    X, y = extract_embeddings(args, model, device, loader)
    print("Embeddings", X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)
    rocauc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)
    print("rocauc", rocauc)
    print("f1", f1)
    print("ap", ap)


if __name__ == "__main__":
    main()
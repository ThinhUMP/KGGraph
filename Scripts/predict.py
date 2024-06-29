import sys
import pathlib

root_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(root_dir)

from KGGraph.KGGProcessor.finetune_dataset import MoleculeDataset
import pandas as pd
from torch_geometric.data import DataLoader
from KGGraph.KGGModel.graph_model import GraphModel
from KGGraph.KGGModel.finetune_utils import get_task_type, predict_reg
from KGGraph.KGGModel.visualize import draw_pred_reg
from KGGraph.KGGProcessor.split import scaffold_split, random_split
import torch
import argparse
from pretrain import seed_everything
import warnings

warnings.filterwarnings("ignore")


def main():
    # set up parameters
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of t-SNE visualization of embeddings"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
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
    parser.add_argument(
        "--decompose_type",
        type=str,
        default="motif",
        help="decompose_type (brics, jin, motif, smotif) (default: motif).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ecoli",
        help="[bbbp, bace, sider, clintox, tox21, toxcast, hiv, muv, esol, freesolv, lipo, qm7, qm8, qm9]",
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

    # set up dataset
    df_pred = pd.read_csv(f"Data/{task_type}/{args.dataset}/raw/{args.dataset}.csv")
    
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
    # print(dataset[0])
    # # data loader
    # predict_loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     drop_last=True,
    # )
    
    # data split
    smiles_list = pd.read_csv(
        "Data/" + task_type + "/" + args.dataset + "/processed/smiles.csv",
        header=None,
    )[0].tolist()
    if args.split == "scaffold":
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
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.seed[i - 1],
        )
        print("random")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

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


    # Load model
    state_dict = torch.load(f"Data/{task_type}/{args.dataset}/{args.dataset}_1.pth")

    model = GraphModel(
            args.num_layer,
            args.emb_dim,
            num_tasks=1,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            x_features=dataset[0].x.size(1),
            edge_features=dataset[0].edge_attr.size(1),
        )
    
    model.load_state_dict(state_dict)
    model.to(device)

    # Predict
    if task_type == "classification":
        pass
    else:
        # df_pred = predict_reg(model, predict_loader, device, df_pred)
        draw_pred_reg(model, test_loader, device)
        # print(df_pred)

if __name__ == "__main__":
    main()
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys
import pathlib
import numpy as np
from matplotlib.colors import ListedColormap

root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.KGGModel.graph_model import GraphModel
from KGGraph.KGGModel.finetune_utils import get_num_task


def plot_metrics(args, df, task_type):
    """
    Plot the training, validation, and test loss, AUC, F1, and AP for each epoch.

    Args:
    args: Argument parser or a similar object with attributes like save_path and dataset.
    df (dict): Dataframe containing lists of training, validation, and test metrics.
    task_type (str): The type of task (e.g., 'classification', 'regression').

    The function saves the plot to a file and displays it.
    """
    if task_type == "classification":
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Plot loss
        axs[0, 0].plot(df["train_loss"], label="Train loss")
        axs[0, 0].plot(df["val_loss"], label="Val loss")
        axs[0, 0].plot(df["test_loss"], label="Test loss")
        axs[0, 0].set_title("Loss")
        axs[0, 0].legend()

        # Plot AUC
        axs[0, 1].plot(df["train_auc"], label="Train AUC")
        axs[0, 1].plot(df["val_auc"], label="Val AUC")
        axs[0, 1].plot(df["test_auc"], label="Test AUC")
        axs[0, 1].set_title("AUC")
        axs[0, 1].legend()

        # Plot F1
        axs[1, 0].plot(df["train_f1"], label="Train F1")
        axs[1, 0].plot(df["val_f1"], label="Val F1")
        axs[1, 0].plot(df["test_f1"], label="Test F1")
        axs[1, 0].set_title("F1 Score")
        axs[1, 0].legend()

        # Plot AP
        axs[1, 1].plot(df["train_ap"], label="Train AP")
        axs[1, 1].plot(df["val_ap"], label="Val AP")
        axs[1, 1].plot(df["test_ap"], label="Test AP")
        axs[1, 1].set_title("Average Precision")
        axs[1, 1].legend()

        # Setting labels for all subplots
        for ax in axs.flat:
            ax.set(xlabel="Epoch", ylabel="Value")

        # Adjust layout and save the plot
        plt.tight_layout()
        if not os.path.isdir(f"{args.save_path+task_type}/{args.dataset}/figures"):
            os.mkdir(f"{args.save_path+task_type}/{args.dataset}/figures")
        plt.savefig(
            f"{args.save_path+task_type}/{args.dataset+'/figures'}/training.png",
            dpi=600,
        )
        plt.show()
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Plot loss
        axs[0, 0].plot(df["train_mae"], label="Train MAE")
        axs[0, 0].plot(df["val_mae"], label="Val MAE")
        axs[0, 0].plot(df["test_mae"], label="Test MAE")
        axs[0, 0].set_title("MAE Loss")
        axs[0, 0].legend()

        # Plot AUC
        axs[0, 1].plot(df["train_mse"], label="Train MSE")
        axs[0, 1].plot(df["val_mse"], label="Val MSE")
        axs[0, 1].plot(df["test_mse"], label="Test MSE")
        axs[0, 1].set_title("MSE Loss")
        axs[0, 1].legend()

        # Plot F1
        axs[1, 0].plot(df["train_rmse"], label="Train RMSE")
        axs[1, 0].plot(df["val_rmse"], label="Val RMSE")
        axs[1, 0].plot(df["test_rmse"], label="Test RMSE")
        axs[1, 0].set_title("RMSE Loss")
        axs[1, 0].legend()

        # Plot AP
        axs[1, 1].plot(df["train_r2"], label="Train R2")
        axs[1, 1].plot(df["val_r2"], label="Val R2")
        axs[1, 1].plot(df["test_r2"], label="Test R2")
        axs[1, 1].set_title("R2")
        axs[1, 1].legend()

        # Setting labels for all subplots
        for ax in axs.flat:
            ax.set(xlabel="Epoch", ylabel="Value")

        # Adjust layout and save the plot
        plt.tight_layout()

        if not os.path.isdir(f"{args.save_path+task_type}/{args.dataset}/figures"):
            os.mkdir(f"{args.save_path+task_type}/{args.dataset}/figures")
        plt.savefig(
            f"{args.save_path+task_type}/{args.dataset+'/figures'}/training.png",
            dpi=600,
        )
        plt.show()


def plot_pretrain_loss(args, pretrain_loss):
    os.makedirs("Data/pretrain_datasets/log", exist_ok=True)
    plt.plot(pretrain_loss["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(
        f"Data/pretrain_datasets/log/pretraining_{args.gnn_type}.png",
        dpi=600,
        bbox_inches="tight",
        transparent=False,
    )
    plt.savefig(
        f"Data/pretrain_datasets/log/pretraining_{args.gnn_type}.pdf",
        dpi=600,
        bbox_inches="tight",
        transparent=False,
        format="pdf",
    )
    plt.show()


def clean_state_dict(state_dict):
    """
    This function processes the given state dictionary by:
    1. Removing the 'gnn.' prefix from keys that start with 'gnn.'.
    2. Removing keys that contain 'graph_pred_linear'.

    Args:
    state_dict (dict): The state dictionary to be processed.

    Returns:
    dict: The processed state dictionary with the specified changes.
    """
    # Create a new dictionary
    new_state_dict = {}

    # Iterate over the state_dict
    for name, param in state_dict.items():
        # If the key starts with 'gnn.', remove the 'gnn.' prefix
        if name.startswith("gnn."):
            new_name = name.replace("gnn.", "")
            new_state_dict[new_name] = param
        else:
            new_state_dict[name] = param

    # Remove keys that contain 'graph_pred_linear'
    keys_to_remove = [key for key in new_state_dict if "graph_pred_linear" in key]
    for key in keys_to_remove:
        del new_state_dict[key]

    return new_state_dict


def extract_embeddings(args, model, device, loader):
    model.to(device)
    model.eval()
    embeddings_list = []
    y_true = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Generating embeddings")):
            batch = batch.to(device)
            node_representation = model(batch.x, batch.edge_index, batch.edge_attr)

            if args.motif_embeddings == True:
                num_node = batch.num_node.detach().cpu().numpy().flatten()
                num_motif = batch.num_motif.detach().cpu().numpy().flatten()
                super_rep = GraphModel.motif_rep(
                    node_representation, num_node, num_motif
                )
            else:
                super_rep = GraphModel.super_node_rep(node_representation, batch.batch)

            embeddings_list.append(super_rep.detach().cpu().numpy())

            y = batch.y.view(-1, 1).to(torch.float64)
            y_true.append(batch.y.view(-1, 1).detach().cpu().numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    y = np.concatenate(y_true, axis=0)
    return embeddings, y


def visualize_embeddings(args, model, device, loader, task_type):
    embeddings, _ = extract_embeddings(args, model, device, loader)

    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_pca)

    custom_cmap = ListedColormap(["#EBBC4E", "#7DB0A8"])

    plt.figure(figsize=(7, 7))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=loader.dataset.y,
        cmap=custom_cmap,
        s=100,  # Increase the size of the points
        edgecolor="w",  # Add white edge color for better visibility
        linewidth=0.5,  # Set the linewidth for the edges
    )

    # Create a custom legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#EBBC4E",
            markersize=10,
            label="0",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#7DB0A8",
            markersize=10,
            label="1",
        ),
    ]

    plt.legend(
        handles=handles,
        title=f"{args.target_column}",
        title_fontsize="13",
        loc="upper left",
        # bbox_to_anchor=(0.5, -0.05),
        prop={"size": 12},
        ncol=2,
    )
    if args.motif_embeddings:
        plt.title(f"{args.dataset}-Motif embeddings".upper(), fontsize=16)
    else:
        plt.title(f"{args.dataset}".upper(), fontsize=16)
    plt.xlabel("t-SNE-0", fontsize=16)
    plt.ylabel("t-SNE-1", fontsize=16)

    # Hide x and y axis values
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if args.motif_embeddings:
        plt.savefig(
            f"{args.save_path+task_type}/{args.dataset+'/figures'}/{args.dataset}_motif_tsne_{args.target_column}.pdf",
            dpi=600,
            bbox_inches="tight",
            transparent=False,
            format="pdf",
        )
    else:
        plt.savefig(
            f"{args.save_path+task_type}/{args.dataset+'/figures'}/{args.dataset}_tsne_{args.target_column}.pdf",
            dpi=600,
            bbox_inches="tight",
            transparent=False,
            format="pdf",
        )

    plt.show()


def visualize_embeddings_reg(args, model, device, loader, task_type):
    embeddings, _ = extract_embeddings(args, model, device, loader)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Use a colormap for continuous values
    plt.figure(figsize=(7, 7))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=loader.dataset.y,  # Continuous values
        cmap="plasma",  # Colormap for continuous values
        s=100,  # Increase the size of the points
        edgecolor="w",  # Add white edge color for better visibility
        linewidth=0.5,  # Set the linewidth for the edges
    )

    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"{args.target_column}", fontsize=16)

    if args.motif_embeddings:
        plt.title(f"{args.dataset}-Motif embeddings".upper(), fontsize=16)
    else:
        plt.title(f"{args.dataset}".upper(), fontsize=16)
    plt.xlabel("t-SNE-0", fontsize=16)
    plt.ylabel("t-SNE-1", fontsize=16)

    # Hide x and y axis values
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if args.motif_embeddings:
        plt.savefig(
            f"{args.save_path+task_type}/{args.dataset+'/figures'}/{args.dataset}_motif_tsne_{args.target_column}.pdf",
            dpi=600,
            bbox_inches="tight",
            transparent=False,
            format="pdf",
        )
    else:
        plt.savefig(
            f"{args.save_path+task_type}/{args.dataset+'/figures'}/{args.dataset}_tsne_{args.target_column}.pdf",
            dpi=600,
            bbox_inches="tight",
            transparent=False,
            format="pdf",
        )

    plt.show()

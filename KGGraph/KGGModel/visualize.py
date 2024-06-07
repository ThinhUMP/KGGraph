import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import sys
import pathlib
import numpy as np
from matplotlib.colors import ListedColormap

root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.KGGModel.graph_model import GraphModel


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
        if args.dataset in ["qm7", "qm8", "qm9"]:
            # Plot loss
            plt.plot(df["train_loss"], label="Train mae loss")
            plt.plot(df["val_loss"], label="Val mae loss")
            plt.plot(df["test_loss"], label="Test mae loss")
            plt.title("MAE Loss")
        else:
            # Plot loss
            plt.plot(df["train_loss"], label="Train rmse loss")
            plt.plot(df["val_loss"], label="Val rmse loss")
            plt.plot(df["test_loss"], label="Test rmse loss")
            plt.title("RMSE Loss")
        plt.savefig(
            f"{args.save_path+task_type}/{args.dataset+'/figures'}/training.png",
            dpi=600,
        )
        plt.show()


def plot_pretrain_loss(pretrain_loss):
    plt.plot(pretrain_loss["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pretraining Loss")
    plt.savefig(f"Data/pretraining.png", dpi=600)
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


def visualize_embeddings(args, model, device, loader, task_type):
    def extract_embeddings(model, device, loader):
        model.to(device)
        model.eval()
        embeddings_list = []

        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader, desc="Iteration")):
                batch = batch.to(device)
                node_representation = model(batch.x, batch.edge_index, batch.edge_attr)
                super_rep = GraphModel.super_node_rep(node_representation, batch.batch)
                embeddings_list.append(super_rep.detach().cpu().numpy())

        embeddings = np.concatenate(embeddings_list, axis=0)
        return embeddings

    embeddings = extract_embeddings(model, device, loader)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Define custom colormap for inactive (purple) and active (yellow)
    custom_cmap = ListedColormap(["yellow", "green"])
    plt.figure(figsize=(15, 15))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=loader.dataset.y,
        cmap=custom_cmap,
        s=50,
    )
    plt.colorbar()
    # Create a custom legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="yellow",
            markersize=10,
            label="Inactive",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="Active",
        ),
    ]

    plt.legend(handles=handles)
    plt.title(f"t-SNE Visualization of {args.dataset} on the test set")
    plt.xlabel("tsne-1")
    plt.ylabel("tsne-2")
    plt.savefig(
        f"{args.save_path+task_type}/{args.dataset+'/figures'}/training.png",
        dpi=600,
    )
    plt.show()

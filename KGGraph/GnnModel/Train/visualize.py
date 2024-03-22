from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import os
def plot_metrics(args,
    metrics_training, task_type
):
    """Plot the metrics for each epoch

    Args:
        train_loss_list (list): training losses for each epoch
        val_loss_list (list): validation losses for each epoch
        train_auc_list (list): training AUC for each epoch
        val_auc_list (list): validation AUC for each epoch
        train_f1_list (list): training F1 for each epoch
        val_f1_list (list): validation F1 for each epoch
        train_ap_list (list): training AP for each epoch
        val_ap_list (list): validation AP for each epoch
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot loss
    axs[0, 0].plot(metrics_training['train loss'], label="Train loss")
    axs[0, 0].plot(metrics_training['val loss'], label="Val loss")
    axs[0, 0].plot(metrics_training['test loss'], label="Test loss")
    axs[0, 0].set_title("Loss")
    axs[0, 0].legend()

    # Plot AUC
    axs[0, 1].plot(metrics_training['train auc'], label="Train AUC")
    axs[0, 1].plot(metrics_training['val auc'], label="Val AUC")
    axs[0, 1].plot(metrics_training['test auc'], label="Test AUC")
    axs[0, 1].set_title("AUC")
    axs[0, 1].legend()

    # Plot F1
    axs[1, 0].plot(metrics_training['train f1'], label="Train F1")
    axs[1, 0].plot(metrics_training['val f1'], label="Val F1")
    axs[1, 0].plot(metrics_training['test f1'], label="Test F1")
    axs[1, 0].set_title("F1 Score")
    axs[1, 0].legend()

    # Plot AP
    axs[1, 1].plot(metrics_training['train ap'], label="Train AP")
    axs[1, 1].plot(metrics_training['val ap'], label="Val AP")
    axs[1, 1].plot(metrics_training['test ap'], label="Test AP")
    axs[1, 1].set_title("Average Precision")
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='Value')

    plt.tight_layout()
    if not os.path.isdir(f"{args.save_path+task_type}/{args.dataset}/figures"):
        os.mkdir(f"{args.save_path+task_type}/{args.dataset}/figures")
    plt.savefig(f"{args.save_path+task_type}/{args.dataset+'/figures'}/training.png", dpi=600)
    plt.show()
    
def plot_targets(pred, ground_truth):
    """Plot true vs predicted value in a scatter plot

    Args:
        pred (array): predicted values
        ground_truth (array): ground truth values
    """
    f, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred, ground_truth, s=0.5)
    plt.xlim(-2, 7)
    plt.ylim(-2, 7)
    ax.axline((1, 1), slope=1)
    plt.xlabel("Predicted Value")
    plt.ylabel("Ground truth")
    plt.title("Ground truth vs prediction")
    plt.show()
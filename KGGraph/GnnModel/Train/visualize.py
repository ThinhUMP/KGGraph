from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

def plot_metrics(args,
    train_loss_list, val_loss_list, test_loss_list,
    train_auc_list, val_auc_list, test_auc_list,
    train_ap_list, val_ap_list, test_ap_list,
    train_f1_list, val_f1_list, test_f1_list,
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
    axs[0, 0].plot(train_loss_list, label="Train loss")
    axs[0, 0].plot(val_loss_list, label="Val loss")
    axs[0, 0].plot(test_loss_list, label="Test loss")
    axs[0, 0].set_title("Loss")
    axs[0, 0].legend()

    # Plot AUC
    axs[0, 1].plot(train_auc_list, label="Train AUC")
    axs[0, 1].plot(val_auc_list, label="Val AUC")
    axs[0, 1].plot(test_auc_list, label="Test AUC")
    axs[0, 1].set_title("AUC")
    axs[0, 1].legend()

    # Plot F1
    axs[1, 0].plot(train_f1_list, label="Train F1")
    axs[1, 0].plot(val_f1_list, label="Val F1")
    axs[1, 0].plot(test_f1_list, label="Test F1")
    axs[1, 0].set_title("F1 Score")
    axs[1, 0].legend()

    # Plot AP
    axs[1, 1].plot(train_ap_list, label="Train AP")
    axs[1, 1].plot(val_ap_list, label="Val AP")
    axs[1, 1].plot(test_ap_list, label="Test AP")
    axs[1, 1].set_title("Average Precision")
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='Value')

    plt.tight_layout()
    plt.savefig(f"{args.save_fig_path+args.dataset+'/figures'}/training.png", dpi=600)
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
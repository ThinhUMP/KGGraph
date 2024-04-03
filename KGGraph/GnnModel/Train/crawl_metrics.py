import pandas as pd
import os
def create_test_round_df(args, roc_list, ap_list, f1_list, task_type, training_round):
    """
    Creates and saves a test metrics DataFrame for various datasets.

    Parameters:
    args: Argument parser or a similar object with attributes dataset, save_path, and task_type.
    roc_list (list): List of ROC AUC values for individual task.
    ap_list (list): List of average precision (AP) values for individual task.
    f1_list (list): List of F1 scores for individual task.
    task_type (str): The type of task for which metrics are being recorded.
    """
    if args.dataset == "tox21":
        tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        test_metrics_tox21 = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_tox21['AUC'] = roc_list
        test_metrics_tox21['AP'] = ap_list
        test_metrics_tox21['F1'] = f1_list
        test_metrics_tox21.to_csv(f"{args.save_path+task_type}/{args.dataset}/test_metrics_round_{training_round}.csv", index=False)
        
    elif args.dataset == "bace":
        tasks = ['Class']
        test_metrics_bace = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_bace['AUC'] = roc_list
        test_metrics_bace['AP'] = ap_list
        test_metrics_bace['F1'] = f1_list
        test_metrics_bace.to_csv(f"{args.save_path+task_type}/{args.dataset}/test_metrics_round_{training_round}.csv", index=False)
        
    elif args.dataset == "bbbp":
        tasks = ['p_np']
        test_metrics_bbbp = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_bbbp['AUC'] = roc_list
        test_metrics_bbbp['AP'] = ap_list
        test_metrics_bbbp['F1'] = f1_list
        test_metrics_bbbp.to_csv(f"{args.save_path+task_type}/{args.dataset}/test_metrics_round_{training_round}.csv", index=False)
        
    elif args.dataset == "clintox":
        tasks = ['FDA_APPROVED', 'CT_TOX']
        test_metrics_clintox = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_clintox['AUC'] = roc_list
        test_metrics_clintox['AP'] = ap_list
        test_metrics_clintox['F1'] = f1_list
        test_metrics_clintox.to_csv(f"{args.save_path+task_type}/{args.dataset}/test_metrics_round_{training_round}.csv", index=False)
    
    elif args.dataset == "sider":
        tasks = ['Hepatobiliary disorders',
       'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
       'Investigations', 'Musculoskeletal and connective tissue disorders',
       'Gastrointestinal disorders', 'Social circumstances',
       'Immune system disorders', 'Reproductive system and breast disorders',
       'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
       'General disorders and administration site conditions',
       'Endocrine disorders', 'Surgical and medical procedures',
       'Vascular disorders', 'Blood and lymphatic system disorders',
       'Skin and subcutaneous tissue disorders',
       'Congenital, familial and genetic disorders',
       'Infections and infestations',
       'Respiratory, thoracic and mediastinal disorders',
       'Psychiatric disorders', 'Renal and urinary disorders',
       'Pregnancy, puerperium and perinatal conditions',
       'Ear and labyrinth disorders', 'Cardiac disorders',
       'Nervous system disorders',
       'Injury, poisoning and procedural complications']
        test_metrics_sider = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_sider['AUC'] = roc_list
        test_metrics_sider['AP'] = ap_list
        test_metrics_sider['F1'] = f1_list
        test_metrics_sider.to_csv(f"{args.save_path+task_type}/{args.dataset}/test_metrics_round_{training_round}.csv", index=False)
        
    elif args.dataset == "toxcast":
        toxcast = pd.read_csv("dataset/classification/toxcast/raw/toxcast.csv")
        tasks = list(toxcast.columns)[1:] 
        test_metrics_toxcast = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_toxcast['AUC'] = roc_list
        test_metrics_toxcast['AP'] = ap_list
        test_metrics_toxcast['F1'] = f1_list
        test_metrics_toxcast.to_csv(f"{args.save_path+task_type}/{args.dataset}/test_metrics_round_{training_round}.csv, index=False")
        
    else:
        raise ValueError("Invalid dataset name.")
    
def create_train_round_df(
    args, train_df, train_loss, train_auc, train_ap, train_f1, 
    val_loss, val_auc, val_ap, val_f1, test_loss, test_auc, test_ap, test_f1,
    task_type, epoch, training_round
    ):
    """
    Updates and saves the training metrics DataFrame with new epoch data.

    Parameters:
    args: Argument parser or a similar object with attributes save_path, task_type, and dataset.
    train_df (DataFrame): The DataFrame containing the training metrics.
    train_loss (float): Training loss for the current epoch.
    train_auc (float): Training AUC for the current epoch.
    train_ap (float): Training AP for the current epoch.
    train_f1 (float): Training F1 score for the current epoch.
    val_loss (float): Validation loss for the current epoch.
    val_auc (float): Validation AUC for the current epoch.
    val_ap (float): Validation AP for the current epoch.
    val_f1 (float): Validation F1 score for the current epoch.
    task_type (str): The type of task for which metrics are being recorded.
    epoch (int): The current epoch number.
    """
    train_df.at[epoch-1, "train_loss"] = train_loss
    train_df.at[epoch-1,"train_auc"] = train_auc
    train_df.at[epoch-1,"train_ap"] = train_ap
    train_df.at[epoch-1,"train_f1"] = train_f1
    train_df.at[epoch-1,"val_loss"] = val_loss
    train_df.at[epoch-1,"val_auc"] = val_auc
    train_df.at[epoch-1,"val_ap"] = val_ap
    train_df.at[epoch-1,"val_f1"] = val_f1
    train_df.at[epoch-1, "test_loss"] = test_loss
    train_df.at[epoch-1, "test_auc"] = test_auc
    train_df.at[epoch-1, "test_ap"] = test_ap
    train_df.at[epoch-1, "test_f1"] = test_f1
    train_df.to_csv(f"{args.save_path+task_type}/{args.dataset}/train_metrics_round_{training_round}.csv", index=False)
    
def average_train_metrics(args, task_type, remove = False):
    dfs = pd.DataFrame()
    for i in range(1, args.training_rounds+1):
        file_path = f"dataset/{task_type}/{args.dataset}/train_metrics_round_{i}.csv"
        round_metrics = pd.read_csv(file_path)
        dfs = pd.concat([dfs, round_metrics], axis=0)
        if remove:
            os.remove(file_path)  # Delete the file
    df_avg = dfs.groupby(dfs.index).mean()
    df_avg.to_csv(f"dataset/{task_type}/{args.dataset}/train_metrics.csv")
    return df_avg
    
def average_test_metrics(args, task_type, remove = True):
    dfs = pd.DataFrame()
    for i in range(1, args.training_rounds+1):
        file_path = f"dataset/{task_type}/{args.dataset}/test_metrics_round_{i}.csv"
        round_metrics = pd.read_csv(file_path)
        dfs = pd.concat([dfs, round_metrics], axis=0)
        if remove:
            os.remove(file_path)  # Delete the file
    df_avg = dfs.groupby(dfs.index).mean()
    df_std = dfs.groupby(dfs.index).std()
    df_avg.to_csv(f"dataset/{task_type}/{args.dataset}/test_metrics_avg.csv")
    df_std.to_csv(f"dataset/{task_type}/{args.dataset}/test_metrics_std.csv")
    print(f"AUC test for {args.dataset} dataset over {args.training_rounds} training rounds: {df_avg.AUC.mean():.2f}±{df_std.AUC.mean():.2f}")
import pandas as pd
def create_test_df(args, roc_list, ap_list, f1_list, task_type):
    
    if args.dataset == "tox21":
        tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        test_metrics_tox21 = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_tox21['AUC'] = roc_list
        test_metrics_tox21['AP'] = ap_list
        test_metrics_tox21['F1'] = f1_list
        test_metrics_tox21.to_csv(f"{args.save_path+task_type}/{args.dataset}/test_metrics_tox21.csv")
        
    elif args.dataset == "bace":
        tasks = ['Class']
        test_metrics_tox21 = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_tox21['AUC'] = roc_list
        test_metrics_tox21['AP'] = ap_list
        test_metrics_tox21['F1'] = f1_list
        test_metrics_tox21.to_csv(f"{args.save_path+task_type}/{args.dataset}/test_metrics_bace.csv")
        
    elif args.dataset == "bbbp":
        tasks = ['p_np']
        test_metrics_tox21 = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_tox21['AUC'] = roc_list
        test_metrics_tox21['AP'] = ap_list
        test_metrics_tox21['F1'] = f1_list
        test_metrics_tox21.to_csv(f"{args.save_path+task_type}/{args.dataset}/test_metrics_bace.csv")
        
    elif args.dataset == "another":
        tasks = ['activity']
        test_metrics_alk = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_alk['AUC'] = roc_list
        test_metrics_alk['AP'] = ap_list
        test_metrics_alk['F1'] = f1_list
        test_metrics_alk.to_csv(f"{args.save_path+args.dataset}/test_metrics.csv")
    else:
        raise ValueError("Invalid dataset name.")
    
def create_train_df(
    args, train_df, train_loss, train_auc, train_ap, train_f1, val_loss, val_auc, val_ap, val_f1, task_type, epoch
    ):
    train_df.at[epoch-1, "train_loss"] = train_loss
    train_df.at[epoch-1,"train_auc"] = train_auc
    train_df.at[epoch-1,"train_ap"] = train_ap
    train_df.at[epoch-1,"train_f1"] = train_f1
    train_df.at[epoch-1,"val_loss"] = val_loss
    train_df.at[epoch-1,"val_auc"] = val_auc
    train_df.at[epoch-1,"val_ap"] = val_ap
    train_df.at[epoch-1,"val_f1"] = val_f1
    train_df.to_csv(f"{args.save_path+task_type}/{args.dataset}/train_metrics.csv")
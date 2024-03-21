import pandas as pd
def create_test_df(args, test_auc, test_ap, test_f1):
    if args.dataset == "tox21":
        tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        test_metrics_tox21 = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_tox21['AUC'] = test_auc
        test_metrics_tox21['AP'] = test_ap
        test_metrics_tox21['F1'] = test_f1
        test_metrics_tox21.to_csv(f"{args.save_path+args.dataset}/test_metrics_tox21.csv")
    elif args.dataset == "alk":
        tasks = ['alk']
        test_metrics_alk = pd.DataFrame(columns=["AUC", "AP", "F1"], index=tasks)
        test_metrics_alk['AUC'] = test_auc
        test_metrics_alk['AP'] = test_ap
        test_metrics_alk['F1'] = test_f1
        test_metrics_alk.to_csv(f"{args.save_path+args.dataset}/test_metrics_alk.csv")
    else:
        raise ValueError("Invalid dataset name.")
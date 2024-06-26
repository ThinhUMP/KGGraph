import sys
import pathlib

root_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(root_dir)

import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
import numpy as np
from KGGraph.KGGModel.visualize import visualize_fgs
from KGGraph.KGGProcessor.split import scaffold_split_df
from KGGraph.KGGModel.finetune_utils import get_task_type
import argparse

import warnings

warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# def generate_scaffold(smiles, include_chirality=False):
#     mol = Chem.MolFromSmiles(smiles)
#     scaffold = MurckoScaffold.GetScaffoldForMol(mol)
#     scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
#     return scaffold_smiles


def smiles_to_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    rdk7 = RDKFingerprint(mol, maxPath=7, fpSize=4096)
    return maccs, ecfp4, rdk7


def prepare_fingerprints(df, smile_column, target_column):
    maccs_fps = []
    ecfp4_fps = []
    rdk7_fps = []
    labels = []
    for idx, row in df.iterrows():
        fingerprints = smiles_to_fingerprints(row[smile_column])
        if fingerprints is not None:
            maccs, ecfp4, rdk7 = fingerprints
            maccs_fps.append(maccs)
            ecfp4_fps.append(ecfp4)
            rdk7_fps.append(rdk7)
            labels.append(row[target_column])
    return maccs_fps, ecfp4_fps, rdk7_fps, labels


def train_and_evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rocauc = roc_auc_score(y_test, y_pred)
    return rocauc


def trainreg_and_evaluate_knn(X_train, X_test, y_train, y_test, n_neighbors=3):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, mae


def split_data(
    df,
    smile_column,
    method="scaffold",
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=42,
):
    if method == "scaffold":
        print("scaffold")
        return scaffold_split_df(df, smile_column, frac_train, frac_valid, frac_test)

    elif method == "random":
        train_df, temp_df = train_test_split(
            df, test_size=(frac_valid + frac_test), random_state=seed
        )
        valid_df, test_df = train_test_split(
            temp_df, test_size=(frac_test / (frac_valid + frac_test)), random_state=seed
        )
        print("random")
        return train_df, valid_df, test_df

    else:
        raise ValueError("Method must be either 'scaffold' or 'random'.")


# Đọc dữ liệu
def main():
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of kNN evaluation of embeddings"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bbbp",
        help="[bbbp, bace, sider, clintox, tox21, toxcast, hiv, muv, esol, freesolv, lipo, qm7, qm8, qm9]",
    )
    parser.add_argument(
        "--smile_column",
        type=str,
        default="mol",
        help="Column containing SMILES strings",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="Class",
        help="Column containing y values",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seeding for reproducibility in ramdom splitting",
    )
    args = parser.parse_args()

    # get task type
    task_type = get_task_type(args.dataset)

    # Read data
    df = pd.read_csv(f"Data/{task_type}/{args.dataset}/raw/{args.dataset}.csv")
    df = df[[args.smile_column, args.target_column]]
    df[args.smile_column] = df[args.smile_column].apply(Chem.MolFromSmiles)
    df.dropna(inplace=True)

    # Data split
    train_df, valid_df, test_df = split_data(
        df, smile_column=args.smile_column, method=args.split
    )

    # Prepare fingerprints
    maccs_fps_train, ecfp4_fps_train, rdk7_fps_train, y_train = prepare_fingerprints(
        train_df
    )
    maccs_fps_valid, ecfp4_fps_valid, rdk7_fps_valid, y_valid = prepare_fingerprints(
        valid_df
    )
    maccs_fps_test, ecfp4_fps_test, rdk7_fps_test, y_test = prepare_fingerprints(
        test_df
    )

    # Training and evaluating
    if task_type == "classification":
        rocauc_maccs = train_and_evaluate_knn(
            maccs_fps_train, maccs_fps_test, y_train, y_test
        )
        rocauc_ecfp4 = train_and_evaluate_knn(
            ecfp4_fps_train, ecfp4_fps_test, y_train, y_test
        )
        rocauc_rdk7 = train_and_evaluate_knn(
            rdk7_fps_train, rdk7_fps_test, y_train, y_test
        )

        print(f"ROC-AUC with MACCS fingerprints: {rocauc_maccs:.4f}")
        print(f"ROC-AUC with ECFP4 fingerprints: {rocauc_ecfp4:.4f}")
        print(f"ROC-AUC with RDK7 fingerprints: {rocauc_rdk7:.4f}")
    else:
        rmse_maccs, mae_maccs = trainreg_and_evaluate_knn(
            maccs_fps_train, maccs_fps_test, y_train, y_test
        )
        rmse_ecfp4, mae_ecfp4 = trainreg_and_evaluate_knn(
            ecfp4_fps_train, ecfp4_fps_test, y_train, y_test
        )
        rmse_rdk7, mae_rdk7 = trainreg_and_evaluate_knn(
            rdk7_fps_train, rdk7_fps_test, y_train, y_test
        )

        print(f"ROC-AUC with MACCS fingerprints: {rmse_maccs, mae_maccs}")
        print(f"ROC-AUC with ECFP4 fingerprints: {rmse_ecfp4, mae_ecfp4}")
        print(f"ROC-AUC with RDK7 fingerprints: {rmse_rdk7, mae_rdk7}")

    visualize_fgs(test_df)
    print("Done!")

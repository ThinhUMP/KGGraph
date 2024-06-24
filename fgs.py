import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from collections import defaultdict
import numpy as np
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

import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
    return scaffold_smiles

def scaffold_split(
    df,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=None,
):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    smiles_list = df['mol'].tolist()

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        # all_scaffolds[scaffold].append(i)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    # get train, valid, test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    print("train set", len(train_idx))
    print("valid set", len(valid_idx))
    print("test set", len(test_idx))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, valid_df, test_df


def smiles_to_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    rdk7 = RDKFingerprint(mol, maxPath=7, fpSize=4096)
    return maccs, ecfp4, rdk7

def prepare_fingerprints(df):
    maccs_fps = []
    ecfp4_fps = []
    rdk7_fps = []
    labels = []
    for idx, row in df.iterrows():
        fingerprints = smiles_to_fingerprints(row['mol'])
        if fingerprints is not None:
            maccs, ecfp4, rdk7 = fingerprints
            maccs_fps.append(maccs)
            ecfp4_fps.append(ecfp4)
            rdk7_fps.append(rdk7)
            labels.append(row['Class'])
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

def split_data(df, method='scaffold', frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42):
    if method == 'scaffold':
        print("scaffold")
        return scaffold_split(df, frac_train, frac_valid, frac_test)
        
    elif method == 'random':
        train_df, temp_df = train_test_split(df, test_size=(frac_valid + frac_test), random_state=seed)
        valid_df, test_df = train_test_split(temp_df, test_size=(frac_test / (frac_valid + frac_test)), random_state=seed)
        print("random")
        return train_df, valid_df, test_df
        
    else:
        raise ValueError("Method must be either 'scaffold' or 'random'.")

# Đọc dữ liệu

df= pd.read_csv("Data/classification/bace/raw/bace.csv")
# df = df[['smiles', 'p_np']]
# df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
# df = df[df["mol"].notnull()]
df.dropna(inplace=True)
print(df.shape)
# print(df['p_np'].unique())
# Lựa chọn phương pháp chia dữ liệu: 'scaffold' hoặc 'random'
method = 'scaffold'  # Thay thế bằng 'random' nếu muốn chia ngẫu nhiên

# Chia dữ liệu
train_df, valid_df, test_df = split_data(df, method=method)

# Chuẩn bị dấu vân tay cho tập huấn luyện
# maccs_fps_train, ecfp4_fps_train, rdk7_fps_train, y_train = prepare_fingerprints(train_df)
# maccs_fps_valid, ecfp4_fps_valid, rdk7_fps_valid, y_valid = prepare_fingerprints(valid_df)
# maccs_fps_test, ecfp4_fps_test, rdk7_fps_test, y_test = prepare_fingerprints(test_df)

# def check_class_balance(y_train, y_test):
#     unique_train = np.unique(y_train)
#     unique_test = np.unique(y_test)
#     print(unique_train, unique_test)
#     return set(unique_train) == set(unique_test)

# if not check_class_balance(y_train, y_test):
#     raise ValueError("Class imbalance detected. Ensure both classes are present in training and test sets.")

# Huấn luyện và đánh giá mô hình k-NN với từng bộ dấu vân tay
# rocauc_maccs = train_and_evaluate_knn(maccs_fps_train, maccs_fps_test, y_train, y_test)
# rocauc_ecfp4 = train_and_evaluate_knn(ecfp4_fps_train, ecfp4_fps_test, y_train, y_test)
# rocauc_rdk7 = train_and_evaluate_knn(rdk7_fps_train, rdk7_fps_test, y_train, y_test)

# print(f"ROC-AUC with MACCS fingerprints: {rocauc_maccs:.4f}")
# print(f"ROC-AUC with ECFP4 fingerprints: {rocauc_ecfp4:.4f}")
# print(f"ROC-AUC with RDK7 fingerprints: {rocauc_rdk7:.4f}")

# rmse_maccs, mae_maccs = trainreg_and_evaluate_knn(maccs_fps_train, maccs_fps_test, y_train, y_test)
# rmse_ecfp4, mae_ecfp4 = trainreg_and_evaluate_knn(ecfp4_fps_train, ecfp4_fps_test, y_train, y_test)
# rmse_rdk7, mae_rdk7 = trainreg_and_evaluate_knn(rdk7_fps_train, rdk7_fps_test, y_train, y_test)

# print(f"ROC-AUC with MACCS fingerprints: {rmse_maccs, mae_maccs}")
# print(f"ROC-AUC with ECFP4 fingerprints: {rmse_ecfp4, mae_ecfp4}")
# print(f"ROC-AUC with RDK7 fingerprints: {rmse_rdk7, mae_rdk7}")

def visualize_embeddings(df):
    maccs_fps, ecfp4_fps, rdk7_fps, y = prepare_fingerprints(df)
    
    # Combine all fingerprints into one array (Optional, if needed)
    fingerprints = {'MACCS': maccs_fps, 'ECFP4': ecfp4_fps, 'RDK7': rdk7_fps}
    
    custom_cmap = ListedColormap(['red', 'green'])
    
    for fp_name, fps in fingerprints.items():
        pca = PCA(n_components=50)
        embeddings_pca = pca.fit_transform(fps)
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(torch.tensor(embeddings_pca))
        
        plt.figure(figsize=(12, 10))
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=y,
            cmap=custom_cmap,
            s=50
        )
        
        # Create a custom legend
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
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
        
        plt.legend(
            handles=handles,
            title="Class",
            title_fontsize="13",
            loc="upper right",
            prop={"size": 12},
        )
        
        plt.title(f"t-SNE Visualization of {fp_name} Fingerprint", fontsize=20)
        plt.xlabel("t-SNE Dimension 1", fontsize=16)
        plt.ylabel("t-SNE Dimension 2", fontsize=16)
        
        plt.tight_layout()
        # plt.savefig(f"Data/fig/{fp_name}_fingerprint_tsne.png", dpi=600, bbox_inches='tight', transparent=False)
        plt.show()
visualize_embeddings(test_df)
print("Done!")



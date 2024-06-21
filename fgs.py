import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np

def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
    return scaffold_smiles

def scaffold_split(df, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=None):
    scaffolds = defaultdict(list)
    for i, smiles in enumerate(df['smiles']):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(i)

    scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
    train_size = int(len(df) * frac_train)
    valid_size = int(len(df) * frac_valid)
    
    train_indices, valid_indices, test_indices = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= train_size:
            train_indices += scaffold_set
        elif len(valid_indices) + len(scaffold_set) <= valid_size:
            valid_indices += scaffold_set
        else:
            test_indices += scaffold_set

    return df.iloc[train_indices], df.iloc[valid_indices], df.iloc[test_indices]

def smiles_to_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    rdk7 = RDKFingerprint(mol, fpSize=2048)
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

def train_and_evaluate_knn(X_train, X_valid, y_train, y_valid, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    return accuracy

def split_data(df, method='scaffold', frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=None):
    if method == 'scaffold':
        return scaffold_split(df, frac_train, frac_valid, frac_test, seed)
    elif method == 'random':
        train_df, temp_df = train_test_split(df, test_size=(frac_valid + frac_test), random_state=seed)
        valid_df, test_df = train_test_split(temp_df, test_size=(frac_test / (frac_valid + frac_test)), random_state=seed)
        return train_df, valid_df, test_df
    else:
        raise ValueError("Method must be either 'scaffold' or 'random'.")

# Đọc dữ liệu
df = pd.read_csv('data/classification/bace/raw/bace.csv')  # Thay thế 'dataset.csv' bằng tệp dữ liệu thực của bạn

# Lựa chọn phương pháp chia dữ liệu: 'scaffold' hoặc 'random'
method = 'scaffold'  # Thay thế bằng 'random' nếu muốn chia ngẫu nhiên

# Chia dữ liệu
train_df, valid_df, test_df = split_data(df, method=method)

# Chuẩn bị dấu vân tay cho tập huấn luyện
maccs_fps_train, ecfp4_fps_train, rdk7_fps_train, y_train = prepare_fingerprints(train_df)
maccs_fps_valid, ecfp4_fps_valid, rdk7_fps_valid, y_valid = prepare_fingerprints(valid_df)
maccs_fps_test, ecfp4_fps_test, rdk7_fps_test, y_test = prepare_fingerprints(test_df)

# Huấn luyện và đánh giá mô hình k-NN với từng bộ dấu vân tay
accuracy_maccs = train_and_evaluate_knn(maccs_fps_train, maccs_fps_valid, y_train, y_valid)
accuracy_ecfp4 = train_and_evaluate_knn(ecfp4_fps_train, ecfp4_fps_valid, y_train, y_valid)
accuracy_rdk7 = train_and_evaluate_knn(rdk7_fps_train, rdk7_fps_valid, y_train, y_valid)

print(f"Accuracy with MACCS fingerprints: {accuracy_maccs:.4f}")
print(f"Accuracy with ECFP4 fingerprints: {accuracy_ecfp4:.4f}")
print(f"Accuracy with RDK7 fingerprints: {accuracy_rdk7:.4f}")
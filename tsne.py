import torch
import torch.nn.functional as F

state_dict = torch.load("Data/classification/bace/bace_1.pth")

# Create a new dictionary
new_state_dict = {}

# Iterate over the state_dict
for name, param in state_dict.items():
    # If the key starts with 'gnn.x', remove the 'gnn.' prefix
    if name.startswith('gnn.'):
        new_name = name.replace('gnn.', '')
        new_state_dict[new_name] = param
    else:
        new_state_dict[name] = param

# Now new_state_dict contains the renamed keys
state_dict = new_state_dict

# Create a list of keys to remove
keys_to_remove = [key for key in state_dict if "graph_pred_linear" in key]

# Remove the keys from the state_dict
for key in keys_to_remove:
    del state_dict[key]
    
from KGGraph.KGGModel.graph_model import GNN
model = GNN(
    5,
    512,
    JK="last",
    drop_ratio=0.5,
    gnn_type="gin",
    x_features=7,
    edge_features=5,
)
model.load_state_dict(state_dict)

from KGGraph.KGGProcessor.finetune_dataset import MoleculeDataset
dataset = MoleculeDataset(
            "Data/" + "classification" + "/" + "bace",
            dataset="bace",
            decompose_type="motif",
            mask_node=False,
            mask_edge=False,
            mask_node_ratio=0.1,
            mask_edge_ratio=0.1,
            fix_ratio=False,
        )
print(dataset)

import pandas as pd
from KGGraph.KGGProcessor.split import scaffold_split
# data split
smiles_list = pd.read_csv(
    "Data/" + "classification" + "/" + "bace" + "/processed/smiles.csv",
    header=None,
)[0].tolist()
train_dataset, valid_dataset, test_dataset, (_, _, test_smiles) = (
    scaffold_split(
        dataset,
        smiles_list,
        null_value=0,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
    )
)
print("scaffold")

model.eval()
embeddings = model(test_dataset).detach().numpy()
print(embeddings.shape)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=data.y, cmap='viridis', s=50)
plt.colorbar()
plt.show()
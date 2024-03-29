from KGGraph import edge_feature, load_clintox_dataset 

smiles_list, mols_list, labels = load_clintox_dataset('./dataset/classification/clintox/raw/clintox.csv')
error_smiles = []
for idx, mol in enumerate(mols_list):
    # try:
    edge_index, edge_attr, directed_adj_matrix  = edge_feature(mol)
    # except:
    #     error_smiles.append(smiles_list[idx])
print(len(error_smiles))
        
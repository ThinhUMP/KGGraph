import os
import torch
import pandas as pd
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat
from pathlib import Path
import sys
import pandas as pd
# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))
from KGGraph.GraphEncode.x_feature import x_feature
from KGGraph.GraphEncode.edge_feature import edge_feature
from KGGraph.Chemistry.chemutils import get_mol
from KGGraph.Dataset.loader import load_tox21_dataset
from joblib import Parallel, delayed
from tqdm import tqdm
def feature(mol):
    x = x_feature(mol)
    edge_index, edge_attr = edge_feature(mol)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

class MoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset='tox21',
        empty=False
):
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'kgg_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset == 'tox21':
            smiles_list, mols_list, labels = load_tox21_dataset(self.raw_paths[0])
            data_result_list = Parallel(n_jobs=-1)(delayed(feature)(mol) for mol in tqdm(mols_list))
            for idx, data in enumerate(data_result_list):
                data.id = torch.tensor([idx])  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[idx])
                data_list.append(data)
                data_smiles_list.append(smiles_list[idx])

        elif self.dataset == 'dataset_x':
            # Update later
            pass

        else:
            raise ValueError(f'Dataset {self.dataset} is not supported')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,'smiles.csv'), index=False, header=False)

        if data_list:  # Ensure data_list is not empty
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            
if __name__ == '__main__':
    dataset = MoleculeDataset('./data/tox21/', dataset='tox21')
    print(dataset)
    print(dataset[0])
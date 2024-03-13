from KGGraph import x_feature, edge_feature, get_mol
from torch_geometric.data import Data
def feature(mol):
    x = x_feature(mol)
    edge_index, edge_attr = edge_feature(mol)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

class MyDataset(Dataset):
    
    def __init__(self, root, dataset, transform=None, pre_transform=None):
        self.dataset = dataset
        super(MyDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.root+ f'/{self.dataset}.csv')
        self.smiles = df['X'].tolist()
        mols = [get_mol(smile) for smile in self.smiles]
        data_list = Parallel(n_jobs=-1)(delayed(feature)(mol) for mol in tqdm((mols)))
        for i, data in enumerate(data_list):
            data.id = torch.tensor([i])
            data.y = torch.tensor([df.at[i, 'SR-p53']])
            torch.save(data, self.processed_dir + '/data_{}.pt'.format(i))

    def len(self):
        return len(self.smiles)  # Return the number of graphs

    def get(self, idx):
        # Load the idx-th graph
        data = torch.load(self.processed_dir + '/data_{}.pt'.format(idx))
        return data
if __name__ == '__main__':
    import time
    t1 = time.time()
    dataset = MyDataset(root='./data/tox21/', dataset='tox21')
    t2 = time.time()
    print(t2-t1)
    print(len(dataset))
    data = dataset[0]  # Get the first data point
    print(data)
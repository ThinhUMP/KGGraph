from KGGraph.Dataset.molecule_dataset import MoleculeDataset
import pandas as pd
from KGGraph.Dataset.scaffold_split import scaffold_split
from torch_geometric.data import DataLoader
from KGGraph.Model.Architecture.gin import GIN
from KGGraph.Model.Train.train_utils import train_epochs
from KGGraph.Model.Train.visualize import plot_loss
import warnings
warnings.filterwarnings('ignore')

def main():
    # Processing dataset
    dataset = MoleculeDataset('./data/tox21/', dataset='tox21')
    print(dataset)
    print('dataset[0]', dataset[0])
    
    # Scaffold split
    print("-------scaffold split----------")
    smiles_list = pd.read_csv('data/tox21/processed/smiles.csv', header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset, _ = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    
    #Load dataset
    print("----------Load dataset----------")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 8)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers = 8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 8)

    print('-----------training----------')
        # Training GIN for 10 epochs
    epochs = 10
    model = GIN(dim_h=1024)

    # Remember to change the path if you want to keep the previously trained model
    gin_train_loss, gin_val_loss, gin_train_target, gin_train_y_target = train_epochs(
        epochs, model, train_loader, test_loader, "data/GIN_model.pt"
    )

if __name__ == '__main__':
    main()
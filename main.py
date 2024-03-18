from KGGraph.Dataset.molecule_dataset import MoleculeDataset
import pandas as pd
from KGGraph.Dataset.scaffold_split import scaffold_split
from torch_geometric.data import DataLoader
from KGGraph.Model.Architecture.gin import GIN
from KGGraph.Model.Train.train_utils import train_epochs
from KGGraph.Model.Train.visualize import plot_metrics
import warnings
warnings.filterwarnings('ignore')

def main():
    # Processing dataset
    dataset = MoleculeDataset('./data/tox21/', dataset='tox21')
    print(dataset)
    print('dataset[0]', dataset[0])
    
    # Scaffold split
    print("-------scaffold split----------")
    smiles_list = pd.read_csv('data/alk/processed/smiles.csv', header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset, _ = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    
    #Load dataset
    print("----------Load dataset----------")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 8)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers = 8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 8)

    print('-----------training----------')
        # Training GIN for 10 epochs
    epochs = 4
    model = GIN(dim_h=512)

    # Remember to change the path if you want to keep the previously trained model
    train_loss_list, train_auc_list, train_f1_list, train_ap_list, val_loss_list, val_auc_list, val_f1_list, val_ap_list =train_epochs(
        epochs, model, train_loader, test_loader, "data/GIN_model.pt"
    )
    plot_metrics(train_loss_list, val_loss_list, train_auc_list, val_auc_list, train_f1_list, val_f1_list, train_ap_list, val_ap_list)
if __name__ == '__main__':
    main()
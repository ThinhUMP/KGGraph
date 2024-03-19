from KGGraph.Dataset.molecule_dataset import MoleculeDataset
import pandas as pd
from KGGraph.Dataset.scaffold_split import scaffold_split
from torch_geometric.data import DataLoader
from KGGraph.Model.Architecture.class_gin import Net
from KGGraph.Model.Train.train_utils import train, evaluate
from KGGraph.Model.Train.visualize import plot_metrics
import torch
import warnings
warnings.filterwarnings('ignore')

def main():
    # Processing dataset
    dataset = MoleculeDataset('./data/alk/', dataset='alk')
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
    model = Net(dim_h=512)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    train_auc_list, test_auc_list = [], []
    for epoch in range(1, epochs+1):
        print('====epoch:',epoch)
        
        train(model, device, train_loader, optimizer)

        print('====Evaluation')
        train_auc, train_loss = evaluate(model, device, train_loader)
        # val_auc, val_loss = evaluate(model, device, val_loader)
        test_auc, test_loss = evaluate(model, device, test_loader)
        test_auc_list.append(float('{:.4f}'.format(test_auc)))
        train_auc_list.append(float('{:.4f}'.format(train_auc)))

        
        print("train_auc: %f test_auc: %f" %(train_auc, test_auc))
        print("train_loss: %f test_loss: %f" %(train_loss, test_loss))
if __name__ == '__main__':
    main()
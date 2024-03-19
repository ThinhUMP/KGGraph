import pandas as pd
from pathlib import Path
import sys
import pandas as pd
# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))
from KGGraph.Chemistry.chemutils import get_mol
        
def load_tox21_dataset(input_path):
    tox21_dataset = pd.read_csv(input_path, sep=',')
    smiles_list = tox21_dataset['smiles']
    mols_list = [get_mol(smile) for smile in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = tox21_dataset[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values

def load_alk_dataset(input_path):
    alk_dataset = pd.read_csv(input_path, sep=',')
    smiles_list = alk_dataset['Canomicalsmiles']
    mols_list = [get_mol(smile) for smile in smiles_list]
    tasks = ['activity']
    labels = alk_dataset[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values

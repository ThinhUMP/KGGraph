import pandas as pd
from KGGraph.KGGChem.atom_utils import get_mol
from rdkit import Chem
ecoli = pd.read_csv("Data/regression/ecoli/raw/ecoli.csv")
ecoli["mol"] = ecoli["Smiles"].apply(get_mol)

from rdkit import Chem, DataStructs
import numpy as np
def RDKFp(mol, maxPath=5, fpSize=2048, nBitsPerHash=2):
    fp = Chem.RDKFingerprint(mol, maxPath=maxPath, fpSize=fpSize, nBitsPerHash=nBitsPerHash)
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar

RDKF = ecoli.copy()
RDKF["FPs"] = RDKF.mol.apply(RDKFp, maxPath=7, fpSize=4096)
X = np.stack(RDKF.FPs.values)
df = pd.DataFrame(X)

if __name__ == "__main__":
    print(df)
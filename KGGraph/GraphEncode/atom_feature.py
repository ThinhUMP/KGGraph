from typing import Union
from rdkit import Chem
import sys
import pathlib
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.Chemistry.features import (
    get_symbol, get_hybridization, get_cip_code, is_chiral_center,
    get_formal_charge, get_total_num_hs, get_total_valence, get_num_radical_electrons,
    get_degree, is_aromatic, is_hetero, is_hydrogen_donor, is_hydrogen_acceptor,
    get_ring_size, is_in_ring, get_ring_membership_count, is_in_aromatic_ring, get_electronegativity,
    ELECTRONEGATIVITY
)

class AtomFeature():
    
    def __init__(self, smile):
        self.smile = smile
        self.molecule = get_s
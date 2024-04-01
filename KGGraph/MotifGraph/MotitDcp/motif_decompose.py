from rdkit.Chem import BRICS
from rdkit import Chem
from KGGraph.Chemistry.chemutils import get_clique_mol
import pathlib
import sys
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
class MotifDecomposition:

    @staticmethod
    def defragment(mol):
        """
        Perform motif decomposition on the molecule.

        Returns:
        list: A list of atom indices representing the decomposed motifs.
        """
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]]

        cliques = MotifDecomposition._initial_cliques(mol)
        cliques = MotifDecomposition._apply_brics_breaks(cliques, mol)
        cliques = MotifDecomposition._merge_cliques(cliques, mol)
        cliques = MotifDecomposition._refine_cliques(cliques, mol)
        return cliques

    @staticmethod
    def _initial_cliques(mol: Chem.Mol):
        """
        Create initial cliques based on the bonds of the molecule.

        Returns:
        list: A list of initial cliques.
        """
        cliques = [[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()] for bond in mol.GetBonds()]
        return cliques
    
    @staticmethod
    def _apply_brics_breaks(cliques, mol):
        """
        Apply BRICS rules to break bonds and update cliques.

        Parameters:
        cliques (list): The current list of cliques.

        Returns:
        list: Updated list of cliques after applying BRICS breaks.
        """
        res = list(BRICS.FindBRICSBonds(mol))
        for bond in res:
            bond_indices = [bond[0][0], bond[0][1]]
            if bond_indices in cliques:
                cliques.remove(bond_indices)
            else:
                cliques.remove(bond_indices[::-1])  # Reverse indices if not found in order
            cliques.extend([[bond[0][0]], [bond[0][1]]])
        return cliques
    
    @staticmethod
    def _merge_cliques(cliques, mol):
        """
        Merge overlapping cliques.

        Parameters:
        cliques (list): The current list of cliques.

        Returns:
        list: Updated list of cliques after merging.
        """
        n_atoms = mol.GetNumAtoms()
        for i in range(len(cliques) - 1):
            if i >= len(cliques):
                break
            for j in range(i + 1, len(cliques)):
                if j >= len(cliques):
                    break
                if set(cliques[i]) & set(cliques[j]):  # Intersection is not empty
                    cliques[i] = list(set(cliques[i]) | set(cliques[j]))  # Union
                    cliques[j] = []
            cliques = [c for c in cliques if c]
        cliques = [c for c in cliques if n_atoms> len(c) > 0]
        return cliques
    
    @staticmethod
    def _refine_cliques(cliques, mol):
        """
        Refine cliques to consider symmetrically equivalent substructures.

        Parameters:
        cliques (list): The current list of cliques.

        Returns:
        list: Refined list of cliques.
        """
        n_atoms = mol.GetNumAtoms()
        num_cli = len(cliques)
        ssr_mol = Chem.GetSymmSSSR(mol)
        for i in range(num_cli):
            c = cliques[i]
            cmol = get_clique_mol(mol, c)
            ssr = Chem.GetSymmSSSR(cmol)
            if len(ssr)>1: 
                for ring in ssr_mol:
                    if set(list(ring)) <= set(c):
                        cliques.append(list(ring))
                cliques[i]=[]
    
        cliques = [c for c in cliques if n_atoms> len(c) > 0]
        return cliques
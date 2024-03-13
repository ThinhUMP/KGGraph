from rdkit.Chem import BRICS
from rdkit import Chem
from KGGraph.Chemistry.chemutils import get_clique_mol
import pathlib
import sys
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
class MotifDecomposition:
    def __init__(self, mol):
        """
        Initialize the MotifDecomposition class with a molecule.

        Parameters:
        mol (rdkit.Chem.Mol): The molecule to be decomposed.
        """
        self.mol = mol
        self.n_atoms = mol.GetNumAtoms()

    def defragment(self):
        """
        Perform motif decomposition on the molecule.

        Returns:
        list: A list of atom indices representing the decomposed motifs.
        """
        if self.n_atoms == 1:
            return [[0]]

        cliques = self._initial_cliques()
        cliques = self._apply_brics_breaks(cliques)
        cliques = self._merge_cliques(cliques)
        cliques = self._refine_cliques(cliques)
        return cliques

    def _initial_cliques(self):
        """
        Create initial cliques based on the bonds of the molecule.

        Returns:
        list: A list of initial cliques.
        """
        cliques = [[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()] for bond in self.mol.GetBonds()]
        return cliques

    def _apply_brics_breaks(self, cliques):
        """
        Apply BRICS rules to break bonds and update cliques.

        Parameters:
        cliques (list): The current list of cliques.

        Returns:
        list: Updated list of cliques after applying BRICS breaks.
        """
        res = list(BRICS.FindBRICSBonds(self.mol))
        for bond in res:
            bond_indices = [bond[0][0], bond[0][1]]
            if bond_indices in cliques:
                cliques.remove(bond_indices)
            else:
                cliques.remove(bond_indices[::-1])  # Reverse indices if not found in order
            cliques.extend([[bond[0][0]], [bond[0][1]]])
        return cliques

    def _merge_cliques(self, cliques):
        """
        Merge overlapping cliques.

        Parameters:
        cliques (list): The current list of cliques.

        Returns:
        list: Updated list of cliques after merging.
        """
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
        return cliques

    def _refine_cliques(self, cliques):
        """
        Refine cliques to consider symmetrically equivalent substructures.

        Parameters:
        cliques (list): The current list of cliques.

        Returns:
        list: Refined list of cliques.
        """
        refined_cliques = []
        ssr_mol = Chem.GetSymmSSSR(self.mol)
        for c in cliques:
            cmol = get_clique_mol(self.mol, c)
            ssr = Chem.GetSymmSSSR(cmol)
            if len(ssr) > 1:
                for ring in ssr_mol:
                    if set(ring) <= set(c):
                        refined_cliques.append(list(ring))
                        c = list(set(c) - set(ring))
            if c:
                refined_cliques.append(c)
        cliques = [c for c in refined_cliques if 0 < len(c) < self.n_atoms]
        return cliques
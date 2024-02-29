from rdkit import Chem
from rdkit.Chem import BRICS
from typing import List, Tuple

class MotifDecomposition:
    @staticmethod
    def create_initial_cliques(mol: Chem.Mol) -> List[List[int]]:
        """
        Create initial cliques for each bond in the molecule.

        Parameters:
        mol (Chem.Mol): RDKit molecule object.

        Returns:
        List[List[int]]: Initial list of cliques.
        """
        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            cliques.append([a1, a2])
        return cliques

    @staticmethod
    def apply_brics_breaks(mol: Chem.Mol, cliques: List[List[int]]) -> List[List[int]]:
        """
        Apply BRICS breaks to the molecule and update cliques.

        Parameters:
        mol (Chem.Mol): RDKit molecule object.
        cliques (List[List[int]]): Existing list of cliques.

        Returns:
        List[List[int]]: Updated list of cliques after BRICS breaks.
        """
        res = list(BRICS.FindBRICSBonds(mol))
        if len(res) != 0:
            for bond in res:
                try:
                    cliques.remove([bond[0][0], bond[0][1]])
                except ValueError:
                    cliques.remove([bond[0][1], bond[0][0]])
                cliques.extend([[bond[0][0]], [bond[0][1]]])
        return cliques

    @staticmethod
    def merge_cliques(cliques: List[List[int]], n_atoms: int) -> List[List[int]]:
        """
        Merge cliques with common elements.

        Parameters:
        cliques (List[List[int]]): List of cliques.
        n_atoms (int): Number of atoms in the molecule.

        Returns:
        List[List[int]]: Merged list of cliques.
        """
        for i in range(len(cliques) - 1):
            for j in range(i + 1, len(cliques)):
                if set(cliques[i]) & set(cliques[j]):  # Check for intersection
                    cliques[i] = list(set(cliques[i]) | set(cliques[j]))  # Merge cliques
                    cliques[j] = []

        # Remove empty cliques and cliques with all atoms
        return [c for c in cliques if 0 < len(c) < n_atoms]

    @staticmethod
    def refine_cliques(mol: Chem.Mol, cliques: List[List[int]]) -> List[List[int]]:
        """
        Refine cliques based on ring structures.

        Parameters:
        mol (Chem.Mol): RDKit molecule object.
        cliques (List[List[int]]): List of cliques.

        Returns:
        List[List[int]]: Refined list of cliques.
        """
        ssr_mol = Chem.GetSymmSSSR(mol)
        n_atoms = mol.GetNumAtoms()
        for i, c in enumerate(cliques):
            cmol = Chem.PathToSubmol(mol, c)
            ssr = Chem.GetSymmSSSR(cmol)
            if len(ssr) > 1:
                for ring in ssr_mol:
                    if set(list(ring)) <= set(c):
                        cliques.append(list(ring))
                cliques[i] = []

        # Remove empty cliques and cliques with all atoms
        return [c for c in cliques if 0 < len(c) < n_atoms]

    def defragment(self, mol: Chem.Mol) -> List[List[int]]:
        """
        Perform motif decomposition on a molecule.

        Parameters:
        mol (Chem.Mol): RDKit molecule object.

        Returns:
        List[List[int]]: List of cliques representing motif decomposition.
        """
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]]

        cliques = self.create_initial_cliques(mol)
        cliques = self.apply_brics_breaks(mol, cliques)
        cliques = self.merge_cliques(cliques, n_atoms)
        cliques = self.refine_cliques(mol, cliques)
        return cliques

from rdkit.Chem import BRICS
from rdkit import Chem
import pathlib
import sys
from typing import List, Tuple, Set
from KGGraph.KGGChem.chemutils import get_clique_mol

root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)


class TMotifDecomposition:
    @staticmethod
    def _initial_cliques(mol: Chem.Mol):
        """
        Create initial cliques based on the bonds of the molecule.

        Returns:
        list: A list of initial cliques.
        """
        cliques = [
            [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
            for bond in mol.GetBonds()
        ]
        return cliques

    @staticmethod
    def _apply_brics_breaks(cliques, mol):
        """
        Apply BRICS rules to break bonds and update cliques.

        Args:
        cliques (list): The current list of cliques.

        Returns:
        list: Updated list of cliques after applying BRICS breaks.
        """
        res = list(BRICS.FindBRICSBonds(mol))
        res_list = [[bond[0][0], bond[0][1]] for bond in res]

        for bond in res:
            bond_indices = [bond[0][0], bond[0][1]]
            if bond_indices in cliques:
                cliques.remove(bond_indices)
            else:
                cliques.remove(
                    bond_indices[::-1]
                )  # Reverse indices if not found in order
            cliques.extend([[bond[0][0]], [bond[0][1]]])
        return cliques, res_list

    @staticmethod
    def _break_ring_bonds(mol, cliques, res_list):
        """
        Breaking ring and non-ring atoms.

        Args:
        cliques (list): The current list of cliques.
        mol (Chem.Mol): RDKit molecule object.

        Returns:
        list: Updated list of cliques after breaking ring and non-ring atoms.
        """

        for c in cliques:
            if len(c) > 1:
                atom1_in_ring = mol.GetAtomWithIdx(c[0]).IsInRing()
                atom2_in_ring = mol.GetAtomWithIdx(c[1]).IsInRing()
                if atom1_in_ring and not atom2_in_ring:
                    cliques.remove(c)
                    cliques.append([c[1]])
                    res_list.append(c)
                elif atom2_in_ring and not atom1_in_ring:
                    cliques.remove(c)
                    cliques.append([c[0]])
                    res_list.append(c)
        return cliques, res_list

    @staticmethod
    def _generate_mark_pattern(mol: Chem.Mol) -> Set[int]:
        """
        Generate marks for atoms that are part of
        identified functional groups in a molecule.

        Parameters:
        mol (Chem.Mol): RDKit molecule object to analyze.

        Returns:
        Set[int]: A set of atom indices that are part of identified functional groups.
        """
        PATTERNS = {
            "HETEROATOM": "[!#6]",  # Matches atoms that are not carbon (heteroatoms).
        }
        PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in PATTERNS.items()}

        marks = []

        for name, patt in PATTERNS.items():
            for sub in mol.GetSubstructMatches(patt):
                atom = mol.GetAtomWithIdx(sub[0])
                if not atom.IsInRing():
                    marks.append(list(sub))

        return marks

    @staticmethod
    def _find_carbonyl(mol: Chem.Mol) -> Tuple[List[List[int]], List[Tuple[int]]]:
        """
        Identify carbonyl groups in the molecule
        and merge adjacent or overlapping CO groups.

        Parameters:
        mol (Chem.Mol): RDKit molecule object.

        Returns:
        Tuple[List[List[int]], List[Tuple[int]]]:
        List of merged CO groups and list of merged COO groups.
        """
        CO = [
            list(CO_group)
            for CO_group in mol.GetSubstructMatches(
                Chem.MolFromSmarts("[C;D3](=[O,N,S])")
            )
        ]
        COO = [
            list(COO_group)
            for COO_group in (
                mol.GetSubstructMatches(Chem.MolFromSmarts("[C;D3](=[O,N,S])[O,N,S]"))
            )
        ]

        for subCOO in COO:
            for subCO in CO:
                if len(set(subCO) & set(subCOO)) == 2:
                    CO.remove(subCO)
            if mol.GetAtomWithIdx(subCO[0]).IsInRing() and subCO in CO:
                CO.remove(subCO)

        return CO, COO

    @staticmethod
    def _merge_functional_groups(CO, COO, marks, mol):
        list_of_functional_groups = []
        for value in CO:
            list_of_functional_groups.append(value)
        for value in COO:
            list_of_functional_groups.append(value)
        for value in marks:
            list_of_functional_groups.append(value)
        functional_groups = TMotifDecomposition._merge_cliques(
            list_of_functional_groups, mol
        )
        return functional_groups

    def _find_functional_group(functional_groups, cliques, mol):
        """
        Find the functional groups in the molecule.

        Args:
        functional_groups (List[List[int]]): The list of functional groups.
        cliques (List[List[int]]): The list of cliques.

        Returns:
        """
        res = []
        for value in functional_groups:
            atom = mol.GetAtomWithIdx(value[0])
            for bond in atom.GetBonds():
                begin_atom = bond.GetBeginAtomIdx()
                end_atom = bond.GetEndAtomIdx()
                if (
                    begin_atom in value
                    and end_atom not in value
                    and not bond.GetBeginAtom().IsInRing()
                    and not bond.GetEndAtom().IsInRing()
                ):
                    res.append([begin_atom, end_atom])
                elif (
                    begin_atom not in value
                    and end_atom in value
                    and not bond.GetBeginAtom().IsInRing()
                    and not bond.GetEndAtom().IsInRing()
                ):
                    res.append([begin_atom, end_atom])
        # print(res)
        for bond in res:
            bond_indices = [bond[0], bond[1]]
            if bond_indices in cliques:
                cliques.remove(bond_indices)
            elif bond_indices[::-1] in cliques:
                cliques.remove(
                    bond_indices[::-1]
                )  # Reverse indices if not found in order
            cliques.extend([[bond[0]], [bond[1]]])
        return cliques

    @staticmethod
    def _merge_cliques(cliques, mol):
        """
        Merge overlapping cliques.

        Args:
        cliques (list): The current list of cliques.

        Returns:
        list: Updated list of cliques after merging.
        """
        n_atoms = mol.GetNumAtoms()
        for i in range(len(cliques) - 1):
            # if i >= len(cliques):
            #     break
            for j in range(i + 1, len(cliques)):
                # if j >= len(cliques):
                #     break
                if set(cliques[i]) & set(cliques[j]):  # Intersection is not empty
                    cliques[i] = list(set(cliques[i]) | set(cliques[j]))  # Union
                    cliques[j] = []
            cliques = [c for c in cliques if c]
        cliques = [c for c in cliques if n_atoms >= len(c) > 0]
        return cliques

    @staticmethod
    def _refine_cliques(cliques, mol):
        """
        Refine cliques to consider symmetrically equivalent substructures.

        Args:
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
            if len(ssr) > 1:
                for ring in ssr_mol:
                    if set(list(ring)) <= set(c):
                        cliques.append(list(ring))
                cliques[i] = []

        cliques = [c for c in cliques if n_atoms > len(c) > 0]
        return cliques

    @staticmethod
    def _find_edges(cliques, res_list):
        """
        Find edges based on the breaks.

        Args:
        cliques (List[List[int]]): The list of cliques.
        res_list (List[Tuple]): BRICS breaks result.

        Returns:
        List[Tuple[int, int]]: List of edges representing the breaks.
        """
        edges = []
        for bond in res_list:
            c1, c2 = None, None  # Initialize c1 and c2
            for c in range(len(cliques)):
                if bond[0] in cliques[c]:
                    c1 = c
                if bond[1] in cliques[c]:
                    c2 = c
            if c1 is not None and c2 is not None:
                edges.append((c1, c2))
        for c in range(len(cliques)):
            for i in range(c + 1, len(cliques)):
                if set(cliques[c]) & set(cliques[i]):
                    c1, c2 = c, i
                    edges.append((c1, c2))
        return edges

    @staticmethod
    def defragment(mol):
        """
        Perform motif decomposition on the molecule.

        Returns:
        list: A list of atom indices representing the decomposed motifs.
        """
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = TMotifDecomposition._initial_cliques(mol)
        cliques, res_list = TMotifDecomposition._apply_brics_breaks(cliques, mol)
        cliques, res_list = TMotifDecomposition._break_ring_bonds(
            mol, cliques, res_list
        )
        # print("break ring bonds", cliques)
        marks = TMotifDecomposition._generate_mark_pattern(mol)
        CO, COO = TMotifDecomposition._find_carbonyl(mol)
        merged_functional_groups = TMotifDecomposition._merge_functional_groups(
            CO, COO, marks, mol
        )
        # print("merged functional groups: ", merged_functional_groups)
        cliques = TMotifDecomposition._find_functional_group(
            merged_functional_groups, cliques, mol
        )
        # print("functional group", cliques)
        cliques = TMotifDecomposition._merge_cliques(cliques, mol)
        cliques = TMotifDecomposition._refine_cliques(cliques, mol)

        edges = TMotifDecomposition._find_edges(cliques, res_list)

        return cliques, edges


if __name__ == "__main__":
    smile = "CC(N)C1N(C)C2=CC=CC(C(OCC)=O)=C2C1C(C)=O"
    # smile = "C1CCCCC12OCCO2"
    mol = Chem.MolFromSmiles(smile)
    cliques, edges = TMotifDecomposition.defragment(mol)
    print(cliques)
    print(edges)

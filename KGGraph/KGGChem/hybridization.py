from rdkit import Chem
from .atom_features import (
    get_degree,
    get_total_num_hs,
    get_hybridization,
)

# five features are in the order of (numbers of orbital s, numbers of orbital p,
# number of orbital d, total neighbors including hydrogens, number of lone pairs)
HYBRIDIZATION = {
    "UNSPECIFIED": [1]+[0, 0, 0, 0, 0, 0, 1],
    "S": [2]+[1, 0, 0, 0, 0, 0, 0],
    "SP": [3]+[0, 1, 0, 0, 0, 0, 0],
    "SP2": [4]+[0, 0, 1, 0, 0, 0, 0],
    "SP3": [5]+[0, 0, 0, 1, 0, 0, 0],
    "SP3D": [6]+[0, 0, 0, 0, 1, 0, 0],
    "SP3D2": [7]+[0, 0, 0, 0, 0, 1, 0],
}

max_bond_hybridization = {
    "SP3D2": 6,
    "SP3D": 5,
    "SP3": 4,
    "SP2": 3,
    "SP": 2,
    "S": 1,
}


class HybridizationFeaturize:
    """
    Class to compute hybridization features for a given dataset of molecules.
    """

    @staticmethod
    def total_sigma_bond(atom: Chem.Atom) -> int:
        """
        Compute the total number of single bonds for a given atom, including the bonds with hydrogen atoms.

        Parameters:
        atom (Chem.Atom): The atom for which the total number of single bonds is to be computed.

        Returns:
        int: The total number of single bonds for the given atom.
        """
        total_sigma_bond = get_degree(atom) + get_total_num_hs(atom)
        return total_sigma_bond

    @staticmethod
    def num_bond_hybridization(atom: Chem.Atom) -> int:
        """
        Compute the number of bonds involved in hybridization for a given atom based on the atom's hybridization state.

        Parameters:
        atom (Chem.Atom): The atom for which the number of bonds involved in hybridization is to be computed.

        Returns:
        int: The number of bonds involved in hybridization for the given atom.
        """
        num_bonds_hybridization = max_bond_hybridization.get(get_hybridization(atom), 0)
        return num_bonds_hybridization

    @staticmethod
    def num_lone_pairs(atom: Chem.Atom) -> int:
        """
        Calculate the number of lone pairs on a given atom. This method estimates the number of lone pairs by subtracting the total number
        of single bonds (including those with hydrogens) from the atom's hybridization-based expected bonding capacity. The calculation assumes
        that each atom has a fixed bonding capacity based on its hybridization state (sp, sp2, sp3, etc.), and any valence electrons not involved
        in single bonding can be considered as part of lone pairs.

        Parameters:
        atom (Chem.Atom): The atom for which the number of lone pairs is to be computed. This atom should be part of a molecule object.

        Returns:
        int: The estimated number of lone pairs on the atom. The value is computed based on the atom's hybridization and its single bonds.

        Note:
        This method relies on the `num_bond_hybridization` and `total_sigma_bond` methods from the `HybridizationFeaturize` class. Ensure that
        these methods correctly compute the atom's expected bonding capacity based on hybridization and the actual count of single bonds,
        respectively, for accurate results.
        """
        num_lone_pairs = HybridizationFeaturize.num_bond_hybridization(
            atom
        ) - HybridizationFeaturize.total_sigma_bond(atom)
        return num_lone_pairs

    @staticmethod
    def feature(atom: Chem.Atom) -> tuple[int, int, list[int]]:
        """
        Compute a feature vector for a given atom, including the total number of single bonds, the number of lone pairs,
        and a predefined feature vector based on the atom's hybridization characteristics. This vector is intended to capture
        aspects of the atom that are relevant to its chemical behavior and properties.

        Parameters:
        atom (Chem.Atom): The atom for which the feature vector is to be computed.

        Returns:
        tuple[int, int, list[int]]: A tuple containing the total number of single bonds to the atom (including hydrogen atoms),
        the number of lone electron pairs on the atom, and a list representing the hybridization feature vector. The hybridization
        feature vector is predefined and retrieved based on the total number of single bonds and the number of lone pairs.
        """
        total_sigma_bonds = HybridizationFeaturize.total_sigma_bond(atom)
        num_lone_pairs = HybridizationFeaturize.num_lone_pairs(atom)
        hybri_feat = HYBRIDIZATION.get(
            (total_sigma_bonds, num_lone_pairs), [0, 0, 0, 0, 0]
        )  # features for UNSPECIFIED hybridization is [0,0,0,0,0]

        return total_sigma_bonds, num_lone_pairs, hybri_feat
    
def HybridizationOnehot(atom):
    return HYBRIDIZATION.get(get_hybridization(atom), [0, 0, 0, 0, 0, 0, 1])

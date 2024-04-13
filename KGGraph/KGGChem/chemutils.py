from rdkit import Chem
from typing import Optional, List, Set, Tuple
from .atom_utils import copy_atom, idxfunc, set_atommap, sanitize

def get_leaves(mol: Chem.Mol) -> List[int]:
    """
    Identify the leaf atoms (degree 1) and rings in a molecule.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.

    Returns:
    - List[int]: A list of indices of leaf atoms and rings.
    """
    leaf_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() == 1]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( set([a1,a2]) )

    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(rings)

    leaf_rings = []
    
    for r in rings:
        inters = [c for c in clusters if r != c and len(r & c) > 0]
        if len(inters) > 1: continue
        nodes = [i for i in r if mol.GetAtomWithIdx(i).GetDegree() == 2]
        leaf_rings.append( max(nodes) )

    return leaf_atoms + leaf_rings

#mol must be RWMol object
def get_sub_mol(mol: Chem.Mol, sub_atoms: List[int]) -> Chem.Mol:
    """
    Extract a sub-molecule from the given molecule.

    Parameters:
    - mol (Chem.Mol): The original molecule.
    - sub_atoms (List[int]): A list of atom indices to include in the sub-molecule.

    Returns:
    - Chem.Mol: The sub-molecule.
    """
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()

def copy_edit_mol(mol: Chem.Mol) -> Chem.Mol:
    """
    Create a deep copy of the given molecule.

    Parameters:
    - mol (Chem.Mol): The molecule to copy.

    Returns:
    - Chem.Mol: A copy of the molecule.
    """
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
        #if bt == Chem.rdchem.BondType.AROMATIC and not aromatic:
        #    bt = Chem.rdchem.BondType.SINGLE
    return new_mol

def get_clique_mol(mol: Chem.Mol, atoms: List[int]) -> Chem.Mol:
    """
    Generate a molecule fragment based on a list of atom indices.

    Parameters:
    - mol (Chem.Mol): The original molecule.
    - atoms (List[int]): A list of atom indices to form the fragment.

    Returns:
    - Chem.Mol: The molecule fragment.
    """
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    # smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=False)
    # Chem.Kekulize(smiles, clearAromaticFlags=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) 
    #if tmp_mol is not None: new_mol = tmp_mol
    return new_mol

def get_assm_cands(mol: Chem.Mol, atoms: List[int], inter_label: List[Tuple[int, str]], cluster: List[int], inter_size: int) -> List:
    """
    Get assembly candidates for a molecule.

    Parameters:
    - mol (Chem.Mol): The original molecule.
    - atoms (List[int]): List of atom indices.
    - inter_label (List[Tuple[int, str]]): Labels of connected atoms and their SMILES.
    - cluster (List[int]): Atom indices in the parent cluster.
    - inter_size (int): Length of the inter_label.

    Returns:
    - List: A list of assembly candidates.
    ==Explain==
    idxfunc(): reduce 1 in the index to get the right atom index
    ranking point for atom in cluster
    """
    
    atoms = list(set(atoms))
    mol = get_clique_mol(mol, atoms)
    atom_map = [idxfunc(atom) for atom in mol.GetAtoms()]
    mol = set_atommap(mol)
    rank = Chem.CanonicalRankAtoms(mol, breakTies=False)
    rank = { x:y for x,y in zip(atom_map, rank) }

    pos, icls = zip(*inter_label)
    if inter_size == 1:
        cands = [pos[0]] + [ x for x in cluster if rank[x] != rank[pos[0]] ] 
    
    elif icls[0] == icls[1]: #symmetric case
        shift = cluster[inter_size - 1:] + cluster[:inter_size - 1]
        cands = zip(cluster, shift)
        cands = [pos] + [ (x,y) for x,y in cands if (rank[min(x,y)],rank[max(x,y)]) != (rank[min(pos)], rank[max(pos)]) ]
    else: 
        shift = cluster[inter_size - 1:] + cluster[:inter_size - 1]
        cands = zip(cluster + shift, shift + cluster)
        cands = [pos] + [ (x,y) for x,y in cands if (rank[x],rank[y]) != (rank[pos[0]], rank[pos[1]]) ]

    return cands

def get_inter_label(mol: Chem.Mol, atoms: List[int], inter_atoms: Set[int]) -> Tuple[Chem.Mol, List[Tuple[int, str]]]:
    """
    Get intersection labels for a molecule.

    Parameters:
    - mol (Chem.Mol): The original molecule.
    - atoms (List[int]): The clique of cluster.
    - inter_atoms (Set[int]): Intersecting atoms with the parent node.

    Returns:
    - Tuple[Chem.Mol, List[Tuple[int, str]]]: A tuple containing the new molecule and intersection labels.
    """
    new_mol = get_clique_mol(mol, atoms)
    if new_mol.GetNumBonds() == 0: 
        inter_atom = list(inter_atoms)[0]
        for a in new_mol.GetAtoms():
            a.SetAtomMapNum(0)
        return new_mol, [ (inter_atom, Chem.MolToSmiles(new_mol)) ]

    inter_label = []
    for a in new_mol.GetAtoms():
        idx = idxfunc(a)
        if idx in inter_atoms and is_anchor(a, inter_atoms):
            inter_label.append( (idx, get_anchor_smiles(new_mol, idx)) )

    for a in new_mol.GetAtoms():
        a.SetAtomMapNum( 1 if idxfunc(a) in inter_atoms else 0 )
    return new_mol, inter_label

def is_anchor(atom: Chem.Atom, inter_atoms: Set[int]) -> bool:
    """
    Check if an atom is an anchor based on its neighbors.

    Parameters:
    - atom (Chem.Atom): The atom to check.
    - inter_atoms (Set[int]): Set of intersecting atom indices.

    Returns:
    - bool: True if the atom is an anchor, False otherwise.
    """
    for a in atom.GetNeighbors():
        if idxfunc(a) not in inter_atoms:
            return True
    return False
            
def get_anchor_smiles(mol: Chem.Mol, anchor: int, idxfunc: callable = idxfunc) -> str:
    """
    Get the SMILES representation of a molecule with a specified anchor atom.

    Parameters:
    - mol (Chem.Mol): The molecule.
    - anchor (int): The index of the anchor atom.
    - idxfunc (callable): Function to adjust atom indices.

    Returns:
    - str: The SMILES representation of the molecule with the anchor atom.
    """
    copy_mol = Chem.Mol(mol)
    for a in copy_mol.GetAtoms():
        idx = idxfunc(a)
        if idx == anchor: a.SetAtomMapNum(1)
        else: a.SetAtomMapNum(0)

    return get_smiles(copy_mol)

def is_aromatic_ring(mol: Chem.Mol) -> bool:
    """
    Check if a molecule forms an aromatic ring.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.

    Returns:
    - bool: True if the molecule forms an aromatic ring, False otherwise.
    """
    if mol.GetNumAtoms() == mol.GetNumBonds(): 
        aroma_bonds = [b for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        return len(aroma_bonds) == mol.GetNumBonds()
    else:
        return False
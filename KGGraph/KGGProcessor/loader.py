from pathlib import Path
import sys

# Get the root directory
root_dir = Path(__file__).resolve().parents[2]
# Add the root directory to the system path
sys.path.append(str(root_dir))
from KGGraph.KGGChem.atom_utils import get_mol
import pandas as pd


def load_tox21_dataset(input_path):
    tox21_dataset = pd.read_csv(input_path, sep=",")
    smiles_list = tox21_dataset["smiles"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    tasks = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]
    labels = tox21_dataset[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values


def load_hiv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    labels = input_df["HIV_active"]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values


def load_muv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    tasks = [
        "MUV-466",
        "MUV-548",
        "MUV-600",
        "MUV-644",
        "MUV-652",
        "MUV-689",
        "MUV-692",
        "MUV-712",
        "MUV-713",
        "MUV-733",
        "MUV-737",
        "MUV-810",
        "MUV-832",
        "MUV-846",
        "MUV-852",
        "MUV-858",
        "MUV-859",
    ]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values


def load_bace_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array
    containing indices for each of the 3 folds, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["mol"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    labels = input_df["Class"]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df["Model"]
    folds = folds.replace("Train", 0)  # 0 -> train
    folds = folds.replace("Valid", 1)  # 1 -> valid
    folds = folds.replace("Test", 2)  # 2 -> test
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, mols_list, folds.values, labels.values


def load_bbbp_dataset(input_path):
    """
    Load the BBBP dataset from a CSV file.

    :param input_path: Path to the CSV file containing the dataset
    :return: Tuple containing a list of SMILES strings, a list of RDKit Mol objects, and a NumPy array containing the labels
    """
    # Load the dataset
    input_df = pd.read_csv(input_path, sep=",")

    # Filter out invalid molecules directly using Pandas
    input_df["mol"] = input_df["smiles"].apply(get_mol)
    input_df = input_df[input_df["mol"].notnull()]

    # Extract SMILES strings and molecule objects
    smiles_list = input_df["smiles"].tolist()
    mols_list = input_df["mol"].tolist()

    # Handle labels: convert 0 to -1, then ensure there are no NaN values
    labels = input_df["p_np"].replace(0, -1)
    assert not labels.isnull().any()

    # Assertions to check list lengths
    assert len(smiles_list) == len(mols_list) == len(labels)

    return smiles_list, mols_list, labels.values


def load_clintox_dataset(input_path):
    """
    Load the clintox dataset from a CSV file.

    :param input_path: Path to the CSV file containing the dataset
    :return: Tuple containing a list of SMILES strings, a list of RDKit Mol objects, and a NumPy array containing the labels
    """
    # Load the dataset
    input_df = pd.read_csv(input_path, sep=",")

    # Filter out invalid molecules directly using Pandas
    input_df["mol"] = input_df["smiles"].apply(get_mol)
    input_df = input_df[input_df["mol"].notnull()]

    # Extract SMILES strings and molecule objects
    smiles_list = input_df["smiles"].tolist()
    mols_list = input_df["mol"].tolist()

    # Handle labels: convert 0 to -1, then ensure there are no NaN values
    tasks = ["FDA_APPROVED", "CT_TOX"]
    labels = input_df[tasks].replace(0, -1)
    # there are no nans

    # Assertions to check list lengths
    assert len(smiles_list) == len(mols_list) == len(labels)

    return smiles_list, mols_list, labels.values


def load_sider_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    tasks = [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values


def load_toxcast_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # Load the dataset
    input_df = pd.read_csv(input_path, sep=",")
    tasks = list(input_df.columns)[1:]
    # Filter out invalid molecules directly using Pandas
    input_df["mol"] = input_df["smiles"].apply(get_mol)
    input_df = input_df[input_df["mol"].notnull()]

    # Extract SMILES strings and molecule objects
    smiles_list = input_df["smiles"].tolist()
    mols_list = input_df["mol"].tolist()

    # Handle labels: convert 0 to -1, then ensure there are no NaN values
    labels = input_df[tasks].replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)

    # Assertions to check list lengths
    assert len(smiles_list) == len(mols_list) == len(labels)

    return smiles_list, mols_list, labels.values


def load_esol_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    labels = input_df["logSolubility"]
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values


def load_freesolv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    labels = input_df["y"]
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values


def load_lipo_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    labels = input_df["lipo"]
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values


def load_qm7_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    labels = input_df["u0_atom"]
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values


def load_qm8_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    tasks = [
        "E1-CC2",
        "E2-CC2",
        "f1-CC2",
        "f2-CC2",
        "E1-PBE0",
        "E2-PBE0",
        "f1-PBE0",
        "f2-PBE0",
        "E1-CAM",
        "E2-CAM",
        "f1-CAM",
        "f2-CAM",
    ]
    labels = input_df[tasks]
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values


def load_qm9_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["smiles"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    tasks = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "cv",
        "u0",
        "u298",
        "h298",
        "g298",
    ]
    labels = input_df[tasks]
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values


def load_ecoli_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=",")
    smiles_list = input_df["Smiles"]
    mols_list = [get_mol(smile) for smile in smiles_list]
    tasks = [
        "MIC",
    ]
    labels = input_df[tasks]
    assert len(smiles_list) == len(mols_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, mols_list, labels.values

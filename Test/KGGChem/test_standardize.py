import unittest
import os
import sys
import pathlib
import warnings
from KGGraph.KGGChem.standardize import SmileStandardizer

root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)


warnings.filterwarnings("ignore")


class TestStandardize(unittest.TestCase):
    def setUp(self):
        self.smile = ["CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"]
        self.error_smile = ["CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2"]
        with open("Data/feature/standardize.txt", "w") as f:
            f.writelines(s + "\n" for s in self.smile)

    def tearDown(self):
        os.remove("Data/feature/standardize.txt")
        if os.path.exists("Data/feature/standardize_output.txt"):
            os.remove("Data/feature/standardize_output.txt")

    def test_standardize(self):
        input_file = "Data/feature/standardize.txt"
        output_file = "Data/feature/standardize_output.txt"
        SmileStandardizer.standardize(input_file, output_file)
        with open(output_file, "r") as f:
            stand_smiles = [line.strip("\r\n").split()[0] for line in f]
        self.assertEqual(stand_smiles[0], "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1")

    def test_safe_standardize_smiles(self):
        stand_smile = SmileStandardizer.safe_standardize_smiles(self.smile[0])
        self.assertEqual(stand_smile, "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1")
        stand_error_smile = SmileStandardizer.safe_standardize_smiles(
            self.error_smile[0]
        )
        self.assertEqual(stand_error_smile, None)


if __name__ == "__main__":
    unittest.main()

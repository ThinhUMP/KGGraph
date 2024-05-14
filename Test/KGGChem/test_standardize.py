import unittest
import os
import sys
import pathlib
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)
from KGGraph.KGGChem.standardize import SmileStandardizer
import warnings
warnings.filterwarnings("ignore")

class TestStandardize(unittest.TestCase):
    def setUp(self):
        self.smile = ["CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"]
        with open("Data/feature/standardize.txt", "w") as f:
            f.writelines(s + "\n" for s in self.smile)
    def tearDown(self):
        os.remove("Data/feature/standardize.txt")
        os.remove("Data/feature/standardize_output.txt")
    
    def test_standardize(self):
        input_file="Data/feature/standardize.txt"
        output_file="Data/feature/standardize_output.txt"
        SmileStandardizer.standardize(input_file, output_file)
        with open(output_file, "r") as f:
            stand_smiles = [line.strip("\r\n").split()[0] for line in f]
        self.assertEqual(stand_smiles[0], "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1")
        
if __name__ == "__main__":
    unittest.main()


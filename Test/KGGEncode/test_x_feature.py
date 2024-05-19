import unittest
import sys
import pathlib
from KGGraph.KGGEncode.x_feature import x_feature
from KGGraph.KGGChem.atom_utils import get_mol

root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)


class TestXFeature(unittest.TestCase):
    def setUp(self):
        self.smiles = ["c1ccccc1", "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"]
        self.mol = [get_mol(smile) for smile in self.smiles]

    def test_x_feature(self):
        # motif decompose with mask node and fix ratio
        x_node, x, num_part = x_feature(
            self.mol[0],
            decompose_type="motif",
            mask_node=True,
            mask_node_ratio=0.25,
            fix_ratio=True,
        )
        self.assertTrue(121.0 in x[:, 0])
        self.assertEqual(num_part, (6, 0, 1))
        # smotif decompose with not mask node
        x_node, x, num_part = x_feature(
            self.mol[1],
            decompose_type="smotif",
            mask_node=False,
            mask_node_ratio=0.25,
            fix_ratio=True,
        )
        self.assertEqual(num_part, (16, 4, 1))
        self.assertFalse(121.0 in x[:, 0])
        # jin decompose with mask node
        x_node, x, num_part = x_feature(
            self.mol[1],
            decompose_type="jin",
            mask_node=True,
            mask_node_ratio=0.25,
            fix_ratio=False,
        )
        self.assertEqual(num_part, (16, 10, 1))
        self.assertTrue(121.0 in x[:, 0])
        # brics decompose with mask node
        x_node, x, num_part = x_feature(
            self.mol[1],
            decompose_type="brics",
            mask_node=True,
            mask_node_ratio=0.25,
            fix_ratio=False,
        )
        self.assertEqual(num_part, (16, 7, 1))
        self.assertTrue(121.0 in x[:, 0])


if __name__ == "__main__":
    unittest.main()

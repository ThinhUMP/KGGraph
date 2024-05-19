import unittest
import sys
import pathlib
from KGGraph.KGGEncode.edge_feature import edge_feature
from KGGraph.KGGChem.atom_utils import get_mol
root_dir = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(root_dir)

class TestEdgeFeature(unittest.TestCase):
    def setUp(self):
        self.smiles = ['c1ccccc1', 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1']
        self.mol = [get_mol(smile) for smile in self.smiles]

    def test_edge_feature(self):
        # motif decompose with mask node and fix ratio
        edge_attr_node, edge_index_node, edge_index, edge_attr = edge_feature(self.mol[0], decompose_type="motif", mask_edge=True, mask_edge_ratio=0.25, fix_ratio=True)
        self.assertTrue(18 > edge_attr.size(0)) # 18 is the number of edges before masking
        self.assertTrue(18 > edge_index.size(1)) # 18 is the number of edges before masking
        self.assertTrue(0.5 in edge_attr_node[:,3]) # 0.5 is the number of pi for aromatic ring
        # smotif decompose with not mask node
        edge_attr_node, edge_index_node, edge_index, edge_attr = edge_feature(self.mol[1], decompose_type="smotif", mask_edge=False, mask_edge_ratio=0.25, fix_ratio=True)
        self.assertTrue(57 == edge_attr.size(0)) # 57 is the number of edges before masking
        self.assertTrue(57 == edge_index.size(1)) # 57 is the number of edges before masking
        self.assertTrue(0.5 in edge_attr_node[:,3]) # 0.5 is the number of pi for aromatic ring
        # jin decompose with mask node
        edge_attr_node, edge_index_node, edge_index, edge_attr = edge_feature(self.mol[1], decompose_type="jin", mask_edge=True, mask_edge_ratio=0.25, fix_ratio=False)
        self.assertTrue(70 > edge_attr.size(0)) # 70 is the number of edges before masking
        self.assertTrue(70 > edge_index.size(1)) # 70 is the number of edges before masking
        self.assertTrue(0.5 in edge_attr_node[:,3]) # 0.5 is the number of pi for aromatic ring
        # brics decompose with mask node
        edge_attr_node, edge_index_node, edge_index, edge_attr = edge_feature(self.mol[1], decompose_type="brics", mask_edge=True, mask_edge_ratio=0.25, fix_ratio=False)
        self.assertTrue(57 > edge_attr.size(0)) # 57 is the number of edges before masking
        self.assertTrue(57 > edge_index.size(1)) # 57 is the number of edges before masking
        self.assertTrue(0.5 in edge_attr_node[:,3]) # 0.5 is the number of pi for aromatic ring
        
if __name__ == '__main__':
    unittest.main()
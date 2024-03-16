from KGGraph.Dataset.molecule_dataset import MoleculeDataset

def main():
    dataset = MoleculeDataset('./data/tox21/', dataset='tox21')
    print(dataset)
    print(dataset[0])

if __name__ == '__main__':
    main()
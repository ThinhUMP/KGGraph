def num_vocab_edge_attr_embedding(dataset) -> int:
    """
    Compute the size of vocabulary for edge attribute embeddings.

    This function is useful for determining the size of the vocabulary for edge attribute embeddings.
    The '+1' is added since torch.nn.Embedding in range 0->unique edge attributes + 1

    Parameters:
    dataset (torch_geometric.data.Dataset): The dataset to compute the vocabulary size for.

    Returns:
    int: The size of the vocabulary for edge attribute embeddings.
    """

    # Get the unique edge attributes in the dataset
    unique_edge_attrs = dataset.edge_attr.unique()

    # Get the maximum unique edge attribute
    max_edge_attr = max(unique_edge_attrs)

    # Convert the maximum edge attribute to size of the vocabulary for embedding
    vocab_size = max_edge_attr.item() + 1

    return vocab_size
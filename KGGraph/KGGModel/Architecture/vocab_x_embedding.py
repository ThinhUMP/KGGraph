def num_vocab_x_embedding(dataset) -> int:
    """
    Compute the size of the vocabulary for node feature (x) embeddings.

    This function is useful for determining the size of the vocabulary for node feature embeddings.
    The '+1' is added since torch.nn.Embedding in range 0->unique node features (x) + 1

    Parameters:
    dataset (torch_geometric.data.Dataset): The dataset to compute the vocabulary size for.

    Returns:
    int: The size of the vocabulary for node feature embeddings.
    """

    # Get the unique node features in the dataset
    unique_node_features = dataset.x.unique()

    # Get the maximum unique node feature
    max_node_feature = max(unique_node_features)

    # Convert the maximum node feature to size of the vocabulary for embedding
    vocab_size = max_node_feature.item() + 1

    return vocab_size
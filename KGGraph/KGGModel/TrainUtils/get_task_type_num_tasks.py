def get_task_type(args):
    """
    Determines the type of task (classification or regression) based on the dataset.

    Parameters:
    args: An argument parser object or a similar structure where args.dataset is the name of the dataset.

    Returns:
    str: The type of task associated with the dataset ('classification' or 'regression').
    """
    # List of datasets associated with classification tasks
    classification_datasets = ['tox21', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox']
    
    if args.dataset in classification_datasets:
        return 'classification'
    else:
        return 'regression'

def get_num_task(args):
    """
    Retrieves the number of tasks associated with a specific dataset.

    Parameters:
    args: An argument parser object or a similar structure where args.dataset is the name of the dataset.

    Returns:
    int: The number of tasks associated with the dataset.
    """
    # Define the number of tasks for each dataset
    num_tasks_dict = {
        "tox21": 12,
        "bace": 1,
        "bbbp": 1,
        "toxcast": 617,
        "sider": 27,
        "clintox": 2,
        "esol": 1,
        "freesolv": 1,
        "lipophilicity": 1,
        "qm7": 1,
        "qm8": 12,
        "qm9": 12
    }

    # Get the number of tasks based on the dataset
    num_tasks = num_tasks_dict.get(args.dataset)

    if num_tasks is None:
        raise ValueError("Invalid dataset name.")

    return num_tasks

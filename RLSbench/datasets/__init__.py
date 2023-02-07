from RLSbench.datasets.data_utils import *
from RLSbench.datasets.get_dataset import *

benchmark_datasets = [
    'camelyon',
    'iwildcam',
    'fmow',
    'cifar10',
    'cifar100',
    'domainnet',
    'entity13', 
    'entity30',
    'living17',
    'nonliving26',
    'office31',
    'officehome',
    'visda', 
    'civilcomments',
    'amazon',
    'retiring_adult',
    'mimic_readmission',
]

supported_datasets = benchmark_datasets 

dataset_map = { 
    "cifar10" : get_cifar10, 
    "cifar100" : get_cifar100,
    'office31' : get_office31,
    'officehome' : get_officehome,
    'visda' : get_visda,
    'domainnet' : get_domainnet,
    'entity13' : get_entity13,
    'entity30' : get_entity30,
    'living17': get_living17,
    'nonliving26': get_nonliving26,
    'fmow': get_fmow,
    'iwildcam': get_iwildcams,
    'rxrx1': get_rxrx1,
    'camelyon': get_camelyon, 
    'civilcomments': get_civilcomments,
    'amazon': get_amazon, 
    'retiring_adult': get_retiring_adult,
    'mimic_readmission': get_mimic_readmission,
}


def get_dataset(dataset, source=True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.8, seed=42):
    """Get dataset.
    
    Args:   
        dataset (str): Name of the dataset.
        source (bool): Whether to return the source dataset.
        target (bool): Whether to return the target dataset.
        root_dir (str): Path to the root directory of the dataset.
        target_split (int): Num of the target split.
        transforms (dict): Dictionary of transformations.
        num_classes (int): Number of classes.
        split_fraction (float): Fraction of the dataset to use for training.
        seed (int): Random seed.
        
    Returns:
        dataset (torch.utils.data.Dataset): Dataset.
    """

    return dataset_map[dataset](source, target, root_dir, target_split, transforms, num_classes, split_fraction, seed)



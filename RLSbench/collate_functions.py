import copy
from typing import List
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pickle
from typing import List
import torch
import os
from torch.nn.utils.rnn import pad_sequence


def initialize_collate_function(collate_function_name):
    """
    Initializes collate_function that takes in a batch of samples
    returned by data loader and combines them into a tensor.
    This is standard for images and datasets where each element
    has the same size. But, for datasets like mimic and arxiv,
    where each element in the batch varies in size, the collate
    function must handle the padding etc..
    """

    if collate_function_name is None or collate_function_name.lower() == "none":
        return None
    elif (
        collate_function_name.lower() == "mimic_readmission"
        or collate_function_name.lower() == "mimic_mortality"
    ):
        return collate_fn_mimic
    else:
        raise ValueError(f"{collate_function_name} not recognized")


def collate_fn_mimic(batch):
    """
    batch is a list, where each element is also a list of size
    at least two. The first element of the inner list is
    [code, type] and the second element is the label. The rest
    of the dimensions may contain auxiliary information.
    """
    codes = [item[0][0] for item in batch]
    types = [item[0][1] for item in batch]
    target_and_aux = [item[1:] for item in batch]
    target_and_aux = list(zip(*target_and_aux))
    target_and_aux = [torch.tensor(item) for item in target_and_aux]
    return [(codes, types), *target_and_aux]

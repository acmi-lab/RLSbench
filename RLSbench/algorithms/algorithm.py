import torch.nn as nn
from RLSbench.utils import detach_and_clone, move_to


class Algorithm(nn.Module):
    def __init__(
        self,
        device,
    ):
        super().__init__()
        self.device = device
        self.out_device = "cpu"

    def update(self, batch):
        """
        Process the batch, update the log, and update the model
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - y_true (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
        """
        raise NotImplementedError

    def evaluate(self, batch):
        """
        Process the batch and update the log, without updating the model
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - y_true (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
        """
        raise NotImplementedError

    def train(self, mode=True):
        """
        Switch to train mode
        """
        self.is_training = mode
        super().train(mode)

    def step_schedulers(self, is_epoch):
        """
        Update all relevant schedulers
        Args:
            - is_epoch (bool): epoch-wise update if set to True, batch-wise update otherwise
            - metrics (dict): a dictionary of metrics that can be used for scheduler updates
            - log_access (bool): whether metrics from self.get_log() can be used to update schedulers
        """
        raise NotImplementedError

import torch
from RLSbench.algorithms.algorithm import Algorithm
from RLSbench.optimizer import initialize_optimizer
from RLSbench.scheduler import initialize_scheduler, step_scheduler
from RLSbench.utils import move_to
from torch.nn import DataParallel


class SingleModelAlgorithm(Algorithm):
    """
    An abstract class for algorithm that has one underlying model.
    """
    def __init__(self, config, model, loss, n_train_steps):
        
        super().__init__(config.device)
        self.loss = loss

        # initialize models, optimizers, and schedulers
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = initialize_optimizer(config, model)

        self.schedulers = [initialize_scheduler(config, self.optimizer, n_train_steps)]

        if config.use_data_parallel:
            model = DataParallel(model)

        model.to(config.device)

        self.batch_idx = 0
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.model = model

    def get_model_output(self, x):
        outputs = self.model(x)
        return outputs

    def process_batch(self, batch, unlabeled_batch = None):
        """
        A helper function for update() and evaluate() that processes the batch
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - y_pred (Tensor): model output for batch 
        """
        x, y_true = batch[:2]
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)

        outputs = self.get_model_output(x)

        results = {
            'y_true': y_true,
            'y_pred': outputs,
        }
        return results

    def objective(self, results):
        raise NotImplementedError

    def evaluate(self, batch):
        """
        Process the batch and update the log, without updating the model
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - y_true (Tensor)
                - outputs (Tensor)
                - y_pred (Tensor)
        """
        assert not self.is_training
        results = self.process_batch(batch)
        return results

    def update(self, batch, unlabeled_batch=None, target_marginal = None, source_marginal = None, target_average = None, is_epoch_end=False):
        """
        Process the batch, update the log, and update the model
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
            - is_epoch_end: whether this batch is the last batch of the epoch. if so, force optimizer to step,
                regardless of whether this batch idx divides self.gradient_accumulation_steps evenly
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - outputs (Tensor)
                - y_pred (Tensor)
                - objective (float)
        """
        assert self.is_training

        # self.optimizer.zero_grad()

        # process this batch
        results = self.process_batch(batch, unlabeled_batch, target_marginal, source_marginal, target_average)

        # update running statistics and update model if we've reached end of effective batch
        # iterate batch index
        if is_epoch_end:
            self.batch_idx = 0

        else:
            self.batch_idx += 1

        self._update(results)

        return results

    def _update(self, results):
        """
        Computes the objective and updates the model.
        Also updates the results dictionary yielded by process_batch().
        Should be overridden to change algorithm update beyond modifying the objective.
        """
        # compute objective
        objective = self.objective(results) / self.gradient_accumulation_steps # normalize by gradient accumulation steps
        results['objective'] = objective.item()
        objective.backward()

        # import pdb; pdb.set_trace()
        if (self.batch_idx) % self.gradient_accumulation_steps == 0 or self.batch_idx == 0:

            self.optimizer.step()
            self.model.zero_grad()
            # self.optimizer.step()
            self.step_schedulers(is_epoch=False)

            
        if self.batch_idx == 0: 
            self.step_schedulers(is_epoch=True)
            

    def step_schedulers(self, is_epoch):
        """
        Updates the scheduler after an epoch.
        """
        for scheduler in self.schedulers:
            if scheduler is None:
                continue
            if scheduler.step_every_batch:
                step_scheduler(scheduler)
            elif is_epoch:
                step_scheduler(scheduler)

import logging

import torch
from RLSbench.algorithms.algorithm import Algorithm
from RLSbench.models.initializer import initialize_model
from RLSbench.models.model_utils import (collect_params,
                                                  configure_model)
from RLSbench.optimizer import initialize_optimizer_with_model_params
from RLSbench.utils import load, move_to
from torch.optim import SGD, Adam

logger = logging.getLogger("label_shift")

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class TENT(Algorithm):
    def __init__(self, config):

        logger.info("Initializing model...")

        model = initialize_model(
            model_name = config.model, 
            dataset_name = config.dataset,
            num_classes = config.num_classes,
            featurize = False,
            pretrained=False,
        )
        
        model.to(config.device)


        # initialize module
        super().__init__(
            device=config.device,
        )
        self.model = model
        

        self.source_balanced = config.source_balanced
        self.num_classes = config.num_classes
        self.config = config

    def get_model_output(self, x):
        outputs = self.model(x)
        return outputs

    def process_batch(self, batch):
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


    def adapt(self, source_loader, target_loader, target_marginal=None, source_marginal=None, target_average=None, pretrained_path = None):
        """
        Load the model and adapt it to the new data 
        Args:
            - unlabeled_batch (tuple of Tensors): a batch of data yielded by unlabeled data loader
            - target_marginal (Tensor): the marginal distribution of the target
            - source_marginal (Tensor): the marginal distribution of the source
            - target_average (Tensor): the average of the target

        Output:
        """

        
        if pretrained_path is not None:
            logger.info(f"Loading pretrained model from {pretrained_path}")
            load(self.model, pretrained_path, device=self.device)

        # TODO: Check what if we adapt to the BN params here
        # logger.info("Adapting BN params...")

        # self.train(True)
        
        # with torch.no_grad():    
        #     for batch in target_loader: 
        #         self.model(batch[0].to(self.device))

        self.model = configure_model(self.model)
        params, param_names = collect_params(self.model)
        
        self.optimizer = SGD(params, lr = 1e-4, momentum=0.9)
        
        logger.info("Adapting model to TENT ...")

        # for epoch in range(5):
        for batch in target_loader: 
            
            self.optimizer.zero_grad()
            outputs = self.model(batch[0].to(self.device))

            loss = softmax_entropy(outputs).mean(0)

            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()

    def reset(self): 
        pass
       


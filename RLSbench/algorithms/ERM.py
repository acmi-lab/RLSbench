import logging

import torch
import torch.nn as nn
from RLSbench.algorithms.single_model_algorithm import \
    SingleModelAlgorithm
from RLSbench.losses import initialize_loss
from RLSbench.models.initializer import initialize_model
from RLSbench.models.model_utils import linear_probe
from RLSbench.utils import move_to

logger = logging.getLogger("label_shift")


class ERM(SingleModelAlgorithm):
    def __init__(self, config, dataloader, loss_function, n_train_steps):

        logger.info("Initializing model...")

        if config.algorithm.startswith("IW"):
            self.use_target_marginal = True
        else:
            self.use_target_marginal = False

        if config.source_balanced or self.use_target_marginal:
            loss = initialize_loss(loss_function, reduction='none')
        else:
            loss = initialize_loss(loss_function)

        
        model = initialize_model(
            model_name = config.model, 
            dataset_name = config.dataset,
            num_classes = config.num_classes,
            featurize = True, 
            pretrained=config.pretrained,
            pretrained_path=config.pretrained_path,
            data_dir=config.root_dir,
        )
        
        if config.pretrained and "clip" in config.model: 
            model = linear_probe(model, dataloader, device= config.device, progress_bar=config.progress_bar)

        model = nn.Sequential(*model)

        # initialize module
        super().__init__(
            config=config,
            model=model,
            loss=loss,
            n_train_steps=n_train_steps,
        )
        
        self.use_unlabeled_y = config.use_unlabeled_y # Expect x,y,m from unlabeled loaders and train on the unlabeled y

        self.source_balanced = config.source_balanced
        self.num_classes = config.num_classes
        
    def process_batch(self, batch, unlabeled_batch=None, target_marginal=None, source_marginal= None, target_average=None):
        """
        Overrides single_model_algorithm.process_batch().
        ERM defines its own process_batch to handle if self.use_unlabeled_y is true.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - y_pred (Tensor): model output for batch 
                - unlabeled_y_pred (Tensor): predictions for unlabeled batch for fully-supervised ERM experiments
                - unlabeled_y_true (Tensor): true labels for unlabeled batch for fully-supervised ERM experiments
        """
        x, y_true = batch[:2]
        # import pdb; pdb.set_trace()
        # print(x)
        # print(y_true)
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)

        outputs = self.get_model_output(x)

        results = {
            'y_true': y_true,
            'y_pred': outputs        
        }
        if unlabeled_batch is not None:
            if self.use_unlabeled_y: # expect loaders to return x,y,m
                x, y = unlabeled_batch[:2]
                y = move_to(y, self.device)
                x = move_to(x, self.device)
                results['unlabeled_y_pred'] = self.get_model_output(x)
                results['unlabeled_y_true'] = y

        if self.use_target_marginal and target_marginal is not None:
            results['im_weights'] = torch.divide(torch.tensor(target_marginal).to(self.device),\
                    torch.tensor(source_marginal).to(self.device))
    
        if self.source_balanced and source_marginal is not None:
            results['source_marginal'] = torch.tensor(source_marginal).to(self.device)

        return results

    def objective(self, results):
        labeled_loss = self.loss(results['y_pred'], results['y_true'])

        if self.use_target_marginal: 
            labeled_loss = torch.mean(labeled_loss*results["im_weights"][results["y_true"]])

        elif self.source_balanced: 
            labeled_loss = torch.mean(labeled_loss/results["source_marginal"][results["y_true"]]/ self.num_classes)
        
        if self.use_unlabeled_y and 'unlabeled_y_true' in results:
            unlabeled_loss = self.loss(
                results['unlabeled_y_pred'], 
                results['unlabeled_y_true'], 
            )

            if self.use_target_marginal or self.source_balanced: 
                unlabeled_loss = torch.mean(unlabeled_loss)

            return labeled_loss + unlabeled_loss

        else:
            return labeled_loss

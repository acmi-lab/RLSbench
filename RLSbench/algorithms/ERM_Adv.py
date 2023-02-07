import logging

import torch
import torch.nn as nn
from RLSbench.algorithms.single_model_algorithm import \
    SingleModelAlgorithm
from RLSbench.losses import initialize_loss
from RLSbench.models.initializer import initialize_model
from RLSbench.models.model_utils import linear_probe
from RLSbench.utils import move_to
from robustness.attacker import AttackerModel

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

logger = logging.getLogger("label_shift")


class ERM_Adv(SingleModelAlgorithm):
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
        )
        model = nn.Sequential(*model)
        
        model = AttackerModel(model, torch.tensor(_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN), torch.tensor(_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD))

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
        self.n_epochs = config.n_epochs
        self.normal_train_epoch = int(self.n_epochs*0.9)

        self.curr_epoch = 0

        self.attack_kwargs = {
            'constraint': '2', 
            'eps': 0.5, 
            'iterations': 3,
            'step_size': 1.0/3, 
            'return_image': False, 
        }
        
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
        if self.is_training and self.batch_idx == 0:
            self.curr_epoch += 1

        x, y_true = batch[:2]
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)

        if self.is_training and self.curr_epoch < self.normal_train_epoch:
            outputs = self.model(x, y_true, make_adv = True, **self.attack_kwargs)
        else: 
            outputs = self.model(x)

        # import pdb; pdb.set_trace()
        results = {
            'y_true': y_true,
            'y_pred': outputs        
        }

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

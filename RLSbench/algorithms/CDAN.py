import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from RLSbench.scheduler import CoeffSchedule
from RLSbench.algorithms.single_model_algorithm import SingleModelAlgorithm
from RLSbench.losses import initialize_loss
from RLSbench.models.domain_adversarial_network import (
    ConditionalDomainAdversarialNetwork,
)
from RLSbench.models.initializer import initialize_model
from RLSbench.models.model_utils import linear_probe
from RLSbench.optimizer import initialize_optimizer_with_model_params
from RLSbench.utils import concat_input, move_to
from RLSbench.label_shift_utils import im_weights_update
import torch.nn.functional as F
from RLSbench.collate_functions import collate_fn_mimic

logger = logging.getLogger("label_shift")


class CDAN(SingleModelAlgorithm):
    """
    Conditional Adversarial Domain Adaptation.

    Original paper:
        @inproceedings{long2018conditional,
            title={Conditional adversarial domain adaptation},
            author={Long, Mingsheng and Cao, Zhangjie and Wang, Jianmin and Jordan, Michael I},
            booktitle={Advances in neural information processing systems},
            year={2018}
        }
    """

    def __init__(
        self, config, dataloader, loss_function, n_train_steps, n_domains=2, **kwargs
    ):
        logger.info("Initializing CDAN models")

        # import pdb; pdb.set_trace()

        if config.algorithm.startswith("IW"):
            self.use_target_marginal = True
        else:
            self.use_target_marginal = False

        if config.source_balanced or self.use_target_marginal:
            loss = initialize_loss(loss_function, reduction="none")
        else:
            loss = initialize_loss(loss_function)

        # Initialize model
        featurizer, classifier = initialize_model(
            model_name=config.model,
            dataset_name=config.dataset,
            num_classes=config.num_classes,
            featurize=True,
            pretrained=config.pretrained,
            pretrained_path=config.pretrained_path,
            data_dir=config.root_dir,
        )

        self.im_weights = np.ones((config.num_classes, 1))
        self.cov = np.zeros((config.num_classes, config.num_classes))
        self.source_marginal = np.zeros((config.num_classes))
        self.psuedo_marginal = np.zeros((config.num_classes))
        self.source_num_samples = 0
        self.target_num_samples = 0

        # if config.pretrained:
        #     featurizer, classifier = linear_probe( (featurizer, classifier), dataloader, device= config.device, progress_bar=config.progress_bar)

        model = ConditionalDomainAdversarialNetwork(
            featurizer, classifier, n_domains, config.num_classes
        )

        featurizer = model.featurizer
        classifier = nn.Sequential(model.bottleneck, model.classifier)

        # if config.pretrained :
        #     linear_probe( (featurizer, classifier), dataloader, device= config.device, progress_bar=config.progress_bar)

        parameters_to_optimize: List[Dict] = model.get_parameters_with_lr(
            featurizer_lr=kwargs["featurizer_lr"],
            classifier_lr=kwargs["classifier_lr"],
            discriminator_lr=kwargs["discriminator_lr"],
        )

        self.optimizer = initialize_optimizer_with_model_params(
            config, parameters_to_optimize
        )

        # Initialize module
        super().__init__(
            config=config,
            model=model,
            loss=loss,
            n_train_steps=n_train_steps,
        )

        self.coeff_schedule = CoeffSchedule(max_iter=n_train_steps)
        self.schedulers.append(self.coeff_schedule)

        self.domain_loss = initialize_loss("cross_entropy", reduction="none")

        # Algorithm hyperparameters
        self.penalty_weight = kwargs["penalty_weight"]
        self.source_balanced = config.source_balanced
        self.num_classes = config.num_classes
        self.dataset = config.dataset

    def process_batch(
        self,
        batch,
        unlabeled_batch=None,
        target_marginal=None,
        source_marginal=None,
        target_average=None,
    ):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch
                - domains_true (Tensor): true domains for batch and unlabeled batch
                - domains_pred (Tensor): predicted domains for batch and unlabeled batch
                - unlabeled_features (Tensor): featurizer outputs for unlabeled_batch
        """
        # Forward pass
        (
            x,
            y_true,
        ) = batch[:2]

        if unlabeled_batch is not None:
            unlabeled_x = unlabeled_batch[0]

            # Concatenate examples and true domains
            if "mimic" in self.dataset:
                # x_cat = collate_fn_mimic([x, unlabeled_x])
                x_cat = [x[0] + unlabeled_x[0], x[1] + unlabeled_x[1]]
                domains_true = torch.cat(
                    [
                        torch.zeros(len(x[0]), dtype=torch.long),
                        torch.ones(len(unlabeled_x[0]), dtype=torch.long),
                    ]
                )

            else:
                x_cat = concat_input(x, unlabeled_x)
                domains_true = torch.cat(
                    [
                        torch.zeros(len(x), dtype=torch.long),
                        torch.ones(len(unlabeled_x), dtype=torch.long),
                    ]
                )

            x_cat = move_to(x_cat, self.device)
            y_true = y_true.to(self.device)
            domains_true = domains_true.to(self.device)
            y_pred, domains_pred = self.model(
                x_cat, self.coeff_schedule.value, domain_classifier=True
            )

            y_source_pred = y_pred[: len(y_true)]
            y_target_pred = y_pred[len(y_true) :]

            results = {
                "y_true": y_true,
                "y_pred": y_source_pred,
                "target_y_pred": y_target_pred,
                "domains_true": domains_true,
                "domains_pred": domains_pred,
            }

            # if self.use_target_marginal and target_marginal is not None:
            #     results["im_weights"] =  torch.divide(torch.tensor(target_marginal).to(self.device),\
            #             torch.tensor(source_marginal).to(self.device))

            if source_marginal is not None:
                self.source_marginal = source_marginal
            #    results["source_marginal"] =  torch.tensor(source_marginal).to(self.device)

            return results
        else:
            x = move_to(x, self.device)
            y_true = y_true.to(self.device)

            y_pred = self.model(x)

            return {
                "y_true": y_true,
                "y_pred": y_pred,
            }

    def objective(self, results):
        if self.use_target_marginal:
            self.source_num_samples += len(results["y_pred"])
            self.target_num_samples += len(results["target_y_pred"])

            target_preds = F.softmax(results["target_y_pred"], dim=1)
            self.psuedo_marginal += (
                torch.sum(target_preds, dim=0).detach().cpu().numpy()
            )

            source_preds = F.softmax(results["y_pred"], dim=1)
            source_onehot = F.one_hot(
                results["y_true"], num_classes=self.num_classes
            ).float()

            self.cov += (
                torch.mm(source_preds.transpose(1, 0), source_onehot)
                .detach()
                .cpu()
                .numpy()
            )

            if self.batch_idx == 0:
                self.cov /= self.source_num_samples
                self.psuedo_marginal /= self.target_num_samples

                self.im_weights = im_weights_update(
                    self.source_marginal,
                    self.psuedo_marginal,
                    self.cov,
                    self.im_weights,
                )

                # import pdb; pdb.set_trace()

                self.cov = np.zeros((self.num_classes, self.num_classes))
                self.source_marginal = np.zeros((self.num_classes))
                self.psuedo_marginal = np.zeros((self.num_classes))

                self.source_num_samples = 0
                self.target_num_samples = 0

        classification_loss = self.loss(results["y_pred"], results["y_true"])
        # import pdb; pdb.set_trace()

        # if self.source_balanced:
        #     classification_loss = torch.mean(classification_loss/results["source_marginal"][results["y_true"]]/ self.num_classes)

        im_weights = torch.tensor(self.im_weights).to(self.device)

        if self.use_target_marginal:
            classification_loss = torch.mean(
                classification_loss * im_weights[results["y_true"]]
            )

        if self.is_training:
            domain_classification_loss = self.domain_loss(
                results["domains_pred"],
                results["domains_true"],
            )
            if self.use_target_marginal:
                source_size = len(results["y_true"])
                domain_classification_loss_source = torch.mean(
                    domain_classification_loss[:source_size]
                    * im_weights[results["y_true"]]
                )
                domain_classification_loss_target = torch.mean(
                    domain_classification_loss[source_size:]
                )
                domain_classification_loss = (
                    domain_classification_loss_source
                    + domain_classification_loss_target
                ) / 2.0
            else:
                domain_classification_loss = torch.mean(domain_classification_loss)
        else:
            domain_classification_loss = 0.0

        return classification_loss + domain_classification_loss * self.penalty_weight

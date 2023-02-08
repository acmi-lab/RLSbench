import logging
from typing import Dict, List

import torch
import torch.nn.functional as F
from RLSbench.algorithms.single_model_algorithm import SingleModelAlgorithm
from RLSbench.losses import initialize_loss
from RLSbench.models.domain_adversarial_network import COALNetwork
from RLSbench.models.initializer import initialize_model
from RLSbench.models.model_utils import linear_probe
from RLSbench.optimizer import initialize_optimizer_with_model_params
from RLSbench.scheduler import LinearScheduleWithWarmupAndThreshold
from RLSbench.utils import (
    concat_input,
    detach_and_clone,
    move_to,
    pseudolabel_multiclass_logits,
)

logger = logging.getLogger("label_shift")


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class COAL(SingleModelAlgorithm):
    """
    COAL.

    Original paper:
       @inproceedings{tan2020class,
        title={Class-imbalanced domain adaptation: An empirical odyssey},
        author={Tan, Shuhan and Peng, Xingchao and Saenko, Kate},
        booktitle={European Conference on Computer Vision},
        pages={585--602},
        year={2020},
        organization={Springer}
        }
    """

    def __init__(self, config, dataloader, loss_function, n_train_steps, **kwargs):
        logger.info("Initializing PseudoLabel models")

        model = initialize_model(
            model_name=config.model,
            dataset_name=config.dataset,
            num_classes=config.num_classes,
            featurize=True,
            pretrained=config.pretrained,
            pretrained_path=config.pretrained_path,
        )

        if config.algorithm.startswith("IW"):
            self.use_target_marginal = True
        else:
            self.use_target_marginal = False

        if config.source_balanced or self.use_target_marginal:
            loss = initialize_loss(loss_function, reduction="none")
        else:
            loss = initialize_loss(loss_function)

        # if config.pretrained:
        #     featurizer, classifier = linear_probe(model, dataloader, device= config.device, progress_bar=config.progress_bar)

        model = COALNetwork(model[0], num_classes=config.num_classes)

        featurizer = model.featurizer
        classifier = model.classifier

        if config.pretrained:
            linear_probe(
                (featurizer, classifier),
                dataloader,
                device=config.device,
                progress_bar=config.progress_bar,
            )

        parameters_to_optimize: List[Dict] = model.get_parameters_with_lr(
            featurizer_lr=kwargs["featurizer_lr"],
            classifier_lr=kwargs["classifier_lr"],
            # discriminator_lr=kwargs["discriminator_lr"],
        )

        self.optimizer = initialize_optimizer_with_model_params(
            config, parameters_to_optimize
        )

        # initialize module
        super().__init__(
            config=config,
            model=model,
            loss=loss,
            n_train_steps=n_train_steps,
        )

        # algorithm hyperparameters
        self.confidence_threshold = kwargs["self_training_threshold"]
        self.alpha = kwargs["alpha"]
        self.process_pseudolabels_function = pseudolabel_multiclass_logits

        self.target_align = False

        self.source_balanced = config.source_balanced
        self.num_classes = config.num_classes

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
                - y_pred (Tensor): model output for batch
                - unlabeled_y_pseudo (Tensor): pseudolabels on the unlabeled batch, already thresholded
                - unlabeled_y_pred (Tensor): model output on the unlabeled batch, already thresholded
        """
        # Labeled examples
        x, y_true = batch[:2]
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)

        n_lab = y_true.shape[0]

        # package the results
        results = {
            "y_true": y_true,
        }

        # TODO: Add target alignment if it is useful
        # alignment_dist = torch.divide(torch.tensor(target_marginal).to(self.device), torch.tensor(target_average).to(self.device))

        if unlabeled_batch is not None:
            x_unlab = unlabeled_batch[0]
            x_unlab = move_to(x_unlab, self.device)

            x_cat = concat_input(x, x_unlab)
            outputs = self.model(x_cat)
            unlabeled_output = outputs[n_lab:]

            if self.target_align:
                (
                    unlabeled_y_pred,
                    unlabeled_y_pseudo,
                    pseudolabels_kept_frac,
                    _,
                ) = self.process_pseudolabels_function(
                    unlabeled_output, self.confidence_threshold, alignment_dist
                )
            else:
                (
                    unlabeled_y_pred,
                    unlabeled_y_pseudo,
                    pseudolabels_kept_frac,
                    _,
                ) = self.process_pseudolabels_function(
                    unlabeled_output, self.confidence_threshold
                )

            results["y_pred"] = outputs[:n_lab]
            results["unlabeled_y_pred"] = unlabeled_y_pred
            results["unlabeled_y_pseudo"] = detach_and_clone(unlabeled_y_pseudo)

            if self.source_balanced and source_marginal is not None:
                results["source_marginal"] = torch.tensor(source_marginal).to(
                    self.device
                )

            if self.use_target_marginal and target_marginal is not None:
                results["im_weights"] = torch.divide(
                    torch.tensor(target_marginal).to(self.device),
                    torch.tensor(source_marginal).to(self.device),
                )
                results["target_marginal"] = torch.tensor(target_marginal).to(
                    self.device
                )

            x_unlab_copy = torch.clone(x_unlab)
            unlabeled_output_ent = self.model(x_unlab_copy, reverse=True)
            results["unlabeled_y_pred_ent"] = unlabeled_output_ent

            ## New edits below

            # outputs = self.model(x)
            # results['y_pred'] = outputs

            # outputs_unlab = self.model(x_unlab, reverse=True)
            # results['unlabeled_y_pred'] = outputs_unlab

        else:
            results["y_pred"] = self.get_model_output(x)
            pseudolabels_kept_frac = 0

        results["pseudolabels_kept_frac"] = pseudolabels_kept_frac

        return results

    def objective(self, results):
        # Labeled loss
        classification_loss = self.loss(results["y_pred"], results["y_true"])

        if self.use_target_marginal:
            classification_loss = torch.mean(
                classification_loss * results["im_weights"][results["y_true"]]
            )

        elif self.source_balanced:
            classification_loss = torch.mean(
                classification_loss
                / results["source_marginal"][results["y_true"]]
                / self.num_classes
            )

        # Pseudolabeled loss
        if "unlabeled_y_pred" in results:
            loss_output = self.loss(
                results["unlabeled_y_pred"],
                results["unlabeled_y_pseudo"],
            )

            if self.source_balanced:
                target_marginal = results["target_marginal"]
                target_marginal[target_marginal == 0] = 1.0

                loss_output = torch.mean(
                    loss_output
                    / target_marginal[results["unlabeled_y_pseudo"]]
                    / self.num_classes
                )

            elif self.use_target_marginal:
                loss_output = torch.mean(loss_output)

            consistency_loss = loss_output * results["pseudolabels_kept_frac"]

            y_pred_ent = results["unlabeled_y_pred_ent"]
            ent_loss = -self.alpha * torch.mean(softmax_entropy(y_pred_ent), dim=0)

            # import pdb; pdb.set_trace()
        else:
            consistency_loss = 0
            ent_loss = 0

        return classification_loss + ent_loss + consistency_loss

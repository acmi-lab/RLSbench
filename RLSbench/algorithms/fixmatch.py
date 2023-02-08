import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from RLSbench.algorithms.single_model_algorithm import SingleModelAlgorithm
from RLSbench.losses import initialize_loss
from RLSbench.models.initializer import initialize_model
from RLSbench.models.model_utils import linear_probe
from RLSbench.utils import detach_and_clone, pseudolabel_multiclass_logits

logger = logging.getLogger("label_shift")


class FixMatch(SingleModelAlgorithm):
    """
    FixMatch.
    This algorithm was originally proposed as a semi-supervised learning algorithm.

    Loss is of the form
        \ell_s + \lambda * \ell_u
    where
        \ell_s = cross-entropy with true labels using weakly augmented labeled examples
        \ell_u = cross-entropy with pseudolabel generated using weak augmentation and prediction
            using strong augmentation

    Original paper:
        @article{sohn2020fixmatch,
            title={Fixmatch: Simplifying semi-supervised learning with consistency and confidence},
            author={Sohn, Kihyuk and Berthelot, David and Li, Chun-Liang and Zhang, Zizhao and Carlini, Nicholas and Cubuk, Ekin D and Kurakin, Alex and Zhang, Han and Raffel, Colin},
            journal={arXiv preprint arXiv:2001.07685},
            year={2020}
            }
    """

    def __init__(self, config, dataloader, loss_function, n_train_steps, **kwargs):
        logger.info("Intializing FixMatch algorithm model")

        model = initialize_model(
            model_name=config.model,
            dataset_name=config.dataset,
            num_classes=config.num_classes,
            featurize=True,
            pretrained=config.pretrained,
            pretrained_path=config.pretrained_path,
        )

        # if config.algorithm.startswith("IW"):
        #     # self.use_target_marginal = True
        #     self.use_target_marginal = False
        # else:
        self.use_target_marginal = False

        # if config.source_balanced or self.use_target_marginal:
        #     loss = initialize_loss(loss_function, reduction='none')
        # else:
        loss = initialize_loss(loss_function)

        # if config.pretrained:
        #     model = linear_probe(model, dataloader, device= config.device, progress_bar=config.progress_bar)

        model = nn.Sequential(*model)

        # initialize module
        super().__init__(
            config=config,
            model=model,
            loss=loss,
            n_train_steps=n_train_steps,
        )
        # algorithm hyperparameters
        self.fixmatch_lambda = kwargs["self_training_lambda"]
        self.target_align = kwargs["target_align"]
        self.confidence_threshold = kwargs["self_training_threshold"]
        self.process_pseudolabels_function = pseudolabel_multiclass_logits

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
            - batch (x, y, m): a batch of data yielded by data loaders
            - unlabeled_batch: examples ((x_weak, x_strong), m) where x_weak is weakly augmented but x_strong is strongly augmented
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - y_pred (Tensor): model output for batch
                - unlabeled_weak_y_pseudo (Tensor): pseudolabels on x_weak of the unlabeled batch, already thresholded
                - unlabeled_strong_y_pred (Tensor): model output on x_strong of the unlabeled batch, already thresholded
        """
        # Labeled examples
        x, y_true = batch[:2]
        x = x.to(self.device)
        y_true = y_true.to(self.device)

        # package the results
        results = {
            "y_true": y_true,
        }

        # if self.source_balanced and source_marginal is not None:
        #     results['source_marginal'] = torch.tensor(source_marginal).to(self.device)

        # if self.use_target_marginal and target_marginal is not None:
        #     results['im_weights'] = torch.divide(torch.tensor(target_marginal).to(self.device),\
        #             torch.tensor(source_marginal).to(self.device))

        pseudolabels_kept_frac = 0

        # Unlabeled examples
        if unlabeled_batch is not None:
            if self.target_align and target_average is not None:
                alignment_dist = torch.divide(
                    torch.tensor(target_marginal).to(self.device),
                    torch.tensor(target_average).to(self.device),
                )

            (x_weak, x_strong) = unlabeled_batch[0]
            x_weak = x_weak.to(self.device)
            x_strong = x_strong.to(self.device)

            with torch.no_grad():
                outputs = self.model(x_weak)

                if self.target_align and target_average is not None:
                    (
                        _,
                        pseudolabels,
                        pseudolabels_kept_frac,
                        mask,
                    ) = self.process_pseudolabels_function(
                        outputs, self.confidence_threshold, alignment_dist
                    )
                else:
                    (
                        _,
                        pseudolabels,
                        pseudolabels_kept_frac,
                        mask,
                    ) = self.process_pseudolabels_function(
                        outputs, self.confidence_threshold
                    )

                results["unlabeled_weak_y_pseudo"] = detach_and_clone(pseudolabels)

        results["pseudolabels_kept_frac"] = pseudolabels_kept_frac

        # Concat and call forward
        n_lab = x.shape[0]
        if unlabeled_batch is not None:
            x_concat = torch.cat((x, x_strong), dim=0)
        else:
            x_concat = x

        outputs = self.model(x_concat)
        results["y_pred"] = outputs[:n_lab]
        if unlabeled_batch is not None:
            results["unlabeled_strong_y_pred"] = (
                outputs[n_lab:] if mask is None else outputs[n_lab:][mask]
            )

        return results

    def objective(self, results):
        # Labeled loss
        classification_loss = self.loss(results["y_pred"], results["y_true"])

        # if self.use_target_marginal:
        #     classification_loss = torch.mean(classification_loss*results["im_weights"][results["y_true"]])

        # elif self.source_balanced:
        #     classification_loss = torch.mean(classification_loss/results["source_marginal"][results["y_true"]]/ self.num_classes)

        # Pseudolabeled loss
        if "unlabeled_weak_y_pseudo" in results:
            loss_output = self.loss(
                results["unlabeled_strong_y_pred"],
                results["unlabeled_weak_y_pseudo"],
            )
            consistency_loss = loss_output * results["pseudolabels_kept_frac"]
        else:
            consistency_loss = 0

        return classification_loss + self.fixmatch_lambda * consistency_loss

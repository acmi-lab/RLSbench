import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from RLSbench.algorithms.single_model_algorithm import SingleModelAlgorithm
from RLSbench.losses import initialize_loss
from RLSbench.models.initializer import initialize_model
from RLSbench.models.model_utils import linear_probe
from RLSbench.utils import detach_and_clone, pseudolabel_multiclass_logits
from RLSbench.optimizer import initialize_optimizer_with_model_params

logger = logging.getLogger("label_shift")


class SENTRY(SingleModelAlgorithm):
    """
    Sentry: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation

    Original paper:
        @inproceedings{prabhu2021sentry,
            title={Sentry: Selective entropy optimization via committee consistency for unsupervised domain adaptation},
            author={Prabhu, Viraj and Khare, Shivam and Kartik, Deeksha and Hoffman, Judy},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            pages={8558--8567},
            year={2021}
        }
    """

    def __init__(self, config, dataloader, loss_function, n_train_steps, **kwargs):
        logger.info("Intializing SENTRY algorithm model")

        model = initialize_model(
            model_name=config.model,
            dataset_name=config.dataset,
            num_classes=config.num_classes,
            featurize=True,
            pretrained=config.pretrained,
            pretrained_path=config.pretrained_path,
        )

        # if config.algorithm.startswith("IW"):
        #     self.use_target_marginal = True
        # else:
        self.use_target_marginal = False

        # if config.source_balanced or self.use_target_marginal:
        #     loss = initialize_loss(loss_function, reduction='none')
        # else:
        loss = initialize_loss(loss_function)

        # if config.pretrained:
        #     model = linear_probe(model, dataloader, device= config.device, progress_bar=config.progress_bar)

        params = [
            {"params": model[0].parameters(), "lr": config.lr * 0.1},
            # {"params": self.bottleneck.parameters(), "lr": classifier_lr},
            {"params": model[1].parameters(), "lr": config.lr},
        ]

        model = nn.Sequential(*model)

        # self.optimizer =
        self.optimizer = initialize_optimizer_with_model_params(config, params)

        # initialize module
        super().__init__(
            config=config,
            model=model,
            loss=loss,
            n_train_steps=n_train_steps,
        )
        # algorithm hyperparameters
        self.lambda_src = kwargs["lambda_src"]
        self.lambda_unsup = kwargs["lambda_unsup"]
        self.lambda_ent = kwargs["lambda_ent"]

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

        y_pred = self.model(x)
        # package the results
        results = {
            "y_true": y_true,
            "y_pred": y_pred,
        }

        # Unlabeled examples
        if unlabeled_batch is not None:
            x = unlabeled_batch[0]
            x_weak, x_strong_augs = x[0], x[1:]
            x_weak = x_weak.to(self.device)

            y_weak_pred = self.model(x_weak)

            y_weak_pseudo_label = y_weak_pred.max(dim=1)[1].detach().reshape(-1)

            correct_mask, incorrect_mask = torch.zeros_like(y_weak_pseudo_label).to(
                self.device
            ), torch.zeros_like(y_weak_pseudo_label).to(self.device)
            score_t_aug_pos, score_t_aug_neg = torch.zeros_like(
                y_weak_pred
            ), torch.zeros_like(y_weak_pred)

            for x_strong in x_strong_augs:
                y_strong_pred = self.model(x_strong.to(self.device))
                y_strong_pseudo_label = y_strong_pred.max(dim=1)[1].reshape(-1)

                consistent_idxs = (
                    y_weak_pseudo_label == y_strong_pseudo_label
                ).detach()
                inconsistent_idxs = (
                    y_weak_pseudo_label != y_strong_pseudo_label
                ).detach()

                correct_mask = correct_mask + consistent_idxs.type(torch.uint8)
                incorrect_mask = incorrect_mask + inconsistent_idxs.type(torch.uint8)

                score_t_aug_pos[consistent_idxs, :] = y_strong_pred[consistent_idxs, :]
                score_t_aug_neg[inconsistent_idxs, :] = y_strong_pred[
                    inconsistent_idxs, :
                ]

            correct_mask, incorrect_mask = correct_mask >= 2, incorrect_mask >= 2

            correct_ratio = (correct_mask).sum().item() / x_weak.shape[0]
            incorrect_ratio = (incorrect_mask).sum().item() / x_weak.shape[0]

            results["correct_ratio"] = correct_ratio
            results["incorrect_ratio"] = incorrect_ratio

            results["y_weak_pred"] = y_weak_pred
            results["y_weak_pseudo_label"] = y_weak_pseudo_label

            results["score_t_aug_pos"] = score_t_aug_pos[correct_mask]
            results["score_t_aug_neg"] = score_t_aug_neg[incorrect_mask]

            if target_average is not None:
                # import pdb; pdb.set_trace()
                results["target_average"] = torch.tensor(target_average).to(self.device)

            # if self.use_target_marginal and target_marginal is not None:
            #     results['im_weights'] = torch.divide(torch.tensor(target_marginal).to(self.device),\
            #         torch.tensor(source_marginal).to(self.device))
            #     results['target_marginal'] = torch.tensor(target_marginal).to(self.device)

            # import  pdb; pdb.set_trace()
            # if self.source_balanced and source_marginal is not None:
            # results['source_marginal'] = torch.tensor(source_marginal).to(self.device)

        return results

    def objective(self, results):
        # Labeled loss
        classification_loss = self.loss(results["y_pred"], results["y_true"])

        # if self.use_target_marginal:
        #     classification_loss = self.lambda_src * torch.mean(classification_loss*results["im_weights"][results["y_true"]])

        #     loss_cent = 0.0

        #     if results["correct_ratio"] > 0.0:
        #         probs_t_pos = F.softmax(results["score_t_aug_pos"], dim=1)
        #         loss_cent_correct = self.lambda_ent * -torch.mean(torch.sum(probs_t_pos * (torch.log(probs_t_pos + 1e-12)), 1))
        #         loss_cent += loss_cent_correct* results["correct_ratio"]

        #     if results["incorrect_ratio"] > 0.0:
        #         probs_t_neg = F.softmax(results["score_t_aug_neg"], dim=1)
        #         loss_cent_incorrect = self.lambda_ent * torch.mean(torch.sum(probs_t_neg * (torch.log(probs_t_neg + 1e-12)), 1))
        #         loss_cent += loss_cent_incorrect* results["incorrect_ratio"]

        # elif self.source_balanced:
        #     classification_loss = torch.mean(classification_loss/results["source_marginal"][results["y_true"]]/ self.num_classes)
        #     loss_cent = 0.0

        #     target_marginal = results["target_marginal"]
        #     target_marginal[target_marginal == 0] = 1.0

        #     if results["correct_ratio"] > 0.0:
        #         probs_t_pos = F.softmax(results["score_t_aug_pos"], dim=1)
        #         loss_cent_correct = torch.sum(probs_t_pos * (torch.log(probs_t_pos + 1e-12)), 1)

        #         loss_cent_correct = -torch.mean(loss_cent_correct* 1.0/target_marginal[results["y_weak_pseudo_label"]]/ self.num_classes)

        #         loss_cent += loss_cent_correct* results["correct_ratio"]

        #     if results["incorrect_ratio"] > 0.0:
        #         probs_t_neg = F.softmax(results["score_t_aug_neg"], dim=1)
        #         loss_cent_incorrect = torch.sum(probs_t_neg * (torch.log(probs_t_neg + 1e-12)), 1)

        #         loss_cent_incorrect = torch.mean(loss_cent_correct* 1.0/target_marginal[results["y_weak_pseudo_label"]]/ self.num_classes)

        #         loss_cent += loss_cent_incorrect* results["incorrect_ratio"]

        # else:
        classification_loss = self.lambda_src * classification_loss

        loss_cent = 0.0

        if results["correct_ratio"] > 0.0:
            probs_t_pos = F.softmax(results["score_t_aug_pos"], dim=1)
            loss_cent_correct = self.lambda_ent * -torch.mean(
                torch.sum(probs_t_pos * (torch.log(probs_t_pos + 1e-12)), 1)
            )
            loss_cent += loss_cent_correct * results["correct_ratio"]

        if results["incorrect_ratio"] > 0.0:
            probs_t_neg = F.softmax(results["score_t_aug_neg"], dim=1)
            loss_cent_incorrect = self.lambda_ent * torch.mean(
                torch.sum(probs_t_neg * (torch.log(probs_t_neg + 1e-12)), 1)
            )
            loss_cent += loss_cent_incorrect * results["incorrect_ratio"]

        if "target_average" in results:
            loss_infoent = self.lambda_unsup * torch.mean(
                torch.sum(
                    F.softmax(results["y_weak_pred"], dim=1)
                    * torch.log(results["target_average"] + 1e-12).reshape(
                        1, self.num_classes
                    ),
                    dim=1,
                )
            )

        else:
            loss_infoent = 0.0

        return classification_loss + loss_cent + loss_infoent

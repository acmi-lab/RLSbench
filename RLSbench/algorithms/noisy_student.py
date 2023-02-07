import logging

import torch
import torch.nn as nn
from RLSbench.algorithms.single_model_algorithm import \
    SingleModelAlgorithm
from RLSbench.losses import initialize_loss
from RLSbench.models.initializer import initialize_model
from RLSbench.models.model_utils import linear_probe
from RLSbench.utils import concat_input, move_to

logger = logging.getLogger("label_shift")

class DropoutModel(nn.Module):
    def __init__(self, featurizer, classifier, dropout_rate):
        super().__init__()
        self.featurizer = featurizer
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = classifier

    def forward(self, x):
        features = self.featurizer(x)
        features_sparse = self.dropout(features)
        return self.classifier(features_sparse)


class NoisyStudent(SingleModelAlgorithm):
    """
    Noisy Student.
    This algorithm was originally proposed as a semi-supervised learning algorithm.

    One run of this codebase gives us one iteration (load a teacher, train student). To run another iteration,
    re-run the previous command, pointing config.teacher_model_path to the trained student weights.

    To warm start the student model, point config.pretrained_model_path to config.teacher_model_path

    Based on the original paper, loss is of the form
        \ell_s + \ell_u
    where
        \ell_s = cross-entropy with true labels; student predicts with noise
        \ell_u = cross-entropy with pseudolabel generated without noise; student predicts with noise
    The student is noised using:
        - Input images are augmented using RandAugment
        - Single dropout layer before final classifier (fc) layer
    We do not use stochastic depth.

    Pseudolabels are generated in run_expt.py on unlabeled images that have only been randomly cropped and flipped ("weak" transform).
    By default, we use hard pseudolabels; use the --soft_pseudolabels flag to add soft pseudolabels.

    This code only supports a teacher that is the same class as the student (e.g. both densenet121s)

    Original paper:
        @inproceedings{xie2020self,
            title={Self-training with noisy student improves imagenet classification},
            author={Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V},
            booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
            pages={10687--10698},
            year={2020}
            }
    """

    def __init__(self, config, dataloader, loss_function, n_train_steps, **kwargs):

        logger.info("Intializing Noisy Student algorithm model")

        # initialize student model with dropout before last layer
        if kwargs['noisystudent_add_dropout']:
            featurizer, classifier = initialize_model(
                model_name = config.model, 
                dataset_name = config.dataset,
                num_classes = config.num_classes,
                featurize = True, 
                pretrained=config.pretrained,
                pretrained_path=config.pretrained_path,
            )

            model = (featurizer, classifier)

            # if config.pretrained : 
            #     featurizer, classifier = linear_probe(model, dataloader, device= config.device, progress_bar=config.progress_bar)

            student_model = DropoutModel(
                featurizer, classifier, kwargs['noisystudent_dropout_rate']
            )

        else:
            featurizer, classifier = initialize_model(
                model_name = config.model, 
                dataset_name = config.dataset,
                num_classes = config.num_classes,
                featurize = True, 
                pretrained=config.pretrained,
                pretrained_path=config.pretrained_path,
            )
            model = (featurizer, classifier)

            # if config.pretrained and config.featurize: 
            #     model = linear_probe(model, dataloader, device= config.device)
    
            student_model = nn.Sequential(*model)

        # if config.algorithm.startswith("IW"):
        #     self.use_target_marginal = True
        # else:
        self.use_target_marginal = False

        # if config.source_balanced or self.use_target_marginal:
        #     loss = initialize_loss(loss_function, reduction='none')
        # else:
        loss = initialize_loss(loss_function)

        # initialize module
        super().__init__(
            config=config,
            model=student_model,
            loss=loss,
            n_train_steps=n_train_steps,
        )
        
        self.source_balanced = config.source_balanced
        self.num_classes = config.num_classes

    def process_batch(self, batch, unlabeled_batch=None,  target_marginal=None, source_marginal = None, target_average=None):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (x, y, m): a batch of data yielded by data loaders
            - unlabeled_batch: examples (x, y_pseudo, m) where y_pseudo is an already-computed teacher pseudolabel
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - y_pred (Tensor): model output for batch
                - unlabeled_y_pseudo (Tensor): pseudolabels for unlabeled batch (from loader)
                - unlabeled_y_pred (Tensor): model output on unlabeled batch
        """
        # Labeled examples
        x, y_true = batch[:2]
        n_lab = len(y_true)
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)

        # package the results
        results = { "y_true": y_true}

        # Unlabeled examples with pseudolabels
        if unlabeled_batch is not None:
            x_unlab, y_pseudo = unlabeled_batch[:2]
            x_unlab = move_to(x_unlab, self.device)
            y_pseudo = move_to(y_pseudo, self.device)

            results["unlabeled_y_pseudo"] = y_pseudo

            x_cat = concat_input(x, x_unlab)

            outputs = self.get_model_output(x_cat)

            results["y_pred"] = outputs[:n_lab]
            results["unlabeled_y_pred"] = outputs[n_lab:]
        else:
            results["y_pred"] = self.get_model_output(x)

        # if self.use_target_marginal and target_marginal is not None:
        #     results['im_weights'] = torch.divide(torch.tensor(target_marginal).to(self.device),\
        #             torch.tensor(source_marginal).to(self.device))

        # if self.source_balanced and source_marginal is not None:
        #     results['source_marginal'] = torch.tensor(source_marginal).to(self.device)

        return results

    def objective(self, results):
        # Labeled loss
        classification_loss = self.loss(
            results["y_pred"], results["y_true"]
        )

        # if self.use_target_marginal: 
        #     classification_loss = torch.mean(classification_loss*results["im_weights"][results["y_true"]])

        # elif self.source_balanced: 
        #     classification_loss = torch.mean(classification_loss/results["source_marginal"][results["y_true"]]/ self.num_classes)


        # Pseudolabel loss
        if "unlabeled_y_pseudo" in results:
            consistency_loss = self.loss(
                results["unlabeled_y_pred"],
                results["unlabeled_y_pseudo"],
            )
            # if self.use_target_marginal or self.source_balanced: 
            # consistency_loss = torch.mean(consistency_loss)
        
        else:
            consistency_loss = 0

        return classification_loss + consistency_loss

import logging

import torch
import torch.nn as nn
from RLSbench.algorithms.algorithm import Algorithm
from RLSbench.models.initializer import initialize_model
from RLSbench.models.model_utils import test_CORAL_params, train_CORAL
from RLSbench.utils import load, move_to

logger = logging.getLogger("label_shift")


class CORALModel(nn.Module):
    def __init__(self, featurizer, classifier):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        # self.mean = None
        self.cov_inv = None

    def forward(self, x):
        features = self.featurizer(x)

        if self.cov_inv is not None:
            # centered = features - self.mean
            features = torch.mm(features, self.cov_inv)

        outputs = self.classifier(features)

        return outputs


class CORAL(Algorithm):
    def __init__(self, config):
        logger.info("Initializing model...")

        if config.algorithm.startswith("IW"):
            self.use_target_marginal = True
        else:
            self.use_target_marginal = False

        model = initialize_model(
            model_name=config.model,
            dataset_name=config.dataset,
            num_classes=config.num_classes,
            featurize=True,
            pretrained=False,
        )

        linear_layer = nn.Linear(model[0].d_out, config.num_classes)
        model = CORALModel(model[0], linear_layer)

        # initialize module
        super().__init__(
            device=config.device,
        )

        model = model.to(config.device)

        self.model = model
        self.source_balanced = config.source_balanced
        self.num_classes = config.num_classes

    def get_model_output(self, x):
        return self.model(x)

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
            "y_true": y_true,
            "y_pred": outputs,
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

    def adapt(
        self,
        source_loader,
        target_loader,
        target_marginal=None,
        source_marginal=None,
        target_average=None,
        pretrained_path=None,
    ):
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

            load(self.model[0], pretrained_path, device=self.device)

        # self.train(True)

        logger.info("Adapting model to CORAL ...")

        im_weights = None

        if self.use_target_marginal:
            im_weights = torch.divide(
                torch.tensor(target_marginal).to(self.device),
                torch.tensor(source_marginal).to(self.device),
            )

        self.model = train_CORAL(
            self.model, source_loader, im_weights=im_weights, device=self.device
        )

        # self.model.mean,
        self.model.cov_inv = test_CORAL_params(
            self.model, target_loader, device=self.device
        )

        # self.model.mean = self.model.mean.to(self.device)
        self.model.cov_inv = self.model.cov_inv.to(self.device)

    def reset(self):
        # self.model.mean = None
        self.model.cov_inv = None

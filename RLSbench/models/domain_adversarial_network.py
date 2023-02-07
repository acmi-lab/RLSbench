from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class DomainDiscriminator(nn.Sequential):
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_
    In the original paper and implementation, we distinguish whether the input features come
    from the source domain or the target domain.

    We extended this to work with multiple domains, which is controlled by the n_domains
    argument.

    Args:
        in_feature (int): dimension of the input feature
        n_domains (int): number of domains to discriminate
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, n_domains)`
    """

    def __init__(
        self, in_feature: int, n_domains, hidden_size: int = 1024, batch_norm=True
    ):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_domains),
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, n_domains),
            )

    def get_parameters_with_lr(self, lr) -> List[Dict]:
        return [{"params": self.parameters(), "lr": lr}]

class GradientReverseFunction(Function):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0
    ) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, input, coeff):
        return GradientReverseFunction.apply(input, coeff)


class DomainAdversarialNetwork(nn.Module):
    def __init__(self, featurizer, classifier, n_domains, num_classes, bottleneck_dim=256):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        # self.classifier = nn.Linear(bottleneck_dim, num_classes)
        # self.bottleneck = nn.Linear(featurizer.d_out, bottleneck_dim)
        # self.domain_classifier = DomainDiscriminator(featurizer.d_out, n_domains)
        self.domain_classifier = DomainDiscriminator(featurizer.d_out, n_domains,  batch_norm=False)
        
        self.gradient_reverse_layer = GradientReverseLayer()

    def forward(self, input, coeff = 1.0, domain_classifier=False):
        features = self.featurizer(input)
        # features = self.bottleneck(features)
        y_pred = self.classifier(features)
        if domain_classifier:
            features = self.gradient_reverse_layer(features, coeff)
            domains_pred = self.domain_classifier(features)
            return y_pred, domains_pred
        else: 
            return y_pred

    def get_parameters_with_lr(self, featurizer_lr, classifier_lr, discriminator_lr) -> List[Dict]:
        """
        Adapted from https://github.com/thuml/Transfer-Learning-Library

        A parameter list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer
        """
        # In TLL's implementation, the learning rate of this classifier is set 10 times to that of the
        # feature extractor for better accuracy by default. For our implementation, we allow the learning
        # rates to be passed in separately for featurizer and classifier.
        params = [
            {"params": self.featurizer.parameters(), "lr": featurizer_lr},
            # {"params": self.bottleneck.parameters(), "lr": classifier_lr},
            {"params": self.classifier.parameters(), "lr": classifier_lr},
        ]
        return params + self.domain_classifier.get_parameters_with_lr(discriminator_lr)


class classifier_deep(nn.Module):
    def __init__(self, num_classes, inc=4096, temp=0.05):
        super(classifier_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_classes, bias=False)
        self.gradient_reverse_layer = GradientReverseLayer()
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = self.gradient_reverse_layer(x, eta)

        # x = F.normalize(x)
        x_out = self.fc2(x) #/ self.temp
        return x_out

    
class COALNetwork(nn.Module):
    def __init__(self, featurizer, num_classes):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier_deep(num_classes = num_classes, inc = featurizer.d_out)

    def forward(self, input, coeff= 1.0, reverse = False):
        features = self.featurizer(input)

        y_pred = self.classifier(features, reverse=reverse, eta= coeff)
        return y_pred


    def get_parameters_with_lr(self, featurizer_lr, classifier_lr) -> List[Dict]:
        """
        Adapted from https://github.com/thuml/Transfer-Learning-Library

        A parameter list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer
        """
        # In TLL's implementation, the learning rate of this classifier is set 10 times to that of the
        # feature extractor for better accuracy by default. For our implementation, we allow the learning
        # rates to be passed in separately for featurizer and classifier.
        params = [
            {"params": self.featurizer.parameters(), "lr": featurizer_lr},
            {"params": self.classifier.parameters(), "lr": classifier_lr},
        ]
        return params 

class ConditionalDomainAdversarialNetwork(nn.Module):
    def __init__(self, featurizer, classifier, n_domains, num_classes, bottleneck_dim=256):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        self.bottleneck = nn.Linear(featurizer.d_out, bottleneck_dim)
        self.domain_classifier = DomainDiscriminator(bottleneck_dim*num_classes, n_domains, batch_norm=False)
        self.gradient_reverse_layer = GradientReverseLayer()

    def forward(self, input, coeff = 1.0, domain_classifier=False):
        features = self.featurizer(input)
        features = self.bottleneck(features)

        y_pred = self.classifier(features)

        if domain_classifier:
            softmax_out = F.softmax(y_pred, dim=1).detach()
            op_out = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))
            op_out = op_out.view(-1, softmax_out.size(1) * features.size(1))

            op_out = self.gradient_reverse_layer(op_out, coeff)
            # op_out.register_hook(grl_hook(coeff))
            # import pdb; pdb.set_trace()
            domains_pred = self.domain_classifier(op_out)
            return y_pred, domains_pred
        else: 
            return y_pred

    def get_parameters_with_lr(self, featurizer_lr, classifier_lr, discriminator_lr) -> List[Dict]:
        """
        Adapted from https://github.com/thuml/Transfer-Learning-Library

        A parameter list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer
        """
        # In TLL's implementation, the learning rate of this classifier is set 10 times to that of the
        # feature extractor for better accuracy by default. For our implementation, we allow the learning
        # rates to be passed in separately for featurizer and classifier.
        params = [
            {"params": self.featurizer.parameters(), "lr": featurizer_lr},
            {"params": self.bottleneck.parameters(), "lr": classifier_lr},
            {"params": self.classifier.parameters(), "lr": classifier_lr},
        ]
        return params + self.domain_classifier.get_parameters_with_lr(discriminator_lr)



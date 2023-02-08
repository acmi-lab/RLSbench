import torch
import torch.nn as nn
import torch.nn.functional as F
from RLSbench.models.domain_adversarial_network import (
    GradientReverseFunction,
    GradientReverseLayer,
)


class MDDNet(nn.Module):
    def __init__(
        self,
        featurizer,
        class_num,
        bottleneck_dim=1024,
        classifier_width=1024,
        classifier_depth=2,
    ):
        super().__init__()

        self.class_num = class_num
        self.bottleneck_dim = bottleneck_dim
        self.freeze_backbone = False
        self.normalize_features = False

        self.base_network = featurizer

        self.use_bottleneck = True
        self.create_bottleneck_layer(use_dropout=True)

        self.create_f_and_fhat_classifiers(
            bottleneck_dim,
            classifier_width,
            class_num,
            classifier_depth,
            use_dropout=True,
        )

        self.softmax = nn.Softmax(dim=1)

        # collect parameters
        self.parameter_list = [
            {"params": self.base_network.parameters(), "lr_scale": 0.1},
            {"params": self.bottleneck_layer.parameters(), "lr_scale": 1},
            {"params": self.classifier_layer.parameters(), "lr_scale": 1},
            {"params": self.classifier_layer_2.parameters(), "lr_scale": 1},
        ]

    def create_bottleneck_layer(self, use_dropout):
        bottleneck_layer_list = [
            nn.Linear(self.base_network.output_num(), self.bottleneck_dim),
            nn.BatchNorm1d(self.bottleneck_dim),
            nn.ReLU(),
        ]
        if use_dropout is True:
            bottleneck_layer_list.append(nn.Dropout(0.5))

        self.bottleneck_layer = nn.Sequential(*bottleneck_layer_list)

        # init
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)

    def create_f_and_fhat_classifiers(
        self,
        bottleneck_dim,
        classifier_width,
        class_num,
        classifier_depth,
        use_dropout=True,
    ):
        self.classifier_layer = self.create_classifier(
            bottleneck_dim,
            classifier_width,
            class_num,
            classifier_depth,
            use_dropout=use_dropout,
        )
        self.classifier_layer_2 = self.create_classifier(
            bottleneck_dim,
            classifier_width,
            class_num,
            classifier_depth,
            use_dropout=use_dropout,
        )
        self.initialize_classifiers()

    def create_classifier(
        self, bottleneck_dim, width, class_num, depth=2, use_dropout=True
    ):
        layer_list = []
        input_size = bottleneck_dim
        for ith_layer in range(depth - 1):
            layer_list.append(nn.Linear(input_size, width))

            layer_list.append(nn.ReLU())

            if use_dropout is True:
                layer_list.append(nn.Dropout(0.5))

            input_size = width

        layer_list.append(nn.Linear(width, class_num))
        classifier = nn.Sequential(*layer_list)
        return classifier

    def forward(self, inputs):
        features = self.feature_forward(inputs)
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        # gradient reversal layer helps fuse the minimax problem into one loss function
        features_adv = GradientReverseLayer.apply(features)
        outputs_adv = self.classifier_layer_2(features_adv)

        return features, outputs, softmax_outputs, outputs_adv

    def feature_forward(self, inputs):
        if self.freeze_backbone is True:
            with torch.no_grad():
                features = self.base_network(inputs)
        else:
            features = self.base_network(inputs)

        if self.use_bottleneck:
            features = self.bottleneck_layer(features)

        if self.normalize_features is True:
            features_norm = torch.norm(features, p=2, dim=1).detach()
            features = features / features_norm.unsqueeze(1)
        return features

    def logits_forward(self, inputs):
        features = self.feature_forward(inputs)
        logits = self.classifier_layer(features)
        return logits

    def initialize_classifiers(self):
        self.xavier_initialization(self.classifier_layer)
        self.xavier_initialization(self.classifier_layer_2)

    def xavier_initialization(self, layers):
        for layer in layers:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(0.0)

    def initialize_bottleneck(self):
        for b_layer in self.bottleneck_layer:
            if type(b_layer) == nn.Linear:
                torch.nn.init.xavier_normal_(b_layer.weight)
                b_layer.bias.data.fill_(0.0)

    def get_parameter_list(self):
        c_net_params = self.parameter_list
        return c_net_params


def get_mdd_loss(outputs, outputs_adv, labels_source, class_criterion, srcweight):
    # f(x)
    outputs_src = outputs.narrow(0, 0, labels_source.size(0))
    label_preds_src = outputs_src.max(1)[1]
    outputs_tgt = outputs.narrow(
        0, labels_source.size(0), outputs.size(0) - labels_source.size(0)
    )
    probs_tgt = F.softmax(outputs_tgt, dim=1)
    # f'(x)
    outputs_adv_src = outputs_adv.narrow(0, 0, labels_source.size(0))
    outputs_adv_tgt = outputs_adv.narrow(
        0, labels_source.size(0), outputs.size(0) - labels_source.size(0)
    )

    # classification loss on source domain
    # if self.args.mask_classifier is True:
    outputs_src_masked, _, _ = mask_clf_outputs(
        outputs_src, outputs_adv_src, outputs_adv_tgt, labels_source
    )
    classifier_loss = class_criterion(outputs_src_masked, labels_source)

    outputs_src, outputs_adv_src, outputs_adv_tgt = mask_clf_outputs(
        outputs_src, outputs_adv_src, outputs_adv_tgt, labels_source
    )

    # use $f$ as the target for $f'$
    target_adv = outputs.max(1)[1]  # categorical labels from $f$
    target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
    target_adv_tgt = target_adv.narrow(
        0, labels_source.size(0), outputs.size(0) - labels_source.size(0)
    )

    # source classification acc
    classifier_acc = (
        label_preds_src == labels_source
    ).sum().float() / labels_source.size(0)

    # adversarial loss for source domain
    classifier_loss_adv_src = class_criterion(outputs_adv_src, target_adv_src)

    # adversarial loss for target domain, opposite sign with source domain
    prob_adv = 1 - F.softmax(outputs_adv_tgt, dim=1)
    prob_adv = prob_adv.clamp(min=1e-7)
    logloss_tgt = torch.log(prob_adv)
    classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

    # total adversarial loss
    adv_loss = srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

    # loss for explicit alignment

    total_loss = classifier_loss + adv_loss

    return total_loss


def mask_clf_outputs(outputs_src, outputs_adv_src, outputs_adv_tgt, labels_source):
    mask = torch.zeros(outputs_src.shape[1])
    mask[labels_source.unique()] = 1
    mask = mask.repeat((outputs_src.shape[0], 1)).cuda()
    outputs_src = outputs_src * mask
    outputs_adv_src = outputs_adv_src * mask
    outputs_adv_tgt = outputs_adv_tgt * mask
    return outputs_src, outputs_adv_src, outputs_adv_tgt

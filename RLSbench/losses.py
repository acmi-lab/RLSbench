import torch.nn as nn
from wilds.common.metrics.loss import ElementwiseLoss

# from wilds.common.metrics.all_metrics import MSE
from RLSbench.utils import cross_entropy_with_logits_loss


def initialize_loss(loss, reduction='mean'):
    if loss == 'cross_entropy':
        return nn.CrossEntropyLoss(reduction=reduction)

    elif loss == 'cross_entropy_logits':
        return ElementwiseLoss(loss_fn=cross_entropy_with_logits_loss)

    else:
        raise ValueError(f'loss {loss} not recognized')

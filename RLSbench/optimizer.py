from torch.optim import SGD, Adam, AdamW

def initialize_optimizer(config, model):
    # initialize optimizers
    if config.optimizer=='SGD':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = SGD(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs)
    elif config.optimizer == 'Adam':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs)
    elif config.optimizer=='AdamW':
        params = filter(lambda p: p.requires_grad, model.parameters())
        
        # import pdb; pdb.set_trace()
        optimizer = AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay, 
            **config.optimizer_kwargs)
    else:
        raise ValueError(f'Optimizer {config.optimizer} not recognized.')

    return optimizer

def initialize_optimizer_with_model_params(config, params):
    if config.optimizer=='SGD':
        optimizer = SGD(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs
        )
    elif config.optimizer == 'Adam':
        optimizer = Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs
        )
    elif config.optimizer=='AdamW':
        optimizer = AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs)
    else:
        raise ValueError(f'Optimizer {config.optimizer} not supported.')

    return optimizer

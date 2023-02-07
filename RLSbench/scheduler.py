import numpy as np
from torch.optim.lr_scheduler import (LambdaLR, MultiStepLR, ReduceLROnPlateau,
                                      StepLR)


def initialize_scheduler(config, optimizer, n_train_steps):
    # construct schedulers
    if config.scheduler is None:
        return None
    elif config.scheduler == 'linear_schedule_with_warmup':
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=n_train_steps,
            num_warmup_steps=int(config.scheduler_kwargs['warmup_frac']* n_train_steps))
        step_every_batch = True
        use_metric = False
    elif config.scheduler == 'cosine_schedule_with_warmup':
        from transformers import get_cosine_schedule_with_warmup
        if 'warmup_frac' not in config.scheduler_kwargs: 
            config.scheduler_kwargs['num_warmup_steps'] = 0

        else: 
            config.scheduler_kwargs['num_warmup_steps'] = int(config.scheduler_kwargs['warmup_frac']* n_train_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=n_train_steps,
            num_warmup_steps=config.scheduler_kwargs['num_warmup_steps']
            )
        step_every_batch = True
        use_metric = False

    # elif config.scheduler=='ReduceLROnPlateau':
    #     assert config.scheduler_metric_name, f'scheduler metric must be specified for {config.scheduler}'
    #     scheduler = ReduceLROnPlateau(
    #         optimizer,
    #         **config.scheduler_kwargs)
    #     step_every_batch = False
    #     use_metric = True
    
    elif config.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, **config.scheduler_kwargs)
        step_every_batch = False
        use_metric = False
    elif config.scheduler == 'FixMatchLR':
        scheduler = LambdaLR(
            optimizer,
            lambda x: (1.0 + 10 * float(x) / n_train_steps) ** -0.75
        )
        step_every_batch = True
        use_metric = False
    elif config.scheduler == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, **config.scheduler_kwargs)
        step_every_batch = False
        use_metric = False
    else:
        raise ValueError(f'Scheduler: {config.scheduler} not supported.')

    # add an step_every_batch field
    scheduler.step_every_batch = step_every_batch
    return scheduler

def step_scheduler(scheduler):
    scheduler.step()

class LinearScheduleWithWarmupAndThreshold():
    """
    Linear scheduler with warmup and threshold for non lr parameters.
    Parameters is held at 0 until some T1, linearly increased until T2, and then held
    at some max value after T2.
    Designed to be called by step_scheduler() above and used within Algorithm class.
    Args:
        - last_warmup_step: aka T1. for steps [0, T1) keep param = 0
        - threshold_step: aka T2. step over period [T1, T2) to reach param = max value
        - max value: end value of the param
    """
    def __init__(self, max_value, last_warmup_step=0, threshold_step=1, step_every_batch=False):
        self.max_value = max_value
        self.T1 = last_warmup_step
        self.T2 = threshold_step
        assert (0 <= self.T1) and (self.T1 < self.T2)

        # internal tracker of which step we're on
        self.current_step = 0
        self.value = 0

        # required fields called in Algorithm when stepping schedulers
        self.step_every_batch = step_every_batch

    def step(self):
        """This function is first called AFTER step 0, so increment first to set value for next step"""
        self.current_step += 1
        if self.current_step < self.T1:
            self.value = 0
        elif self.current_step < self.T2:
            self.value = (self.current_step - self.T1) / (self.T2 - self.T1) * self.max_value
        else:
            self.value = self.max_value


class CoeffSchedule(): 
    def __init__(self, max_iter, high = 1.0, low=0.0, alpha = 10.0): 
        self.max_iter = max_iter 
        self.high = high 
        self.low = low 
        self.alpha = alpha 
        self.iter_num = 0.0
        self.step_every_batch = True
        self.value = 0.0 

    def step(self): 
        
        self.value = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha*self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
        self.iter_num = self.iter_num + 1

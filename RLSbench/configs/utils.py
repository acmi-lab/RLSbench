import copy

from RLSbench.configs.algorithm import algorithm_defaults
from RLSbench.configs.datasets import dataset_defaults


def populate_defaults(config):
    """Populates hyperparameters with defaults implied by choices
    of other hyperparameters."""

    orig_config = copy.deepcopy(config)
    assert config.dataset is not None, 'dataset must be specified'
    assert config.algorithm is not None, 'algorithm must be specified'

    config = populate_config(
        config,
        dataset_defaults[config.dataset]
    )

    # # implied defaults from choice of split
    # if config.dataset in split_defaults and config.split_scheme in split_defaults[config.dataset]:
    #     config = populate_config(
    #         config,
    #         split_defaults[config.dataset][config.split_scheme]
    #     )
    
    
    # implied defaults from choice of algorithm
    config = populate_config(
        config,
        algorithm_defaults[config.algorithm]
    )

    # implied defaults from choice of model
    # if config.model: config = populate_config(
    #     config,
    #     model_defaults[config.model],
    # )

    # # implied defaults from choice of scheduler
    # if config.scheduler: config = populate_config(
    #     config,
    #     scheduler_defaults[config.scheduler]
    # )

    # # implied defaults from choice of loader
    # config = populate_config(
    #     config,
    #     loader_defaults
    # )

    # import pdb; pdb.set_trace()
    if config.use_target: 
        assert config.target_split is not None, 'target_split must be specified if use_target is True' 

    if config.eval_only: 
        assert config.use_source_model is not None, 'use_source_model must be True if eval_only is True'
        
    if config.use_source_model: 
        assert config.source_model_path is not None, 'source_model_path must be specified if use_source_model is True'

    
    if config.simulate_label_shift:
        assert config.use_target is not None, 'when simulating label shift target split is needed'
        assert config.dirichlet_alpha is not None, 'when simulating label shift dirchilet alpha is needed'

    # basic checks
    required_fields = [
        'batch_size', 'model', 'loss_function', 'n_epochs', 'optimizer', 'lr', 'weight_decay', 'default_normalization'
        ]
    for field in required_fields:
        assert getattr(config, field) is not None, f"Must manually specify {field} for this setup."

    
    if config.pretrain_type == "imagenet": 
        config.pretrained = True

    if 'clip' in config.model:
        config.pretrain_type = "clip"
        config.transform = "clip"
        config.pretrained = True
        config.optimizer = "AdamW"
        config.optimizer_kwargs = {}
        config.scheduler = 'cosine_schedule_with_warmup'
        config.scheduler_kwargs = { 'warmup_frac': 0.1}

    if not config.pretrained:
        assert config.pretrain_type == "rand", "When pre-trained loading is False, pre-train type must be rand"

    if config.algorithm in ('ERM', 'ERM-aug', 'ERM-oracle', 'IS-ERM', 'IS-ERM-aug', 'IS-ERM-oracle'): 
        config.algorithm = f"{config.algorithm}-{config.pretrain_type}"
   

    # if "NoisyStudent" in config.algorithm: 
    #     assert "teacher_model_path" in config.noisystudent_kwargs, "Teacher model path needed for noisy student training."

    if "SENTRY" in config.algorithm: 
        import math 
        config.batch_size = int((config.batch_size/6))
        config.gradient_accumulation_steps = 6

    if ("DANN" in config.algorithm or \
    "FixMatch" in config.algorithm or \
    "PseudoLabel" in config.algorithm or \
    "NoisyStudent" in config.algorithm or \
    "COAL" in config.algorithm): 
        if "civilcomments" in config.dataset:
            config.batch_size = config.batch_size//3
            config.gradient_accumulation_steps = 3
        else:
            config.batch_size = config.batch_size//2
            config.gradient_accumulation_steps = 2
    
    return config


def populate_config(config, template: dict, force_compatibility=False):
    """Populates missing (key, val) pairs in config with (key, val) in template.
    Example usage: populate config with defaults
    Args:
        - config: namespace
        - template: dict
        - force_compatibility: option to raise errors if config.key != template[key]
    """
    if template is None:
        return config

    d_config = vars(config)
    for key, val in template.items():
        if not isinstance(val, dict): # config[key] expected to be a non-index-able
            if key not in d_config or d_config[key] is None:
                d_config[key] = val
            elif d_config[key] != val and force_compatibility:
                raise ValueError(f"Argument {key} must be set to {val}")

        else: # config[key] expected to be a kwarg dict
            for kwargs_key, kwargs_val in val.items():
                if kwargs_key not in d_config[key] or d_config[key][kwargs_key] is None:
                    d_config[key][kwargs_key] = kwargs_val
                elif d_config[key][kwargs_key] != kwargs_val and force_compatibility:
                    raise ValueError(f"Argument {key}[{kwargs_key}] must be set to {val}")
    return config

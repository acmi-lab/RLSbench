import argparse
import copy
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import RLSbench.configs.supported as supported
from RLSbench.algorithms.initializer import initialize_algorithm
from RLSbench.configs.utils import populate_defaults
from RLSbench.datasets import *
from RLSbench.helper import (
    adapt,
    eval_models,
    infer_predictions,
    train,
    rebalance_loader,
)
from RLSbench.label_shift_utils import *
from RLSbench.models.initializer import initialize_model
from RLSbench.transforms import initialize_transform
from RLSbench.utils import (
    ParseKwargs,
    ResultsLogger,
    initialize_wandb,
    load,
    log_config,
    parse_bool,
    set_seed,
)


from RLSbench.collate_functions import initialize_collate_function

try:
    import wandb
except Exception as e:
    pass

logFormatter = logging.Formatter(
    "%(asctime)s, [%(levelname)s, %(filename)s:%(lineno)d] %(message)s"
)

logger = logging.getLogger("label_shift")
logger.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


def get_parser():
    """Arg defaults are filled in according to configs/"""
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("-d", "--dataset", choices=supported_datasets, required=True)
    parser.add_argument(
        "--root_dir",
        required=True,
        help="The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--default_normalization",
        type=parse_bool,
        const=True,
        nargs="?",
        help="Default normalization is applied to the data.",
    )
    parser.add_argument("--mean", type=list, nargs="+", help="Mean of the dataset.")
    parser.add_argument("--std", type=list, nargs="+", help="Std of the dataset.")

    # Dataset
    parser.add_argument(
        "--use_source",
        type=parse_bool,
        const=True,
        nargs="?",
        help="If true, return source datasets.",
    )
    parser.add_argument(
        "--use_target",
        type=parse_bool,
        const=True,
        nargs="?",
        help="If true, return target datasets.",
    )
    parser.add_argument("--target_split", type=int, help="Identifies the target")
    parser.add_argument(
        "--use_unlabeled_y",
        type=parse_bool,
        const=True,
        nargs="?",
        help="If true, return unlabeled y for target.",
    )
    parser.add_argument(
        "--source_balanced",
        type=parse_bool,
        const=True,
        nargs="?",
        help="If true, source balanced ERM.",
    )
    # parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
    #                     help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
    parser.add_argument(
        "--split_fraction",
        type=float,
        default=0.80,
        help="Parameter that scales dataset splits down to the specified fraction, for validation purposes.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes in the dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for the data loader.",
    )

    # Label Shift params
    parser.add_argument(
        "--simulate_label_shift",
        type=parse_bool,
        const=True,
        default=False,
        nargs="?",
        help="If true, simulate label shift in target",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        help="Dirchilet alpha parameter for label shift",
    )
    parser.add_argument(
        "--estimation_method",
        default="RLLS",
        type=str,
        help="Estimation method for label shift",
        choices=supported.label_shift_adapt,
    )
    parser.add_argument(
        "--with_replacement",
        type=parse_bool,
        const=True,
        default=False,
        nargs="?",
        help="If true, simulate label shift in target with replacement",
    )

    # Model
    parser.add_argument("--model", choices=supported.models)
    # parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
    # help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument(
        "--pretrain_type",
        type=str,
        default=None,
        choices=supported.pretrainining_options,
        help="Type of pretraining to use.",
    )
    parser.add_argument(
        "--pretrained",
        type=parse_bool,
        const=True,
        nargs="?",
        help="If true, load pretrained model.",
    )
    # parser.add_argument('--linear_probe', type=parse_bool, const=True, default=True, nargs='?', help='If true, initialize with a linear probe model.')
    parser.add_argument(
        "--pretrained_path",
        default="./pretrained_models/resnet18_imagenet32.pt",
        type=str,
        help="Specify a path to pretrained model weights",
    )

    # Transforms
    parser.add_argument("--transform", choices=supported.transforms)
    parser.add_argument("--collate_function", choices=supported.collate_functions)
    parser.add_argument(
        "--additional_train_transform",
        choices=supported.additional_transforms,
        help="Optional data augmentations to layer on top of the default transforms.",
    )
    parser.add_argument(
        "--target_resolution",
        type=int,
        help="The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 for a standard ResNet.",
    )
    parser.add_argument("--resize_resolution", type=int)
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=512,
        help="Maximum number of tokens in a sentence.",
    )
    parser.add_argument(
        "--randaugment_n",
        type=int,
        help="Number of RandAugment transformations to apply.",
    )

    # Objective
    parser.add_argument(
        "--loss_function", type=str, default="cross_entropy", choices=supported.losses
    )

    # Algorithm
    parser.add_argument("--algorithm", required=True, choices=supported.algorithms)
    parser.add_argument(
        "--dann_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for algorithm passed as key1=value1 key2=value2",
    )
    parser.add_argument(
        "--cdan_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for algorithm passed as key1=value1 key2=value2",
    )
    parser.add_argument(
        "--fixmatch_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for algorithm passed as key1=value1 key2=value2",
    )
    parser.add_argument(
        "--pseudolabel_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for algorithm passed as key1=value1 key2=value2",
    )
    parser.add_argument(
        "--noisystudent_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for algorithm passed as key1=value1 key2=value2",
    )
    parser.add_argument(
        "--coal_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for algorithm passed as key1=value1 key2=value2",
    )
    parser.add_argument(
        "--sentry_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for algorithm passed as key1=value1 key2=value2",
    )

    parser.add_argument(
        "--use_source_model",
        type=parse_bool,
        const=True,
        default=None,
        nargs="?",
        help="If true, use an existing source model for test-time adaptation.",
    )
    parser.add_argument(
        "--source_model_path",
        default=None,
        type=str,
        help="Specify a path to ERM models weights.",
    )
    parser.add_argument(
        "--test_time_adapt",
        type=parse_bool,
        const=True,
        nargs="?",
        help="Specify if we are doing test time adaptation",
    )

    # Optimization
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients over.",
    )
    parser.add_argument("--optimizer", choices=supported.optimizers)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument(
        "--optimizer_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for optimizer initialization passed as key1=value1 key2=value2",
    )

    # Scheduler
    parser.add_argument("--scheduler", choices=supported.schedulers)
    parser.add_argument(
        "--scheduler_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for scheduler initialization passed as key1=value1 key2=value2",
    )

    # Evaluation
    parser.add_argument(
        "--eval_only", type=parse_bool, const=True, nargs="?", default=False
    )
    # parser.add_argument('--eval_epoch', type=int, default=None, help='If specified, evaluate on this epoch.')

    # Misc
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--log_dir", default="./logs_consistent", type=str)
    parser.add_argument("--evaluate_every", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument(
        "--save_last",
        type=parse_bool,
        const=True,
        nargs="?",
        default=True,
        help="Save the last epoch",
    )
    parser.add_argument(
        "--progress_bar", type=parse_bool, const=True, nargs="?", default=False
    )
    parser.add_argument(
        "--resume",
        type=parse_bool,
        const=True,
        default=False,
        nargs="?",
        help="Whether to resume from the most recent saved model in the current log_dir.",
    )

    # Weights & Biases
    parser.add_argument(
        "--use_wandb", type=parse_bool, const=True, nargs="?", default=False
    )
    parser.add_argument(
        "--wandb_api_key_path",
        type=str,
        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.",
    )
    parser.add_argument(
        "--wandb_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for wandb.init() passed as key1=value1 key2=value2",
    )

    return parser


def main(config):
    # config = parser.parse_args()
    config = populate_defaults(config)

    # Initialize logs
    if os.path.exists(config.log_dir) and config.resume:
        resume = True
        mode = "a"
    elif os.path.exists(config.log_dir) and config.eval_only:
        resume = False
        mode = "a"
    else:
        resume = False
        mode = "w"

    if config.simulate_label_shift and config.use_target:
        config.log_dir = f"{config.log_dir}/{config.dataset}_split:{config.target_split}_alpha:{config.dirichlet_alpha}_seed:{config.seed}/{config.algorithm}_{config.estimation_method}_source_bal:{config.source_balanced}_pretrained:{config.pretrain_type}/lr:{config.lr}_wd:{config.weight_decay}_bs:{config.batch_size}_opt:{config.optimizer}/"
    elif config.use_target:
        config.log_dir = f"{config.log_dir}/{config.dataset}_split:{config.target_split}_seed:{config.seed}/{config.algorithm}_{config.estimation_method}_source_bal:{config.source_balanced}_pretrained:{config.pretrain_type}/lr:{config.lr}_wd:{config.weight_decay}_bs:{config.batch_size}_opt:{config.optimizer}/"
    else:
        config.log_dir = f"{config.log_dir}/{config.dataset}_seed:{config.seed}/{config.algorithm}_pretrained:{config.pretrain_type}/lr:{config.lr}_wd:{config.weight_decay}_bs:{config.batch_size}_opt:{config.optimizer}/"

    if os.path.exists(f"{config.log_dir}/finish.txt"):
        logger.info("The run already existed before ....")
        sys.exit()

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

        logger.info("Logging directory created .... ")
    else:
        logger.info("Logging directory already exists .... ")

    # Set up logging
    fileHandler = logging.FileHandler(
        "{0}/{1}".format(config.log_dir, "run.log"), mode=mode
    )

    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    # Set device
    if torch.cuda.is_available():

        device_count = torch.cuda.device_count()
        if len(config.device) > device_count:
            raise ValueError(
                f"Specified {len(config.device)} devices, but only {device_count} devices found."
            )

        config.use_data_parallel = len(config.device) > 1
        device_str = ",".join(map(str, config.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        config.device = torch.device("cuda")
    else:
        config.use_data_parallel = False
        config.device = torch.device("cpu")

    # Record config
    logger.info("Config:")
    log_config(config, logger)

    # Set random seed
    set_seed(config.seed)

    # Transforms
    data_transforms = {}

    if config.use_source:
        logger.info("Loading source  transforms...")
        data_transforms["source_train"] = initialize_transform(
            transform_name=config.transform,
            config=config,
            additional_transform_name=config.additional_train_transform,
            is_training=True,
            model_name=config.model,
        )

        data_transforms["source_test"] = initialize_transform(
            transform_name=config.transform,
            config=config,
            is_training=False,
            model_name=config.model,
        )

        logger.info("Done loading source transforms.")

    if config.use_target:
        logger.info("Loading target transforms...")

        if "FixMatch" in config.algorithm:
            # import pdb; pdb.set_trace()
            data_transforms["target_train"] = initialize_transform(
                transform_name=config.transform,
                config=config,
                additional_transform_name="fixmatch",
                is_training=True,
                model_name=config.model,
            )

        elif "SENTRY" in config.algorithm:
            data_transforms["target_train"] = initialize_transform(
                transform_name=config.transform,
                config=config,
                additional_transform_name="sentry",
                is_training=True,
                model_name=config.model,
            )

        else:
            data_transforms["target_train"] = initialize_transform(
                transform_name=config.transform,
                config=config,
                additional_transform_name=config.additional_train_transform,
                is_training=True,
                model_name=config.model,
            )

        data_transforms["target_test"] = initialize_transform(
            transform_name=config.transform,
            config=config,
            is_training=False,
            model_name=config.model,
        )

        logger.info("Done loading target transforms.")

    # Data
    logger.info("Loading data...")

    full_dataset = get_dataset(
        dataset=config.dataset,
        source=config.use_source,
        target=config.use_target,
        root_dir=config.root_dir,
        target_split=config.target_split,
        transforms=data_transforms,
        num_classes=config.num_classes,
        split_fraction=config.split_fraction,
        seed=config.seed,
    )

    # import pdb; pdb.set_trace()

    logger.info("Done loading data.")

    simulated_label_shift = False
    if config.simulate_label_shift and config.use_target:
        logger.info("Simulating label shift...")

        # import pdb; pdb.set_trace()
        resample = True
        if config.dirichlet_alpha > 0.001 and config.dirichlet_alpha < 50:
            target_ydist = calculate_marginal(
                np.array(full_dataset["target_train"].y_array), config.num_classes
            )
            target_label_dist = get_dirichlet_marginal(
                config.dirichlet_alpha * config.num_classes * target_ydist, config.seed
            )
        elif config.dirichlet_alpha > 50.0:
            target_ydist = calculate_marginal(
                np.array(full_dataset["target_train"].y_array), config.num_classes
            )
            target_label_dist = target_ydist
            resample = False
        else:
            source_ydist = calculate_marginal(
                np.array(full_dataset["source_train"].y_array), config.num_classes
            )
            target_label_dist = source_ydist

        # target_label_dist = np.random.dirichlet([config.dirichlet_alpha] * config.num_classes)
        logger.info(f"Target label marginal {target_label_dist}")
        logger.info(
            f"Target train size before simulating shift: {len(full_dataset['target_train'])}"
        )
        if resample:
            # logger.debug(f"Target train array {full_dataset['target_train'].y_array} and target test array {full_dataset['target_test'].y_array}")

            if config.with_replacement:
                target_train_idx = tweak_dist_idx(
                    np.array(full_dataset["target_train"].y_array),
                    config.num_classes,
                    len(full_dataset["target_train"].y_array),
                    target_label_dist,
                    seed=config.seed,
                )
                target_test_idx = tweak_dist_idx(
                    np.array(full_dataset["target_test"].y_array),
                    config.num_classes,
                    len(full_dataset["target_test"].y_array),
                    target_label_dist,
                    seed=config.seed,
                )
            else:
                target_train_idx = get_resampled_indices(
                    np.array(full_dataset["target_train"].y_array),
                    config.num_classes,
                    target_label_dist,
                    seed=config.seed,
                )
                target_test_idx = get_resampled_indices(
                    np.array(full_dataset["target_test"].y_array),
                    config.num_classes,
                    target_label_dist,
                    seed=config.seed,
                )

            # logger.debug(f"Train idx {target_train_idx}")
            # logger.debug(f"Test idx {target_test_idx}")

            # import pdb; pdb.set_trace()

            full_dataset["target_train"] = Subset(
                full_dataset["target_train"], target_train_idx
            )
            full_dataset["target_test"] = Subset(
                full_dataset["target_test"], target_test_idx
            )
            simulated_label_shift = True

        logger.info(
            f"Target train size after simulating shift: {len(full_dataset['target_train'])}"
        )

        logger.info("Done simulating label shift.")

    logger.info("Loading collate function.")

    collate_function = initialize_collate_function(config.collate_function)

    logger.info("Done loading collate function.")

    dataloaders = {}

    if config.use_source:
        logger.info("Loading source dataloaders...")

        dataloaders["source_train"] = DataLoader(
            full_dataset["source_train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_function,
        )

        dataloaders["source_test"] = DataLoader(
            full_dataset["source_test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_function,
        )

        logger.info("Done loading source dataloaders.")

    if config.use_target:

        logger.info("Loading target dataloaders...")

        dataloaders["target_train"] = DataLoader(
            full_dataset["target_train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_function,
        )

        dataloaders["target_test"] = DataLoader(
            full_dataset["target_test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_function,
        )

        logger.info("Done loading target dataloaders.")

    # import pdb; pdb.set_trace()
    # Loggers
    results_logger = ResultsLogger(
        os.path.join(config.log_dir, "eval.csv"), mode=mode, use_wandb=config.use_wandb
    )

    if config.use_wandb:
        initialize_wandb(config)

    logger.info("Initialize the algorithm...")

    # Initialize algorithm & load pretrained weights if provided

    if config.eval_only:
        config.pretrained = False

    if "NoisyStudent" not in config.algorithm:
        algorithm = initialize_algorithm(
            config=config,
            datasets=full_dataset,
            dataloader=dataloaders,
        )

    # TODO: For future, add support to balance source and target dataloaders
    # based on the true and estimated label distributions. This is specifically
    # useful for SENTRY, COAL and other algorithms that by default train on
    # balanced source and target datasets. One caveat is that this is only
    # possible when target dataset is of sufficient size and has non-severe
    # label imbalance (or missing labels).

    if not config.eval_only:

        if config.test_time_adapt:

            logger.info("Testing time adapting ...")
            if config.use_source_model:

                adapt(
                    algorithm=algorithm,
                    config=config,
                    dataloaders=dataloaders,
                    results_logger=results_logger,
                    datasets=full_dataset,
                )
            else:
                raise ValueError(
                    "Test time adaptation is not supported without source models."
                )
        else:

            if "NoisyStudent" in config.algorithm:

                logger.info("Loading teacher model ...")

                teacher_model = initialize_model(
                    model_name=config.model,
                    dataset_name=config.dataset,
                    num_classes=config.num_classes,
                    featurize=False,
                    pretrained=False,
                )

                # load(teacher_model, config.noisystudent_kwargs["teacher_model_path"], device=config.device)
                load(
                    teacher_model,
                    f"./model_ckpt/{config.seed}/{config.dataset}.pth",
                    device=config.device,
                )

                teacher_model.to(config.device)

                logger.info("Done.")

                noisy_iterations = config.noisystudent_kwargs["iterations"]

                for iteration in range(noisy_iterations):

                    logger.info(f"Starting iteration {iteration} for Noisy Student...")

                    noisy_data_transforms = initialize_transform(
                        transform_name=config.transform,
                        config=config,
                        additional_transform_name="weak",
                        is_training=True,
                        model_name=config.model,
                    )

                    # import pdb; pdb.set_trace()

                    if simulated_label_shift:
                        assert (
                            full_dataset["target_train"].dataset.transform is not None
                        ), "Target train transform is None when trying to replace with weak transform."
                        weak_target_train = copy.deepcopy(full_dataset["target_train"])
                        weak_target_train.dataset.transform = noisy_data_transforms

                    else:
                        assert (
                            full_dataset["target_train"].transform is not None
                        ), "Target train transform is None when trying to replace with weak transform."
                        weak_target_train = copy.deepcopy(full_dataset["target_train"])
                        weak_target_train.transform = noisy_data_transforms

                    noisy_dataloader = DataLoader(
                        weak_target_train,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=config.num_workers,
                        pin_memory=True,
                    )

                    target_predictions = infer_predictions(
                        teacher_model, noisy_dataloader, config
                    )

                    target_trainset = DatasetwithPseudoLabels(
                        full_dataset["target_train"], target_predictions
                    )

                    if "IS-NoisyStudent" in config.algorithm:
                        # import pdb; pdb.set_trace()
                        dataloaders["source_train"] = rebalance_loader(
                            full_dataset["source_train"], config, use_true_target=True
                        )
                        dataloaders["target_train"] = rebalance_loader(
                            target_trainset, config, use_true_target=True
                        )

                    else:
                        dataloaders["target_train"] = DataLoader(
                            target_trainset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.num_workers,
                            pin_memory=True,
                            collate_fn=collate_function,
                        )

                    logger.info(
                        f"Initialization the model to prepare for iteration {iteration} training ..."
                    )

                    algorithm = initialize_algorithm(
                        config=config, datasets=full_dataset, dataloader=dataloaders
                    )

                    logger.info("Done.")

                    # import pdb; pdb.set_trace()
                    train(
                        algorithm=algorithm,
                        config=config,
                        dataloaders=dataloaders,
                        results_logger=results_logger,
                        epoch_offset=0,
                    )

                    teacher_model = algorithm.model

                    logger.info(f"Done iteration {iteration} for Noisy Student.")

            else:

                # Resume from most recent model in log_dir
                resume_success = False
                if resume:
                    save_path = f"{config.log_dir}/epoch:last_model.pth"
                    try:
                        prev_epoch = load(algorithm, save_path, device=config.device)
                        epoch_offset = prev_epoch + 1
                        logger.info(f"Resuming from epoch {epoch_offset}...")
                        resume_success = True
                    except FileNotFoundError:
                        logger.info("Model loading failed.")
                        pass

                if resume_success == False:
                    epoch_offset = 0
                    logger.info("Starting from epoch 0...")

                train(
                    algorithm=algorithm,
                    dataloaders=dataloaders,
                    results_logger=results_logger,
                    config=config,
                    epoch_offset=epoch_offset,
                    datasets=full_dataset,
                )
    else:

        logger.info(f"Evaluating models from {config.source_model_path}")

        eval_models(
            algorithm=algorithm,
            dataloaders=dataloaders,
            results_logger=results_logger,
            config=config,
        )

    with open(f"{config.log_dir}/finish.txt", "w") as f:
        f.write("Done")

    if config.use_wandb:
        wandb.finish()

    results_logger.close()

    return config.log_dir


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    main(args)

import logging

from RLSbench.algorithms.BN_adapt import BN_adapt
from RLSbench.algorithms.BN_adapt_adv import BN_adapt_adv
from RLSbench.algorithms.CDAN import CDAN
from RLSbench.algorithms.COAL import COAL
from RLSbench.algorithms.CORAL import CORAL
from RLSbench.algorithms.DANN import DANN
from RLSbench.algorithms.ERM import ERM
from RLSbench.algorithms.ERM_Adv import ERM_Adv
from RLSbench.algorithms.fixmatch import FixMatch
from RLSbench.algorithms.noisy_student import NoisyStudent
from RLSbench.algorithms.pseudolabel import PseudoLabel
from RLSbench.algorithms.SENTRY import SENTRY
from RLSbench.algorithms.TENT import TENT

logger = logging.getLogger("label_shift")


def initialize_algorithm(config, datasets, dataloader):
    logger.info(f"Initializing algorithm {config.algorithm} ...")

    source_dataset = datasets["source_train"]
    trainloader_source = dataloader["source_train"]

    # Other config
    n_train_steps = (
        len(trainloader_source) * config.n_epochs // config.gradient_accumulation_steps
    )

    if config.algorithm in (
        "ERM-rand",
        "ERM-imagenet",
        "ERM-clip",
        "ERM-bert",
        "ERM-aug-rand",
        "ERM-aug-imagenet",
        "ERM-swav",
        "ERM-oracle-rand",
        "ERM-oracle-imagenet",
        "IS-ERM-rand",
        "IS-ERM-imagenet",
        "IS-ERM-clip",
        "IS-ERM-aug-rand",
        "IS-ERM-aug-imagenet",
        "IS-ERM-swav",
        "IS-ERM-oracle-rand",
        "IS-ERM-oracle-imagenet",
    ):
        algorithm = ERM(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
        )

    elif config.algorithm in ("ERM-adv"):
        algorithm = ERM_Adv(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
        )

    elif config.algorithm in ("DANN", "IW-DANN", "IS-DANN"):
        algorithm = DANN(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            n_domains=2,
            **config.dann_kwargs,
        )

    elif config.algorithm in ("CDANN", "IW-CDANN", "IS-CDANN"):
        algorithm = CDAN(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            n_domains=2,
            **config.cdan_kwargs,
        )

    elif config.algorithm in ("FixMatch", "IS-FixMatch"):
        algorithm = FixMatch(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.fixmatch_kwargs,
        )

    elif config.algorithm in ("PseudoLabel", "IS-PseudoLabel"):
        algorithm = PseudoLabel(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.pseudolabel_kwargs,
        )

    elif config.algorithm in ("NoisyStudent", "IS-NoisyStudent"):
        algorithm = NoisyStudent(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.noisystudent_kwargs,
        )

    elif config.algorithm in ("COAL", "IW-COAL"):
        algorithm = COAL(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.coal_kwargs,
        )

    elif config.algorithm in ("SENTRY", "IW-SENTRY"):
        algorithm = SENTRY(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.sentry_kwargs,
        )

    elif config.algorithm in ("CORAL", "IS-CORAL"):
        algorithm = CORAL(config=config)

    elif config.algorithm in ("BN_adapt", "IS-BN_adapt"):
        algorithm = BN_adapt(config=config)

    elif config.algorithm in ("BN_adapt-adv", "IS-BN_adapt-adv"):
        algorithm = BN_adapt_adv(config=config)

    elif config.algorithm in ("TENT", "IS-TENT"):
        algorithm = TENT(config=config)

    else:
        raise ValueError(f"Algorithm {config.algorithm} not recognized")

    return algorithm

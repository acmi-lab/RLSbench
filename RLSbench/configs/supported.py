import itertools

# See algorithms/initializer.py
algorithms = [
    "ERM",
    "IS-ERM",
    "ERM-aug",
    "IS-ERM-aug",
    "ERM-oracle",
    "IS-ERM-oracle",
    "ERM-adv",
    "DANN",
    "CDANN",
    "IW-DANN",
    "IW-CDANN",
    "IS-DANN",
    "IS-CDANN",
    "COAL",
    "IW-COAL",
    "SENTRY",
    "IW-SENTRY",
    "FixMatch",
    "IW-FixMatch",
    "IS-FixMatch",
    "PseudoLabel",
    "IS-PseudoLabel",
    "NoisyStudent",
    "IS-NoisyStudent",
    "CORAL",
    "IS-CORAL",
    "BN_adapt",
    "BN_adapt-adv",
    "IS-BN_adapt",
    "IS-BN_adapt-adv",
    "TENT",
    "IS-TENT",
]

label_shift_adapt = ["MLLS", "true", "RLLS", "None", "baseline"]


# See transforms.py
transforms = [
    "image_base",
    "image_resize_and_center_crop",
    "image_none",
    "rxrx1",
    "clip",
    "bert",
    "None",
]

additional_transforms = [
    "randaugment",
    "weak",
]
collate_functions = ["mimic_readmission", "mimic_mortality", "None"]


# See models/initializer.py
# We register all models that are added to any of the following lists of architectures
from RLSbench.models.initializer import (
    CIFAR_ARCHITECTURES,
    CLIP_ARCHITECTURES,
    TORCHVISION_ARCHITECTURES,
    OTHER_ARCHITECTURES,
)

# Preserve order but only keep unique model names
models = list(
    dict.fromkeys(
        itertools.chain(
            CIFAR_ARCHITECTURES,
            CLIP_ARCHITECTURES,
            TORCHVISION_ARCHITECTURES,
            OTHER_ARCHITECTURES,
        )
    )
)

# Pre-training type
pretrainining_options = ["clip", "imagenet", "swav", "rand", "bert"]

# See optimizer.py
optimizers = ["SGD", "Adam", "AdamW"]

# See scheduler.py
schedulers = [
    "linear_schedule_with_warmup",
    "cosine_schedule_with_warmup",
    "ReduceLROnPlateau",
    "StepLR",
    "FixMatchLR",
    "MultiStepLR",
]

# See losses.py
losses = ["cross_entropy", "cross_entropy_logits"]

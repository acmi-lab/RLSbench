import copy
from typing import List

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

from RLSbench.data_augmentation.randaugment import (
    FIX_MATCH_AUGMENTATION_POOL,
    RandAugment,
)

from transformers import BertTokenizerFast, DistilBertTokenizerFast


_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]


def initialize_transform(
    transform_name,
    config,
    is_training,
    additional_transform_name=None,
    model_name=None,
):
    """
    By default, transforms should take in `x` and return `transformed_x`.
    For transforms that take in `(x, y)` and return `(transformed_x, transformed_y)`,
    set `do_transform_y` to True when initializing the WILDSSubset.
    """
    if transform_name is None:
        return None
    elif transform_name == "None" or transform_name == "none":
        return None
    elif transform_name == "to_tensor":
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    elif transform_name == "rxrx1":
        return initialize_rxrx1_transform(is_training)
    elif transform_name == "clip":
        return initialize_clip_transform(is_training)
    elif transform_name == "bert":
        return initialize_bert_transform(model_name, config.max_token_length)

    if transform_name == "image_base":
        transform_steps = get_image_base_transform_steps(config)

    elif transform_name == "image_resize_and_center_crop":
        transform_steps = get_image_resize_and_center_crop_transform_steps(config)
    elif transform_name == "image_none":
        transform_steps = []
    else:
        raise ValueError(f"{transform_name} not recognized")

    if config.default_normalization:
        default_normalization = transforms.Normalize(
            _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
            _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
        )
    else:
        default_normalization = transforms.Normalize(
            config.mean,
            config.std,
        )
    if additional_transform_name == "fixmatch":
        transformations = add_fixmatch_transform(
            config, transform_steps, default_normalization
        )
        transform = MultipleTransforms(transformations)
    elif additional_transform_name == "sentry":
        transformations = add_sentry_transform(
            config, transform_steps, default_normalization
        )
        transform = MultipleTransforms(transformations)

    elif additional_transform_name == "randaugment":
        transform = add_rand_augment_transform(
            config, transform_steps, default_normalization
        )
    elif additional_transform_name == "weak":
        transform = add_weak_transform(config, transform_steps, default_normalization)
    else:
        target_resolution = _get_target_resolution(config)
        transform_steps.append(
            transforms.CenterCrop((target_resolution, target_resolution)),
        )
        transform_steps.append(transforms.ToTensor())

        if "adv" not in config.algorithm:
            transform_steps.append(default_normalization)

        transform = transforms.Compose(transform_steps)

    return transform


def initialize_rxrx1_transform(is_training):
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.0] = 1.0
        return TF.normalize(x, mean, std)

    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]

    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x

    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform


def get_image_base_transform_steps(config) -> List:
    transform_steps = []

    # if config.resize_resolution is not None:
    #     crop_size = config.resize_resolution
    #     transform_steps.append(transforms.CenterCrop(crop_size))

    if config.resize_resolution is not None:
        transform_steps.append(transforms.Resize(config.resize_resolution))

    return transform_steps


def get_image_resize_and_center_crop_transform_steps(config) -> List:
    """
    Resizes the image to a slightly larger square then crops the center.
    """
    transform_steps = get_image_base_transform_steps(config)
    target_resolution = _get_target_resolution(config)
    transform_steps.append(
        transforms.CenterCrop((target_resolution, target_resolution)),
    )
    return transform_steps


def add_fixmatch_transform(config, base_transform_steps, normalization):
    return (
        add_weak_transform(config, base_transform_steps, normalization),
        add_rand_augment_transform(config, base_transform_steps, normalization),
    )


def add_sentry_transform(config, base_transform_steps, normalization):
    return (
        add_weak_transform(config, base_transform_steps, normalization),
        add_rand_augment_transform(config, base_transform_steps, normalization),
        add_rand_augment_transform(config, base_transform_steps, normalization),
        add_rand_augment_transform(config, base_transform_steps, normalization),
    )


def add_weak_transform(config, base_transform_steps, normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config)
    weak_transform_steps = copy.deepcopy(base_transform_steps)
    if "cifar" in config.dataset:
        weak_transform_steps.extend(
            [
                transforms.RandomCrop(size=target_resolution, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        weak_transform_steps.extend(
            [
                transforms.RandomCrop(
                    size=target_resolution,
                ),
                transforms.RandomHorizontalFlip(),
            ]
        )
    weak_transform_steps.append(transforms.ToTensor())
    weak_transform_steps.append(normalization)

    return transforms.Compose(weak_transform_steps)


def add_rand_augment_transform(config, base_transform_steps, normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config)
    strong_transform_steps = copy.deepcopy(base_transform_steps)
    strong_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=target_resolution),
            RandAugment(
                n=config.randaugment_n,
                augmentation_pool=FIX_MATCH_AUGMENTATION_POOL,
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return transforms.Compose(strong_transform_steps)


def _get_target_resolution(config):
    return config.target_resolution


class MultipleTransforms(object):
    """When multiple transformations of the same data need to be returned."""

    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x):
        return tuple(transform(x) for transform in self.transformations)


def initialize_clip_transform(is_training):
    from torchvision.transforms import (
        Compose,
        Resize,
        CenterCrop,
        ToTensor,
        Normalize,
        RandomResizedCrop,
    )

    def _convert_to_rgb(image):
        return image.convert("RGB")

    normalize = Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    if is_training:
        return Compose(
            [
                RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )
    else:
        return Compose(
            [
                Resize(224, interpolation=Image.BICUBIC, max_size=None, antialias=None),
                CenterCrop(224),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )


def getBertTokenizer(model):
    if model == "bert-base-uncased":
        tokenizer = BertTokenizerFast.from_pretrained(model)
    elif model == "distilbert-base-uncased":
        tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    else:
        raise ValueError(f"Model: {model} not recognized.")

    return tokenizer


def initialize_bert_transform(net, max_token_length=512):
    # assert 'bert' in config.model
    # assert config.max_token_length is not None

    tokenizer = getBertTokenizer(net)

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_token_length,
            return_tensors="pt",
        )
        if net == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif net == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform

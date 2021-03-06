import kvt.callbacks
import kvt.collate_fns
import kvt.datasets
import kvt.hooks
import kvt.lightning_modules
import kvt.losses
import kvt.metrics
import kvt.models.backbones
import kvt.models.segmentations
import kvt.models.sound_event_detections
import kvt.optimizers
import kvt.samplers
import kvt.transforms
import lightly
import pretrainedmodels
import pytorch_lightning as pl
import resnest.torch
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
import torchvision.models
import transformers
from kvt.registry import (
    BACKBONES,
    CALLBACKS,
    COLLATE_FNS,
    DATASETS,
    HOOKS,
    LIGHTNING_MODULES,
    LOSSES,
    METRICS,
    MODELS,
    OPTIMIZERS,
    SAMPLERS,
    SCHEDULERS,
    TRANSFORMS,
)

try:
    import torch_optimizer
except ImportError:
    torch_optimizer = None


def register_torch_modules():
    # register backbones
    for name, cls in kvt.models.backbones.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # torchvision models
    for name, cls in torchvision.models.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # resnest
    for name, cls in resnest.torch.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # pretrained models
    for name, cls in pretrainedmodels.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # timm
    for name in timm.list_models():
        cls = timm.model_entrypoint(name)
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # segmentation models
    for name, cls in kvt.models.segmentations.__dict__.items():
        if not callable(cls):
            continue
        MODELS.register(cls)

    # sound event detection models
    for name, cls in kvt.models.sound_event_detections.__dict__.items():
        if not callable(cls):
            continue
        MODELS.register(cls)

    # lightly models
    for name, cls in lightly.models.__dict__.items():
        if not callable(cls):
            continue
        MODELS.register(cls)

    # register losses
    for name, cls in nn.__dict__.items():
        if not callable(cls):
            continue
        if "Loss" in name:
            LOSSES.register(cls)

    for name, cls in kvt.losses.__dict__.items():
        if not callable(cls):
            continue
        LOSSES.register(cls)

    for name, cls in lightly.loss.__dict__.items():
        if not callable(cls):
            continue
        LOSSES.register(cls)

    # register optimizers
    for name, cls in optim.__dict__.items():
        if not callable(cls):
            continue
        OPTIMIZERS.register(cls)

    for name, cls in kvt.optimizers.__dict__.items():
        if not callable(cls):
            continue
        OPTIMIZERS.register(cls)

    if torch_optimizer is not None:
        for name, cls in torch_optimizer.__dict__.items():
            if not callable(cls):
                continue
            if hasattr(cls, "__name__"):
                OPTIMIZERS.register(cls)

    # from transformers
    OPTIMIZERS.register(transformers.AdamW)
    OPTIMIZERS.register(transformers.Adafactor)

    # register schedulers
    for name, cls in optim.lr_scheduler.__dict__.items():
        if not callable(cls):
            continue
        SCHEDULERS.register(cls)

    # from transformers
    SCHEDULERS.register(transformers.get_constant_schedule)
    SCHEDULERS.register(transformers.get_constant_schedule_with_warmup)
    SCHEDULERS.register(transformers.get_cosine_schedule_with_warmup)
    SCHEDULERS.register(
        transformers.get_cosine_with_hard_restarts_schedule_with_warmup
    )
    SCHEDULERS.register(transformers.get_linear_schedule_with_warmup)
    SCHEDULERS.register(transformers.get_polynomial_decay_schedule_with_warmup)

    # register lightning module
    for name, cls in kvt.lightning_modules.__dict__.items():
        if not callable(cls):
            continue
        LIGHTNING_MODULES.register(cls)

    # register datasets
    for name, cls in kvt.datasets.__dict__.items():
        if not callable(cls):
            continue
        DATASETS.register(cls)

    for name, cls in lightly.data.__dict__.items():
        if not callable(cls):
            continue
        DATASETS.register(cls)

    # register hooks
    for name, cls in kvt.hooks.__dict__.items():
        if not callable(cls):
            continue
        HOOKS.register(cls)

    # register metrics
    for name, cls in kvt.metrics.__dict__.items():
        if not callable(cls):
            continue
        METRICS.register(cls)

    for name, cls in torchmetrics.__dict__.items():
        if not callable(cls):
            continue
        METRICS.register(cls)

    # register transforms
    for name, cls in kvt.transforms.__dict__.items():
        if not callable(cls):
            continue
        TRANSFORMS.register(cls)

    # register collate_fn
    COLLATE_FNS.register(transformers.DataCollatorForTokenClassification)
    COLLATE_FNS.register(transformers.DataCollatorForSeq2Seq)
    COLLATE_FNS.register(transformers.DataCollatorForLanguageModeling)
    COLLATE_FNS.register(transformers.DataCollatorForWholeWordMask)
    COLLATE_FNS.register(
        transformers.DataCollatorForPermutationLanguageModeling
    )

    for name, cls in lightly.data.collate.__dict__.items():
        if not callable(cls):
            continue
        if hasattr(cls, "__name__"):
            COLLATE_FNS.register(cls)

    for name, cls in kvt.collate_fns.__dict__.items():
        if not callable(cls):
            continue
        COLLATE_FNS.register(cls)

    # register sampler
    for name, cls in kvt.samplers.__dict__.items():
        if not callable(cls):
            continue
        SAMPLERS.register(cls)

    # register callbacks
    for name, cls in kvt.callbacks.__dict__.items():
        if not callable(cls):
            continue
        CALLBACKS.register(cls)


def initialize():
    register_torch_modules()

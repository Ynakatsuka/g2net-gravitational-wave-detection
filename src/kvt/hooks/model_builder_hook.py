import abc
import copy
import inspect
import os

import kvt.registry
import lightly
import torch
import torch.nn as nn
from kvt.models.layers import MixLinear
from kvt.registry import BACKBONES, MODELS
from kvt.utils import (
    analyze_in_features,
    build_from_config,
    replace_last_linear,
    update_input_layer,
)


class ModelBuilderHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, config):
        pass


class DefaultModelBuilderHook(ModelBuilderHookBase):
    def __call__(self, config):
        lightly_models = [
            name
            for name, cls in lightly.models.__dict__.items()
            if inspect.isclass(cls)
        ]

        # classification models
        if BACKBONES.get(config.name) is not None:
            model = self.build_classification_model(config)
        # lightly models
        elif config.name in lightly_models:
            model = self.build_backbone_wrapping_model(
                config, filter_keys_of_args=["backbone"]
            )
        # UNet
        elif "UNet" in config.name:
            model = build_from_config(config, MODELS)
        # sound event detection models and others
        else:
            model = self.build_backbone_wrapping_model(config)

        # load pretrained model trained on external data
        if hasattr(config.params, "pretrained") and isinstance(
            config.params.pretrained, str
        ):
            path = config.params.pretrained
            print(f"[Loaded pretrained model] {path}")
            if os.path.exists(path):
                loaded_object = torch.load(path)
                if "state_dict" in loaded_object.keys():
                    state_dict = loaded_object["state_dict"]
                else:
                    state_dict = loaded_object
            else:
                state_dict = torch.hub.load_state_dict_from_url(path, progress=True)[
                    "state_dict"
                ]

            # fix state_dict: local model trained on dp
            state_dict = kvt.utils.fix_dp_model_state_dict(state_dict)

            # fix state_dict: SSL
            if hasattr(config, "fix_state_dict"):
                if config.fix_state_dict == "mocov2":
                    state_dict = kvt.utils.fix_mocov2_state_dict(state_dict)
                elif config.fix_state_dict == "transformers":
                    state_dict = kvt.utils.fix_transformers_state_dict(state_dict)
                else:
                    raise KeyError

            model = kvt.utils.load_state_dict_on_same_size(
                model, state_dict, infer_key=True
            )

        # mixout
        if hasattr(config, "mixout") and (config.mixout > 0):
            for sup_module in model.modules():
                for name, module in sup_module.named_children():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0
                    if isinstance(module, nn.Linear):
                        target_state_dict = module.state_dict()
                        bias = True if module.bias is not None else False
                        new_module = MixLinear(
                            module.in_features,
                            module.out_features,
                            bias,
                            target_state_dict["weight"],
                            config.mixout,
                        )
                        new_module.load_state_dict(target_state_dict)
                        setattr(sup_module, name, new_module)

        return model

    def build_classification_model(self, config):
        force_replace_last_linear = False
        # build model
        if hasattr(config.params, "pretrained") and config.params.pretrained:
            pretrained_config = copy.deepcopy(config)
            if isinstance(config.params.pretrained, bool):
                # if pretrained is True and num_classes is not 1000,
                # loading pretraining model fails
                # To avoid this issue, load as default num_classes
                try:
                    model = build_from_config(pretrained_config, BACKBONES)
                except RuntimeError:
                    if hasattr(pretrained_config["params"], "num_classes"):
                        del pretrained_config["params"]["num_classes"]
                    model = build_from_config(pretrained_config, BACKBONES)
                    force_replace_last_linear = True
            else:
                # when pretrained is passed as local path
                pretrained_config.params.pretrained = False
                model = build_from_config(pretrained_config, BACKBONES)
        else:
            model = build_from_config(config, BACKBONES)

        # replace last linear
        if hasattr(config, "last_linear") and config.last_linear.replace:
            model = replace_last_linear(
                model, config.params.num_classes, **config.last_linear.params
            )
        elif force_replace_last_linear:
            model = replace_last_linear(model, config.params.num_classes)

        return model

    def build_backbone_wrapping_model(
        self, config, filter_keys_of_args=None,
    ):
        # build model
        backbone_config = {"name": config.params.backbone.name}
        params = config.params.backbone.params

        backbone = build_from_config(
            backbone_config, BACKBONES, params, match_object_args=True
        )
        if ("in_chans" in params.keys()) and (params["in_chans"] != 3):
            backbone = update_input_layer(backbone, params["in_chans"])

        in_features = analyze_in_features(backbone)

        if config.params.backbone.name == "resnest50":
            layers = list(backbone.children())[:-2]
            backbone = nn.Sequential(*layers)
        elif hasattr(config, "last_linear") and config.last_linear.replace:
            backbone = replace_last_linear(
                backbone, config.params.num_classes, **config.last_linear.params
            )
        else:
            backbone = replace_last_linear(backbone, use_identity_as_last_layer=True)

        args = {"backbone": backbone, "in_features": in_features}
        if filter_keys_of_args is not None:
            args = {k: v for k, v in args.items() if k in filter_keys_of_args}

        # auto detecting "num_ftrs" for lightly models
        if ("num_ftrs" in config.params.keys()) and (
            config.params.num_ftrs != in_features
        ):
            print(f"[Replace num_ftrs] {config.params.num_ftrs} -> {in_features}")
            config.params.num_ftrs = in_features

        model = build_from_config(config, MODELS, default_args=args)

        return model

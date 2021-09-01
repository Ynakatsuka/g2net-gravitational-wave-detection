import torch
import torch.nn as nn

from kvt.models.heads import (AdaCos, AddMarginProduct, ArcMarginProduct,
                              CurricularFace, MultiSampleDropout,
                              SphereProduct)
from kvt.models.layers import Flatten, Identity, SEBlock
from kvt.models.necks import (RMAC, AdaptiveConcatPool2d, GeM, Rpool,
                              TripletAttention)
from kvt.models.wrappers import MetricLearningModelWrapper


def analyze_in_features(model):
    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
    elif hasattr(model, "classif"):
        in_features = model.classif.in_features
    elif hasattr(model, "fc"):
        if isinstance(model.fc, nn.Sequential):
            in_features = model.fc[-1].in_features
        else:
            in_features = model.fc.in_features
    elif hasattr(model, "last_linear"):
        in_features = model.last_linear.in_features
    elif hasattr(model, "head"):
        if hasattr(model.head, "fc"):
            if hasattr(model.head.fc, "in_features"):
                in_features = model.head.fc.in_features
            else:
                in_features = model.head.fc.in_channels
        else:
            in_features = model.head.in_features
    else:
        raise ValueError(f"Model has no last linear layer: {model}")

    return in_features


def replace_last_linear(
    model,
    num_classes=1,
    pool_type=None,
    dropout_rate=0,
    use_seblock=False,
    use_identity_as_last_layer=False,
    last_linear_type="linear",
    n_last_linear_layers=0,
    multi_sample_dropout_p=0.5,
    n_multi_samples=5,
    # neck
    apply_triplet_attention=False,
    gem_p=3,
    # for metric learing
    s=64.0,
    m=0.5,
    # for multi head
    num_classes_list=None,
    head_names=None,
    **kwargs,
):
    if not last_linear_type in (
        "linear",
        "multi_sample_dropout",
        "arcface",
        "cosface",
        "adacos",
        "sphereface",
        "curricularface",
    ):
        raise ValueError(f"Invalid last linear type: {last_linear_type}")

    # replace pooling
    def replace_pooling_layer(original, layer_name, apply_triplet_attention=False):
        neck = []

        if apply_triplet_attention:
            neck.append(TripletAttention())

        fc_input_shape_ratio = 1
        if pool_type == "concat":
            neck.append(AdaptiveConcatPool2d())
            fc_input_shape_ratio = 2
        elif pool_type == "avg":
            neck.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif pool_type == "adaptive_avg":
            neck.append(nn.AdaptiveAvgPool2d((10, 10)))
            fc_input_shape_ratio = 100
        elif pool_type == "gem":
            neck.append(GeM(p=gem_p))
        elif pool_type == "identity":
            neck.append(Identity())
        elif pool_type == "rmac":
            neck.append(RMAC())
        elif pool_type == "rpool":
            neck.append(Rpool())

        neck = nn.Sequential(*neck)
        setattr(original, layer_name, neck)
        print(f"[Replace Neck] {getattr(original, layer_name)}")
        return fc_input_shape_ratio

    for layer_name in ["avgpool", "global_pool"]:
        if hasattr(model, layer_name):
            fc_input_shape_ratio = replace_pooling_layer(
                model, layer_name, apply_triplet_attention
            )
        elif hasattr(model, "head") and hasattr(model.head, layer_name):
            fc_input_shape_ratio = replace_pooling_layer(
                model.head, layer_name, apply_triplet_attention
            )
        else:
            fc_input_shape_ratio = 1

    in_features = analyze_in_features(model)
    in_features *= fc_input_shape_ratio

    # replace fc
    if use_identity_as_last_layer:
        last_layers = Identity()
    else:
        last_layers = [Flatten()]

        if use_seblock:
            last_layers.append(SEBlock(in_features))

        if n_last_linear_layers > 0:
            for _ in range(n_last_linear_layers):
                last_layers.append(
                    nn.Sequential(
                        nn.BatchNorm1d(in_features),
                        nn.Linear(in_features, in_features),
                        nn.Dropout(dropout_rate),
                        nn.ReLU(),
                    )
                )

        if last_linear_type == "multi_sample_dropout":
            last_layers.append(
                MultiSampleDropout(
                    in_features, num_classes, multi_sample_dropout_p, n_multi_samples
                )
            )
        elif (last_linear_type == "multi_head") and (num_classes_list is not None):
            last_layers.append(MultiHead(in_features, num_classes_list, head_names, dropout_rate))
        elif last_linear_type == "linear":
            last_layers.extend([nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes)])

        last_layers = nn.Sequential(*last_layers)

    # replace last layers
    print("-" * 100)
    print("Replace Last Linear: ", last_layers)

    if hasattr(model, "classifier"):
        model.classifier = last_layers
    elif hasattr(model, "fc"):
        model.fc = last_layers
    elif hasattr(model, "last_linear"):
        model.last_linear = last_layers
    elif hasattr(model, "head") and (hasattr(model.head, "fc")):
        model.head.fc = last_layers
    elif hasattr(model, "head"):
        model.head = last_layers

    # for metric learning
    if last_linear_type == "arcface":
        final_layer = ArcMarginProduct(in_features, num_classes, s=s, m=m)
    elif last_linear_type == "cosface":
        final_layer = AddMarginProduct(in_features, num_classes, s=s, m=m)
    elif last_linear_type == "adacos":
        final_layer = AdaCos(in_features, num_classes, m=m)
    elif last_linear_type == "sphereface":
        final_layer = SphereProduct(in_features, num_classes, m=m)
    elif last_linear_type == "curricularface":
        final_layer = CurricularFace(in_features, num_classes, s=s, m=m)
    else:
        final_layer = None

    if final_layer is not None:
        model = MetricLearningModelWrapper(model, final_layer)

    return model


def update_input_layer(model, in_channels):
    for l in model.children():
        if isinstance(l, nn.Sequential):
            for ll in l.children():
                assert ll.bias is None
                data = torch.mean(ll.weight, axis=1).unsqueeze(1)
                data = data.repeat((1, in_channels, 1, 1))
                ll.weight.data = data
                break
        else:
            assert l.bias is None
            data = torch.mean(l.weight, axis=1).unsqueeze(1)
            data = data.repeat((1, in_channels, 1, 1))
            l.weight.data = data
        break
    return model

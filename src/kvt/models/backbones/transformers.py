import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModel, AutoModelForMaskedLM,
                          AutoModelForSequenceClassification)

from kvt.models.heads import MultiSampleDropout
from kvt.models.layers import SEBlock
from kvt.models.necks import gem1d


class FirstTokenPool(nn.Module):
    def forward(self, x):
        return x[:, :, 0]


class BertPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, :, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GlobalMaxPool1d(nn.Module):
    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.size()[-1])


class GlobalAvgPool1d(nn.Module):
    def forward(self, x):
        return F.avg_pool1d(x, kernel_size=x.size()[-1])


class GlobalGemPool1d(nn.Module):
    def forward(self, x):
        return gem1d(x)


def get_pooler(pooler_name, hidden_size):
    if pooler_name == "first_token":
        pooler = FirstTokenPool()
    elif pooler_name == "bert_pool":
        pooler = BertPool(hidden_size)
    elif pooler_name == "max_pool":
        pooler = GlobalMaxPool1d()
    elif pooler_name == "avg_pool":
        pooler = GlobalAvgPool1d()
    elif pooler_name == "gem_pool":
        pooler = GlobalGemPool1d()
    elif pooler_name is None:
        pooler = None
    else:
        raise ValueError(f"Invalid pooler_name: {pooler_name}")
    return pooler


def transformers_for_sequence_classification(model_name, num_classes, **kwargs):
    config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
    print("-" * 100)
    print(f"Model Config: {config}")
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=config)


def transformers_for_masked_language_model(model_name, **kwargs):
    config = AutoConfig.from_pretrained(model_name)
    print("-" * 100)
    print(f"Model Config: {config}")
    return AutoModelForMaskedLM.from_pretrained(model_name, config=config)


class CustomTransformersForSequenceClassification(nn.Module):
    """similar to AutoModelForSequenceClassification,
    but this class can collect last hidden outputs of transformers
    and change pooling layer
    """

    def __init__(
        self,
        model_name,
        num_classes,
        pooler_name=None,
        n_collect_hidden_states=None,
        dropout_rate=0.3,
        use_seblock=False,
        use_multisample_dropout=False,
        multi_sample_dropout_p=0.5,
        n_multi_samples=5,
        reinit_layers=5,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.n_collect_hidden_states = n_collect_hidden_states

        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        if self.n_collect_hidden_states is not None:
            config.update({"output_hidden_states": True})
            hidden_size *= n_collect_hidden_states
        self.config = config
        self.net = AutoModel.from_pretrained(model_name, config=config)
        print("-" * 100)
        print(f"Model Config: {config}")

        # update hidden_size
        if isinstance(pooler_name, list):
            self.bert_pooling = nn.ModuleList(
                [get_pooler(name, hidden_size) for name in pooler_name]
            )
            hidden_size *= len(pooler_name)
        else:
            self.bert_pooling = get_pooler(pooler_name, hidden_size)

        # last_layers = [nn.ReLU()]
        last_layers = [nn.LayerNorm(hidden_size)]

        if use_seblock:
            last_layers.append(SEBlock(hidden_size))

        if use_multisample_dropout:
            last_layers.append(
                MultiSampleDropout(
                    hidden_size, num_classes, multi_sample_dropout_p, n_multi_samples
                )
            )
        else:
            last_layers.extend([nn.Dropout(dropout_rate), nn.Linear(hidden_size, num_classes)])
        self.fc = nn.Sequential(*last_layers)
        self._init_weights(self.fc)

        if reinit_layers > 0:
            print(f"[Reinitializing Last {reinit_layers} Layers]")
            for layer in self.net.encoder.layer[-reinit_layers:]:
                for module in layer.modules():
                    self._init_weights(module)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for mdl in module:
                self._init_weights(mdl)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        encoder_output = self.net(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        if self.bert_pooling is None:
            pooler_output = encoder_output["pooler_output"]
        else:
            if self.n_collect_hidden_states is not None:
                hidden_layers = encoder_output["hidden_states"][-self.n_collect_hidden_states :]
                hidden_states = torch.cat(
                    hidden_layers, dim=-1
                )  # -> [bs, max_length, hidden_states * n_collect_hidden_states]
                hidden_states = hidden_states.transpose(1, 2)  # -> [bs, hidden_states, max_length]
            else:
                hidden_states = encoder_output[
                    "last_hidden_state"
                ]  # -> [bs, max_length, hidden_states]
                hidden_states = hidden_states.transpose(1, 2)  # -> [bs, hidden_states, max_length]

            if isinstance(self.bert_pooling, nn.ModuleList):
                pooler_output = torch.cat(
                    [pooler(hidden_states).squeeze() for pooler in self.bert_pooling],
                    dim=-1,
                )  # -> [bs, hidden_states*len(pooler_name)]
            else:
                pooler_output = self.bert_pooling(hidden_states)  # -> [bs, hidden_states]

        output = self.fc(pooler_output)  # -> [bs, num_classes]
        return output

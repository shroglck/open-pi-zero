"""
Agnostic to mixture setup, e.g., names and block masks

"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.model.kv_cache import KVCache, MixtureKVCache
from src.model.vla_mixture.mixture import Mixture


def forward_mixture_layers(
    mixtures: nn.ModuleList,
    attention_mask: torch.Tensor,
    position_ids_all: dict[torch.LongTensor],
    embeds_all: dict[torch.FloatTensor],
    layer_idx: int,
    is_final_layer: bool = False,
    final_layer_post_attn_skip_names: Optional[List[str]] = ["action"],
    kv_caches: Optional[dict[MixtureKVCache | KVCache]] = None,
    cache_mode: Optional[str] = "fixed",  # or append used in text generation
    time_cond: Optional[torch.FloatTensor] = None,
) -> dict[torch.FloatTensor]:
    names = list(embeds_all.keys())  # e.g., ["vlm", "proprio", "action"]

    residuals_pre_attn = embeds_all
    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_input_norm = {}
    for name in names:
        hidden_states_input_norm[name] = mixtures[name].layer_func(
            "forward_norm",
            layer_idx,
            "input_layernorm",
            embeds_all[name],
            time_cond,
        )  # a bit convoluted
    hidden_states_pre_attn = hidden_states_input_norm

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_post_attn = forward_mixture_attn(
        mixtures,
        hidden_states_all=hidden_states_pre_attn,
        attention_mask=attention_mask,
        position_ids_all=position_ids_all,
        layer_idx=layer_idx,
        is_final_layer=is_final_layer,
        final_layer_post_attn_skip_names=final_layer_post_attn_skip_names,
        kv_caches=kv_caches,
        cache_mode=cache_mode,
    )
    hidden_states_pre_res = hidden_states_post_attn

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_post_res = {}
    for name in names:
        if is_final_layer and name in final_layer_post_attn_skip_names:
            hidden_states_post_res[name] = None
        else:
            hidden_states_pre_res[name] = mixtures[name].layer_func(
                "forward_adaptive_scale",
                layer_idx,
                "post_attn",
                hidden_states_pre_res[name],
                time_cond,
            )
            hidden_states_post_res[name] = (
                residuals_pre_attn[name] + hidden_states_pre_res[name]
            )
    hidden_states_pre_post_attn = hidden_states_post_res

    # [Batch_Size, Seq_Len, Hidden_Size]
    residuals_pre_post_attn = hidden_states_pre_post_attn
    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_post_post_attn = {}
    for name in names:
        if is_final_layer and name in final_layer_post_attn_skip_names:
            hidden_states_post_post_attn[name] = None
        else:
            hidden_states_post_post_attn[name] = mixtures[name].layer_func(
                "forward_norm",
                layer_idx,
                "post_attention_layernorm",
                hidden_states_pre_post_attn[name],
                time_cond,
            )
    hidden_states_pre_mlp = hidden_states_post_post_attn

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_pos_mlp = {}
    for name in names:
        if is_final_layer and name in final_layer_post_attn_skip_names:
            hidden_states_pos_mlp[name] = None
        else:
            hidden_states_pos_mlp[name] = mixtures[name].layer_func(
                "mlp",
                layer_idx,
                hidden_states_pre_mlp[name],
            )
    hidden_states_pre_final_res = hidden_states_pos_mlp

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_final = {}
    for name in names:
        if is_final_layer and name in final_layer_post_attn_skip_names:
            hidden_states_final[name] = None
        else:
            hidden_states_pre_final_res[name] = mixtures[name].layer_func(
                "forward_adaptive_scale",
                layer_idx,
                "final",
                hidden_states_pre_final_res[name],
                time_cond,
            )
            hidden_states_final[name] = (
                residuals_pre_post_attn[name] + hidden_states_pre_final_res[name]
            )
    return hidden_states_final


def forward_mixture_attn(
    mixtures: nn.ModuleList,
    attention_mask: torch.Tensor,
    position_ids_all: dict[torch.LongTensor],
    hidden_states_all: dict[torch.FloatTensor],
    layer_idx: int,
    is_final_layer: bool,
    final_layer_post_attn_skip_names: Optional[List[str]] = ["action"],
    kv_caches: Optional[dict[MixtureKVCache | KVCache]] = None,
    cache_mode: Optional[str] = "fixed",  # or append used in text generation
    attn_softclamp: float = 50.0,
    attention_dropout: float = 0.0,
) -> dict[torch.FloatTensor]:
    """Assume all mixtures have the same head dim"""
    bsz = len(attention_mask)
    q_lens = [hidden_states.size(1) for hidden_states in hidden_states_all.values()]
    names = list(hidden_states_all.keys())  # e.g., ["vlm", "proprio", "action"]
    sample_kv_cache_name = (
        next(iter(kv_caches.keys())) if kv_caches is not None else None
    )

    # always re-compute queries
    query_states_all = {}
    for name in names:
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = mixtures[name].attn_func(
            "forward_q_proj", layer_idx, hidden_states_all[name]
        )
        query_states_all[name] = query_states

    # use kv caches
    key_states_all = {}
    value_states_all = {}
    cache_has_layer = (
        kv_caches[sample_kv_cache_name].has_item(layer_idx) if kv_caches else False
    )  # assume cached mixtures use the same number of layers
    flag_cached = kv_caches is not None and cache_has_layer
    flag_to_cache = (
        kv_caches is not None and not cache_has_layer
    ) or cache_mode == "append"
    for name in names:
        flag_cached_mixture = flag_cached and name in kv_caches
        flag_to_cache_mixture = flag_to_cache and name in kv_caches
        if flag_cached_mixture:
            key_states_cached, value_states_cached = kv_caches[name].get(
                layer_idx
            )  # take the existing cache at the layer, already applied rotary
            print("layer", layer_idx, "use cache", key_states_cached.shape)

        # prep rotary embeddings
        query_states = query_states_all[name]
        cos, sin = mixtures[name].attn_func(
            "forward_rotary_emb", layer_idx, position_ids_all[name]
        )

        # always compute new ones if cache_mode is append
        key_states_new, value_states_new = None, None
        if not flag_cached_mixture or cache_mode == "append":
            hidden_states = hidden_states_all[name]
            # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            key_states_new = mixtures[name].attn_func(
                "forward_k_proj", layer_idx, hidden_states
            )
            value_states_new = mixtures[name].attn_func(
                "forward_v_proj", layer_idx, hidden_states
            )

            # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            key_states_new = mixtures[name].attn_func(
                "forward_apply_rotary_emb", layer_idx, key_states_new, cos, sin
            )
            print("layer", layer_idx, "compute", key_states_new.shape)
            print("layer", layer_idx, "k rotary updated")

            # always cache in append mode, or cache if no cache yet in fixed mode
            if flag_to_cache_mixture:
                kv_caches[name].update(
                    key_states_new,
                    value_states_new,
                    layer_idx,
                )
                print(
                    "layer",
                    layer_idx,
                    "cache updated",
                    kv_caches[name].key_cache[layer_idx].shape,
                )

        # always apply rotary embeddings to Q
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = mixtures[name].attn_func(
            "forward_apply_rotary_emb", layer_idx, query_states, cos, sin
        )
        query_states_all[name] = query_states
        print("layer", layer_idx, "q rotary updated")

        # assign K and V carefully
        if flag_cached_mixture:
            key_states = key_states_cached
            value_states = value_states_cached
            if key_states_new is not None:
                key_states = torch.cat((key_states, key_states_new), dim=-2)
            if value_states_new is not None:
                value_states = torch.cat((value_states, value_states_new), dim=-2)
        else:
            key_states = key_states_new
            value_states = value_states_new
        key_states_all[name] = key_states
        value_states_all[name] = value_states

    # Repeat the key and values to match the number of heads of the query
    for name in names:
        key_states, value_states = mixtures[name].attn_func(
            "repeat_kv",
            layer_idx,
            key_states_all[name],
            value_states_all[name],
        )
        key_states_all[name] = key_states
        value_states_all[name] = value_states

    # Concatenate all the blocks along sequence
    # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim]
    # [Batch_Size, Num_Heads_KV, Full_Seq_Len, Head_Dim]
    query_states = torch.cat(tuple(query_states_all.values()), dim=-2)
    key_states = torch.cat(tuple(key_states_all.values()), dim=-2)
    value_states = torch.cat(tuple(value_states_all.values()), dim=2)

    # Perform the calculation as usual, Q * K^T / sqrt(head_dim). Shape: [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        mixtures[names[0]].head_dim
    )

    # Soft capping
    attn_weights = attn_weights / attn_softclamp
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * attn_softclamp

    # Apply the softmax
    attn_weights = attn_weights + attention_mask
    # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).type_as(query_states)
    # Apply the dropout
    attn_weights = nn.functional.dropout(
        attn_weights, p=attention_dropout, training=mixtures[names[0]].training
    )
    # Multiply by the values. [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len] x [Batch_Size, Num_Heads_KV, Full_Seq_Len, Head_Dim] -> [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim]
    attn_output = torch.matmul(attn_weights, value_states)

    # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim] -> [Batch_Size, Full_Seq_Len, Num_Heads_Q, Head_Dim]
    attn_output = attn_output.transpose(1, 2).contiguous()
    # Concatenate all the heads together. [Batch_Size, Full_Seq_Len, Num_Heads_Q, Head_Dim] -> [Batch_Size, Full_Seq_Len, Num_Heads_Q * Head_Dim]
    attn_output = attn_output.view(bsz, sum(q_lens), -1)

    # split for blocks
    attn_outputs = torch.split(attn_output, q_lens, dim=1)
    attn_outputs = {key: value for key, value in zip(names, attn_outputs)}

    # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
    attn_outputs_final = {}
    for name in names:
        if is_final_layer and name in final_layer_post_attn_skip_names:
            attn_outputs_final[name] = None
        else:
            attn_outputs_final[name] = mixtures[name].attn_func(
                "forward_o_proj", layer_idx, attn_outputs[name]
            )
    return attn_outputs_final


class JointModel(nn.Module):
    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
    ):
        super().__init__()
        config.use_quantize = use_quantize
        config.use_lora = use_lora
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.num_mixture = len(config.mixture)
        self.cache_names = [
            name for name in config.mixture if config.mixture[name].cache
        ]  # name of the mixtures that use cache during generation; no cache during training

        # Mixture --- named modulelist
        self.mixtures = nn.ModuleDict()
        for mixture_name, mixture_config in config.mixture.items():
            mixture_config = OmegaConf.merge(config, mixture_config)
            self.mixtures[mixture_name] = Mixture(
                mixture_config, use_quantize, use_lora
            )
        self.mixture_names = list(config.mixture.keys())

    # def _check_action_parameter_by_name(self, name: str) -> bool:
    #     if (
    #         "q_projs.2" in name
    #         or "k_projs.2" in name
    #         or "v_projs.2" in name
    #         or "o_projs.2" in name
    #         or "mlp.2" in name
    #         or "layernorms.2" in name
    #         or "action_norm" in name
    #         or "_scale" in name  # adaptive layerscale
    #     ):
    #         return True
    #     return False

    # def _check_gemma_trainable_parameter_by_name(self, name: str) -> bool:
    #     last_hidden_layer_index = self.num_hidden_layers - 1
    #     if not (
    #         "q_projs.2" in name
    #         or "k_projs.2" in name
    #         or "v_projs.2" in name
    #         or "o_projs.2" in name
    #         or "mlp.2" in name
    #         or "layernorms.2" in name
    #         or "action_norm" in name
    #         or "_scale" in name  # adaptive layerscale
    #         or name == "norm.weight"  # no need to tune final norm
    #         or f"{last_hidden_layer_index}.post" in name
    #         or f"{last_hidden_layer_index}.mlp" in name
    #         or f"{last_hidden_layer_index}.self_attn.o_projs" in name
    #         or f"{last_hidden_layer_index}.self_attn.v_projs"
    #         in name  # no need to tune part of last layer
    #     ):
    #         return True
    #     return False

    # def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
    #     last_hidden_layer_index = self.num_hidden_layers - 1
    #     if not self._check_action_parameter_by_name(name) and (
    #         f"{last_hidden_layer_index}.post" in name
    #         or f"{last_hidden_layer_index}.mlp" in name
    #         or f"{last_hidden_layer_index}.self_attn.o_projs" in name
    #         or f"{last_hidden_layer_index}.self_attn.v_projs" in name
    #     ):
    #         return True
    #     return False

    # @property
    # def action_parameters(self):
    #     action_parameters = []
    #     for name, param in self.named_parameters():
    #         if self._check_action_parameter_by_name(name):
    #             action_parameters.append(param)
    #     return action_parameters

    # @property
    # def trainable_gemma_parameters(self):
    #     gemma_parameters = []
    #     for name, param in self.named_parameters():
    #         if self._check_gemma_trainable_parameter_by_name(name):
    #             gemma_parameters.append(param)
    #     return gemma_parameters

    # @property
    # def trainable_lora_gemma_parameters(self):
    #     gemma_parameters = []
    #     for name, param in self.named_parameters():
    #         if self._check_gemma_trainable_parameter_by_name(name):
    #             if "lora_" in name:
    #                 gemma_parameters.append(param)
    #     return gemma_parameters

    # def freeze_non_lora_weights_in_gemma(self):
    #     for name, param in self.named_parameters():
    #         if self._check_gemma_trainable_parameter_by_name(name):
    #             param.requires_grad = True if "lora_" in name else False

    def build_mixture_caches(self):
        return {name: MixtureKVCache() for name in self.cache_names}

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids_all: dict[torch.LongTensor],
        embeds_all: dict[torch.FloatTensor],
        time_cond: Optional[torch.FloatTensor] = None,
        kv_caches: Optional[dict[MixtureKVCache | KVCache]] = None,
        cache_mode: Optional[str] = "fixed",  # or append used in text generation
        mixture_names: Optional[List[str]] = None,
    ) -> dict[torch.FloatTensor]:
        """
        Assume attention_mask is in the right block attention form

        embeds_all and position_ids_all need to be in the correct order, e.g., {"vlm": ..., "proprio": ..., "action": ...}
        """
        # override mixture names, e.g., when only generating text
        mixture_names = self.mixture_names if mixture_names is None else mixture_names
        mixtures = {name: self.mixtures[name] for name in mixture_names}

        # normalization
        # [Batch_Size, Seq_Len, Hidden_Size]
        for name in mixture_names:
            hidden_size = embeds_all[name].shape[-1]
            normalizer = torch.tensor(hidden_size**0.5, dtype=embeds_all[name].dtype)
            embeds_all[name] *= normalizer

        # layers
        for layer_idx in range(self.num_hidden_layers):
            embeds_all = forward_mixture_layers(
                mixtures,
                attention_mask,
                position_ids_all,
                embeds_all,
                layer_idx=layer_idx,
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode=cache_mode,
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_all = {}
        for name in mixture_names:
            hidden_states_all[name] = mixtures[name].forward_norm(
                embeds_all[name], time_cond
            )  # can be None
        return hidden_states_all

    # def forward_text_only(
    #     self,
    #     attention_mask: torch.Tensor,
    #     position_ids: torch.LongTensor,
    #     inputs_embeds: torch.FloatTensor,
    #     kv_cache: Optional[KVCache] = None,
    # ) -> torch.FloatTensor:
    #     # normalization
    #     # [Batch_Size, Seq_Len, Hidden_Size]
    #     hidden_size = inputs_embeds.shape[-1]
    #     normalizer = torch.tensor(hidden_size**0.5, dtype=inputs_embeds.dtype)
    #     inputs_embeds *= normalizer

    #     # layers --- only run vlm mixture
    #     embeds_all = {"vlm": inputs_embeds}
    #     position_ids_all = {"vlm": position_ids}
    #     for layer_idx in range(self.num_hidden_layers):
    #         # [Batch_Size, Seq_Len, Hidden_Size]
    #         embeds_all = forward_mixture_layers(
    #             self.mixtures,
    #             attention_mask,
    #             position_ids_all,
    #             embeds_all,
    #             layer_idx=layer_idx,
    #             time_cond=None,
    #             kv_caches={"vlm": kv_cache},
    #         )

    #     # [Batch_Size, Seq_Len, Hidden_Size]
    #     hidden_states = self.mixtures["vlm"].forward_norm(embeds_all["vlm"])
    #     return hidden_states


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/train/pg_bridge_mixture.yaml")
    cfg.mixture_names = ["vlm", "proprio", "action"]
    model = JointModel(cfg.joint.config)

    # dummy inputs
    dummy_num_image_tokens = 7
    q_lens = [
        dummy_num_image_tokens,
        cfg.cond_steps,
        cfg.horizon_steps,
    ]
    total_len = sum(q_lens)
    inputs_embeds = torch.randn(
        1,
        dummy_num_image_tokens,
        cfg.mixture.vlm.hidden_size,
    )  # no history
    proprio_embeds = torch.randn(
        1,
        cfg.cond_steps,
        cfg.mixture.proprio.hidden_size,
    )
    action_embeds = torch.randn(
        1,
        cfg.horizon_steps,
        cfg.mixture.action.hidden_size,
    )
    time_cond = None
    if cfg.action_expert_adaptive_mode:
        time_cond = torch.randn(1, cfg.time_hidden_size)

    kv_caches = model.build_mixture_caches()
    position_ids_all = {
        "vlm": torch.arange(dummy_num_image_tokens)[None],  # no text
        "proprio": torch.arange(cfg.cond_steps)[None],
        "action": torch.arange(cfg.horizon_steps)[None],
    }  # add batch dim

    # block attention
    proprio_start = dummy_num_image_tokens
    proprio_end = dummy_num_image_tokens + 1
    action_start = proprio_end
    causal_mask = torch.full(
        (1, total_len, total_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )  # smallest value, avoid using inf for softmax nan issues with padding
    causal_mask[:, :dummy_num_image_tokens, :dummy_num_image_tokens] = (
        0  # image/text attend to itself
    )
    causal_mask[:, proprio_start:proprio_end, :dummy_num_image_tokens] = (
        0  # proprio attend to image/text
    )
    causal_mask[:, action_start:, :dummy_num_image_tokens] = (
        0  # action attend to image/text
    )
    causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
        0  # proprio attend to itself
    )
    causal_mask[:, action_start:, proprio_start:] = (
        0  # action attend to itself and proprio
    )

    # Add the head dimension
    # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
    causal_mask = causal_mask.unsqueeze(1)

    # dummy denoising
    print("Initial action embeds", action_embeds)
    num_step = 3
    for _step in range(num_step):
        print("running dummy denoising step", _step)
        action_embeds = model(
            attention_mask=causal_mask,
            position_ids_all=position_ids_all,
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "action": action_embeds,
            },
            kv_caches=kv_caches,
            time_cond=time_cond,
        )["action"]
        print("Updated action embeds", action_embeds)

from typing import List, Tuple

import torch


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2]  # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :]  # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim)  # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim,
    )


class KVCache:
    def __init__(self) -> None:
        """list for layers"""
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class JointKVCache:
    def __init__(self) -> None:
        """outer list for layers, inner list for blocks"""
        self.key_cache: List[List[torch.Tensor]] = []
        self.value_cache: List[List[torch.Tensor]] = []

    def has_item(self, layer_idx) -> bool:
        return len(self.key_cache) > layer_idx

    def get(self, layer_idx) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states_all: List[torch.Tensor],
        value_states_all: List[torch.Tensor],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = len(self.key_cache)
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states_all)
            self.value_cache.append(value_states_all)
            return key_states_all, value_states_all
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            for block_idx in range(num_blocks):
                self.key_cache[layer_idx][block_idx] = torch.cat(
                    [self.key_cache[layer_idx][block_idx], key_states_all[block_idx]],
                    dim=-2,
                )
                self.value_cache[layer_idx][block_idx] = torch.cat(
                    [
                        self.value_cache[layer_idx][block_idx],
                        value_states_all[block_idx],
                    ],
                    dim=-2,
                )

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class MixtureKVCache:
    def __init__(self) -> None:
        """layers"""
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def has_item(self, layer_idx) -> bool:
        return len(self.key_cache) > layer_idx

    def get(self, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # num_blocks = len(self.key_cache)
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            return key_states, value_states
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states],
                dim=-2,
            )
            self.value_cache[layer_idx] = torch.cat(
                [
                    self.value_cache[layer_idx],
                    value_states,
                ],
                dim=-2,
            )

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

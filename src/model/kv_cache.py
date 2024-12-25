from typing import List, Tuple

import torch


class KVCache:
    def __init__(self) -> None:
        """list for layers"""
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def has_item(self, layer_idx) -> bool:
        return len(self.key_cache) > layer_idx

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def get(self, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

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


# deprecated
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

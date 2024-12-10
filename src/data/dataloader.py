"""
From: https://github.com/octo-models/octo/blob/main/examples/06_pytorch_oxe_dataloader.py

This example shows how to use the `src.data` dataloader with PyTorch by wrapping it in a simple PyTorch dataloader. The config below also happens to be our exact pretraining config (except for the batch size and shuffle buffer size, which are reduced for demonstration purposes).
"""

import time

import numpy as np
import tensorflow as tf
import torch
import tqdm
from torch.utils.data import DataLoader

from src.data.dataset import make_interleaved_dataset
from src.data.oxe import make_oxe_dataset_kwargs_and_weights

tf.config.set_visible_devices([], "GPU")


class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="/n/fs/llm-unc/data/resize_224"
    )
    parser.add_argument("--mix", type=str, default="oxe_simple")
    args = parser.parse_args()

    # config
    start_time = time.time()
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        args.mix,
        args.data_path,
        load_camera_views=("primary", "wrist"),
    )

    # dataset
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=True,
        shuffle_buffer_size=1000,  # change to 500k for training, large shuffle buffers are important, but adjust to your RAM
        batch_size=None,  # batching will be handles in PyTorch Dataloader object
        balance_weights=True,
        traj_transform_kwargs=dict(
            goal_relabeling_strategy="uniform",
            window_size=2,
            action_horizon=4,
            subsample_length=100,
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs={
                "primary": dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
                "wrist": dict(
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            },
            resize_size=dict(
                primary=(256, 256),
                wrist=(128, 128),
            ),
            num_parallel_calls=200,
        ),
        traj_transform_threads=48,
        traj_read_threads=48,
    )

    # convert for torch
    pytorch_dataset = TorchRLDSDataset(dataset)
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=16,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )
    prep_time = time.time()
    print(f"Preparation time: {prep_time - start_time:.2f}s")

    print("Starting dataloader")
    for i, _sample in tqdm.tqdm(enumerate(dataloader)):
        # _sample: dict with keys 'observation', 'task', 'action', 'dataset_name', 'action_pad_mask'
        # observation: 'image_primary' (torch.Size([16, 2, 256, 256, 3]), 'image_wrist', 'timestep' (torch.Size([16, 2])), 'pad_mask_dict', 'timestep_pad_mask', 'task_completed' (torch.Size([16, 2, 4])
        # task: 'language_instruction', 'pad_mask_dict', 'image_primary', 'image_wrist', 'timestep' (torch.Size([16]))
        # action (torch.Size([16, 2, 4, 7])
        # dataset_name
        # action_pad_mask (torch.Size([16, 2, 4, 7]))
        if i == 100:
            break
    load_time = time.time()
    print(f"Iterative over 100 batches: {load_time - prep_time:.2f}s")

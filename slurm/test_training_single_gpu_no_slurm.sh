#!/bin/bash

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 uv run \
    scripts/run.py \
    --config-name=fractal \
    device=cuda:0 \
    debug=False \
    wandb=null \
    log_dir=results/test/ \
    global_batch_size=32 \
    data.train.shuffle_buffer_size=10000 \
    data.train.num_parallel_calls=10 \
    eval_freq=2000000

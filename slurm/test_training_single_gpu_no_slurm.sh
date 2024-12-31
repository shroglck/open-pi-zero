#!/bin/bash

# no wandb logging, logging GPU memory usage
# smaller global batch size, small resource for dataloading
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 uv run \
    scripts/run.py \
    --config-name=bridge \
    device=cuda:0 \
    debug=True \
    wandb=null \
    log_dir=results/test/ \
    global_batch_size=32 \
    per_device_batch_size=16 \
    flow_schedule=uniform \
    data.train.shuffle_buffer_size=10000 \
    data.train.num_parallel_calls=10 \
    eval_freq=20 \
    eval_size=32 \
    use_torch_compile=True \
    use_bfloat16=True

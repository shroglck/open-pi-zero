# open-pi-zero

This repo implements the [pi0](https://www.physicalintelligence.company/download/pi0.pdf) model from Physical Intelligence. The model adopts a MoE-like architecture, and uses a pre-trained 3B PaliGemma VLM (2.291B to be fine-tuned) and a new set of action expert parameters (0.315B).

Pre-trained checkpoints and eval results coming soon...

## Installation
Clone the repository to your home directory

If running Simpler eval, clone my fork to your home directory (addded proprio support)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the dependencies will be configured automatically when running any `uv run ...` command. Or `pip install -e .`

Download PaliGemma weights
```console
git clone https://huggingface.co/google/paligemma-3b-pt-224    # at TRANSFORMERS_CACHE
```

### Tests
VLA with real img/text, and output text using paligemma weights
```console
uv run src/model/vla_mixture/model.py --text_only --load_pretrained_weights
```

VLA with dummy img/text, proprio, and action, output dummy flow matching action
```console
uv run src/model/vla_mixture/model.py
```

## Data

Download fractal data
```console
uv run gsutil -m cp -r gs://gresearch/robotics/fractal20220817_data/0.1.0 ../data   # fractal data
```

Download bridge dataset from [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/) instead.

Preprocess (taken from [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod/tree/main) and [Octo](https://github.com/kpertsch/rlds_dataset_mod/blob/main/prepare_open_x.sh)). Then run (with enough RAM) [possible error](doc/error.md#5)
```console
uv run scripts/data/modify_rlds_dataset.py \
    --dataset=fractal20220817_data \
    --data_dir=/n/fs/llm-unc/data/ \
    --target_dir=/n/fs/llm-unc/data/resize_224 \
    --mods=resize_and_jpeg_encode \
    --n_workers=40 \
    --max_episodes_in_memory=400
```
This resizes the images to 224x224 for PaliGemma (as opposed to 256x256 in Octo).

To calculate the normalization statistics and save it before training, run dataloader, which might take some time:
```console
uv run src/data/dataloader.py \
    --data_path=/n/fs/llm-unc/data/resize_224 \
    --mix=oxe_simple
```

[Conventions on proprio / gripper actions](doc/convention.md)

## Training

See launch file. 400(?)GB RAM
```console
uv run torchrun --nnodes=1 --nproc_per_node=$NUM_GPU --rdzv_id=100 --rdzv_backend=c10d --max-restarts=1 --standalone --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT scripts/run.py \
    --config-name=pg_bridge_mixture \  # or pg_fractal_mixture
    --config_path=../config/train
```

The model was trained with per GPU batch size 16 with L40 using bfloat16 and 8-bit optimizer --- single image (256 tokens), max 20 text tokens, 1 proprio token, and 4 action tokens (chunk size 4). Optimizer offloading and (Q)LoRA are also implemented.

[Possible error if running quantization](doc/error.md#9) [Some observations](doc/notes.md)

## Evaluation

Simpler with Bridge tasks
```console
uv run scripts/run.py \
    --config-name=pg_bridge \
    --config-path=../config/eval
```

Full sweeping script will be added

## Things to implement/try

Use EMA (Simpler evals seem sensitive right now; EMA should help). Switch to GPU Simpler. Fine-tuning with a new mixture (e.g., second camera view into pre-trained Dino/Siglip) and gradual unmasking. Co-training with (self-)supervision on modalities other than action.

## Acknowledgement

PaliGemma setup is largely adopted from [Open-source PaliGemma](https://github.com/hkproj/pytorch-paligemma/tree/main).

Dataset loading is adopted from [Octo](https://octo-models.github.io/) and [dlimp](https://github.com/kvablack/dlimp).

[OpenVLA](https://github.com/openvla/openvla), [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod/tree/main), [Pi0](https://www.physicalintelligence.company/download/pi0.pdf), [Flow matching](https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb)

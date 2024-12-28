# pg-vla

This repo implements the [pi0](https://www.physicalintelligence.company/download/pi0.pdf) model from Physical Intelligence. The code is written in a fairly modular way so it is easy to add/remove mixtures to the MoE architecture.

The model uses a pre-trained 3B PaliGemma VLM from Google (2.291B fine-tuned) and a new set of action expert parameters (0.315B). Currently it is trained with per GPU batch size 16 with L40 using bfloat16 and 8-bit optimizer --- single image (256 tokens), max 20 text tokens, 1 proprio token, and 4 action tokens (chunk size 4). Optimizer offloading and (Q)LoRA are also implemented.

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

VLA with dummy img/text, proprio, and action, output flow matching action
```console
uv run src/model/vla_mixture/model.py
```

Block attention with dummy embeddings for img/text, proprio, and action
```console
uv run src/model/vla_mixture/modules.py
```

Text generation with original paligemma implementation
```console
uv run scripts/tests/run_paligemma.py \
    --prompt "this image shows " \
    --image_file_path "media/maniskill_pp.png" \
    --max_tokens_to_generate 100 \
    --temperature 0.8 \
    --top_p 0.9
```

## Data

Currently using Google robot (EDR) and Widowx data (fractal/RT-1, bridge, haven't tried BC-Z or RoboVQA)

Download fractal data
```console
uv run gsutil -m cp -r gs://gresearch/robotics/fractal20220817_data/0.1.0 ../data   # fractal data
```

Download bridge dataset from [here](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/) instead.

Preprocess (taken from [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod/tree/main) and [Octo](https://github.com/kpertsch/rlds_dataset_mod/blob/main/prepare_open_x.sh)). First we need to comment out Line 299-306 in `.venv/lib/python3.10/site-packages/tensorflow_datasets/core/dataset_builder.py` to avoid the `AttributeError: 'MultiplexedPath' object has no attribute 'parts'` error (seems an issue with running python3.10; using `tensorflow_datasets==4.9.2` fixes this issue but disabling gcs does not work any more somehow). Then run (with enough RAM)
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

### EE proprio

Fractal data has xyzw quaternion in proprio (upon inspection), and I have been using wxyz in Simpler since it follows the transforms3d library. Bridge uses sxyz euler. EE pose saved in bridge data is relative to a top-down pose (instead of base pose). Both datasets use +x for forward, +y for left, and +z for upward.

### Gripper proprio and action

In Octo, bridge data has 1 for gripper state open and -1 for closed after normalization (continuous), and 1 for gripper action open and 0 for closing (without normalization, binarized). Fractal data has -1 for gripper state open and 1 for open closed after normalization (continuous), and also 1 for gripper action open and 0 for closing (without normalization, binarized).

I added gripper width (1 for open and 0 for closed) to the environment observation in Simpler. Then for the action in Simpler, widowx robot (bridge) has 1 for opening gripper and -1 for closing. Google robot has 1 for closing gripper and -1 for opening.

## Training

See launch file
```console
uv run torchrun --nnodes=1 --nproc_per_node=$NUM_GPU --rdzv_id=100 --rdzv_backend=c10d --max-restarts=1 --standalone --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT scripts/run.py \
    --config-name=pg_bridge_mixture \  # or pg_fractal_mixture
    --config_path=../config/train
```

If using quantization, need to modify Line 474 in `.venv/lib64/python3.10/site-packages/bitsandbytes/autograd/_functions.py` to `return output.clone` from `return output` ([related issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/736)).

## Evaluation

Simpler with Bridge tasks
```console
uv run scripts/run.py \
    --config-name=pg_bridge \
    --config-path=../config/eval
```

Full sweeping script will be added


## Acknowledgement

Octo, OpenVLA, [dlimp](https://github.com/kvablack/dlimp), [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod/tree/main), Pi0 paper, [Open-source PaliGemma](https://github.com/hkproj/pytorch-paligemma/tree/main)

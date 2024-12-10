# pg-vla

## Installation
Install uv
```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```
The dependencies will be configured automatically when running any `uv run ...` command.

### Test PaliGemma text generation
```console
uv run python scripts/tests/run_paligemma.py \
    --model_path "/n/fs/llm-unc/.cache/paligemma-3b-pt-224" \
    --prompt "this image shows " \
    --image_file_path "media/maniskill_pp.png" \
    --max_tokens_to_generate 100 \
    --temperature 0.8 \
    --top_p 0.9
```

## Data

### OXE
Currently using meta robot and widowx data (fractal/RT-1, bridge, optionally BC-Z and RoboVQA)

Download data by
```console
uv run gsutil -m cp -r gs://gresearch/robotics/fractal20220817_data/0.1.0 ../data
```

Preprocess (taken from [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod/tree/main) and [Octo](https://github.com/kpertsch/rlds_dataset_mod/blob/main/prepare_open_x.sh)). First we need to comment out Line 299-306 in `/n/fs/llm-unc/pg-vla/.venv/lib/python3.10/site-packages/tensorflow_datasets/core/dataset_builder.py` to avoid the `AttributeError: 'MultiplexedPath' object has no attribute 'parts'` error (seems an issue with running python3.10; using `tensorflow_datasets==4.9.2` fixes this issue but disabling gcs does not work any more somehow). Then run
```console
uv run python scripts/data/modify_rlds_dataset.py \
    --dataset=fractal20220817_data \
    --data_dir=/n/fs/llm-unc/data/ \
    --target_dir=/n/fs/llm-unc/data/resize_224 \
    --mods=resize_and_jpeg_encode \
    --n_workers=40 \
    --max_episodes_in_memory=400
```
This resizes the images to 224x224 (as opposed to 256x256 in Octo).

Running dataloader for the first time will take some time for calculating the normalization statistics, which is then saved.
```console
uv run python src/data/dataloader.py \
    --data_path=/n/fs/llm-unc/data/resize_224 \
    --mix=oxe_simple
```

## Model

## Resources

Octo, OpenVLA

Paligemma [https://github.com/hkproj/pytorch-paligemma/tree/main](https://github.com/hkproj/pytorch-paligemma/tree/main)

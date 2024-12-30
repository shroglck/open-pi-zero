# open-pi-zero

This repo implements the [pi0](https://www.physicalintelligence.company/download/pi0.pdf) model from Physical Intelligence.

The model adopts a MoE-like architecture, and uses a pre-trained 3B PaliGemma VLM (2.291B to be fine-tuned) and a new set of action expert parameters (0.315B). Bi-directional block-wise causal masking is used such that VLM block attends to itself, proprioception (sharing weights with action) attends to itself and VLM, and action attends to all. The model is trained with flow matching loss on the output of action expert.

## Installation
Clone this repository. If running Simpler eval, clone my [fork](https://github.com/allenzren/SimplerEnv) (addded proprio support) to the same directory
```console
git clone https://github.com/allenzren/SimplerEnv --recurse-submodules  # Simpler and ManiSkill2_real2sim submodule
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run `uv sync`. Or `pip install -e .` instead of using uv.

Set environment variables `VLA_DATA_DIR`, `VLA_LOG_DIR', and `VLA_WANDB_ENTITY` by running `source script/set_path.sh`

Download PaliGemma weights at `TRANSFORMERS_CACHE`
```console
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

### Test text generation with pre-trained weights
```console
uv run src/model/vla/pizero.py --text_only --load_pretrained_weights
```

<!-- VLA with dummy img/text, proprio, and action, output dummy flow matching action
```console
uv run src/model/vla/pizero.py
``` -->

## Try checkpoints

I have only tried training with either fractal or bridge dataset so far (no mixing with other OXE data).

...

The model was trained with per GPU batch size 16 with L40 using bfloat16 and 8-bit optimizer --- single image (256 tokens), max 20 text tokens, 1 proprio token, and 4 action tokens (chunk size 4). Optimizer offloading and (Q)LoRA are also implemented.

### Eval results

For both datasets, I have tried two different schedule for sampling flow matching timesteps during training: Linear (uniform between 0 and 1) and Gamma (higher density at earlier timesteps). Gamma is proposed by the Pi0 paper. Below shows the success rates in the visual matching setting in Simpler

| Dataset-flow matching schedule  | Carrot on plate | Eggplant in basket | Spoon on towel | Stack cube |
|---------------------------------|-----------------|--------------------|----------------|------------|
| [Bridge-Linear](...)   | ... | ... | ... | ... |
| [Bridge-Gamma](...)    | ... | ... | ... | ... |

|                                 | Pick up Coke Can | Move Near | Close Drawer | Open Drawer | Open Drawer and Put Apple In |
|---------------------------------|------------------|-----------|--------------|-------------|------------------------------|
| [Fractal-Linear](...)   | ... | ... | ... | ... |
| [Fractal-Gamma](...)    | ... | ... | ... | ... |

Visual aggregation results coming soon.

### Inference speed

...

## Run training

### Download data

Download fractal data (following [OXE](https://github.com/google-deepmind/open_x_embodiment?tab=readme-ov-file))
```console
uv run gsutil -m cp -r gs://gresearch/robotics/fractal20220817_data/0.1.0 <data_path>
```

Download bridge data from [RAIL link](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/) as suggested by OXE.

### Data pre-processing

Run (with enough RAM) [slurm/modify_rlds.sh](slurm/modify_rlds.sh), which resizes the images to 224x224 for PaliGemma. [Possible error](doc/error.md#5)

[Conventions on proprio / gripper actions](doc/convention.md)

### Training scripts

See examples in the [slurm](slurm/) folder. TFDS dataloading takes a growing amount of CPU RAM. With, it peaks at ...

[Possible error if running quantization](doc/error.md#9)   | [Some observations](doc/notes.md)

## Run evaluation

### Simpler

See examples in the [slurm](slurm/) folder. You need to set `env.adapter.dataset_statistics_path` to the dataset statistics json file generated in your training, located in the dataset folder.

## Things to implement / try

Use EMA (Simpler evals seem sensitive right now; EMA should help). Switch to GPU Simpler. Fine-tuning with a new mixture (e.g., second camera view into pre-trained DINO/SigLIP) and gradual unmasking. Co-training with (self-)supervision on modalities other than action.

## Acknowledgement

PaliGemma setup is largely adopted from [Open-source PaliGemma](https://github.com/hkproj/pytorch-paligemma/tree/main).

Dataset loading is adopted from [Octo](https://octo-models.github.io/) and [dlimp](https://github.com/kvablack/dlimp).

[Pi0](https://www.physicalintelligence.company/download/pi0.pdf), [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod/tree/main), [OpenVLA](https://github.com/openvla/openvla), [Flow matching](https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb), [Implementation by lucidrains](https://github.com/lucidrains/pi-zero-pytorch).

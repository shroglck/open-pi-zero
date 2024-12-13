"""
Launcher for all experiments.

"""

import logging
import math
import os
import sys

import hydra
import pretty_errors
import torch
from omegaconf import OmegaConf, open_dict

# dummy
print(pretty_errors.__version__)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def _main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # figure out the current gpu
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        from torch.distributed import destroy_process_group, init_process_group

        def ddp_setup():
            # os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"].split(",")[0]
            # os.environ["MASTER_PORT"] = "29500"
            # os.environ["NCCL_DEBUG"] = "INFO"
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            torch.cuda.empty_cache()
            # torch.cuda.set_device(rank)
            init_process_group(backend="nccl")

        ddp_setup()
        gpu_id = int(os.environ["LOCAL_RANK"])
    else:
        gpu_id = 0
    with open_dict(cfg):
        cfg.gpu_id = gpu_id

    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()

    if num_gpus > 1:
        destroy_process_group()


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config/train"),
    config_name="pg_oxe.yaml",
)  # defaults
def main(cfg: OmegaConf):
    _main(cfg)


if __name__ == "__main__":
    main()

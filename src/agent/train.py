"""
Main training agent. Using bfloat16 and AMP.

"""

import logging
import os
import random
from collections import deque

import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb
from src.agent.dataset import TorchRLDSInterleavedDataset
from src.model.vla.model import VLA
from src.model.vla.processing import VLAProcessor
from src.utils.monitor import Timer, log_allocated_gpu_memory, log_execution_time
from src.utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions

log = logging.getLogger(__name__)


class TrainAgent:
    def __init__(self, cfg):
        # seeding
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # devices
        self.num_gpus = torch.cuda.device_count()
        self.gpu_id = int(cfg.gpu_id)
        self.multi_gpu = self.num_gpus > 1
        self.main_rank = self.gpu_id == 0

        # training params
        self.n_epochs = cfg.n_epochs
        self.max_grad_norm = cfg.max_grad_norm
        self.log_freq = cfg.log_freq
        self.save_model_batch_freq = cfg.save_model_batch_freq
        self.save_model_epoch_freq = cfg.save_model_epoch_freq
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.use_wandb = cfg.get("wandb", None)
        if self.use_wandb and self.main_rank:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        self.debug = cfg.get("debug", False)

        # model
        assert not (
            (cfg.quantize or cfg.lora) and not cfg.load_pretrained_weights
        ), "Please load pretrained weights if quantizing VLM or using Lora."
        if cfg.quantize and not cfg.lora:
            log.warning(
                "Quantizing VLM but not adding Lora weights, which means the VLM will be fully frozen!"
            )  # since the weights have requires_grad=False. However, we are not excluding the weights from the optimizer yet!
        self.model = VLA(cfg)
        if cfg.load_pretrained_weights:
            self.model.load_pretrained_weights()
            self.model.freeze_unused_weights()
        if cfg.lora:
            self.model.freeze_non_lora_weights_in_vlm()
        self.model = self.model.to(torch.bfloat16)
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.model.to(self.device)  # quantization happens
        if self.multi_gpu:
            log.info(f"Using {self.num_gpus} GPUs.")
            log.info(f"GPU for the current process: {self.gpu_id}")
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = DDP(
                self.model,
                device_ids=[self.gpu_id],
                gradient_as_bucket_view=True,
                static_graph=False,
            )
            model = self.model.module
        else:
            model = self.model
        if self.multi_gpu:
            dist.barrier()
        log_allocated_gpu_memory(log, "loading model")

        # determine batch size and gradient accumulation steps
        self.grad_accumulation_steps = (
            cfg.global_batch_size // cfg.per_device_batch_size // self.num_gpus
        )

        # tokenizer --- assume paligemma for now
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )

        # processor
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )

        # dataloader --- use the same for all ranks
        dataset_wrapper = TorchRLDSInterleavedDataset(
            cfg.data,
            train=True,
        )
        self.dataloader = DataLoader(
            dataset_wrapper.dataset,
            batch_size=cfg.per_device_batch_size,
            num_workers=0,  # let TFDS handle parallelism
            shuffle=False,
            pin_memory=True,
            # sampler=DistributedSampler(dataset_wrapper.dataset),
        )  # full bridge dataset has 2195527 transitions and 60064 trajectories
        num_batch_per_epoch = len(dataset_wrapper.dataset) // cfg.global_batch_size
        log.info(f"Global batch size: {cfg.global_batch_size}")
        log.info(f"Per device batch size: {cfg.per_device_batch_size}")
        log.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")
        log.info(f"Number of batches per epoch: {num_batch_per_epoch}")

        # optimizer - action only: 0.315B (0.342B with adaptive and time_dim=256), rest: 2.359B
        self.train_action_only = cfg.train_action_only
        self.trained_parameters = model.action_expert_parameters
        if cfg.offload_optimizer:
            from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

            self.action_optimizer = CPUOffloadOptimizer(
                model.action_expert_parameters,
                torch.optim.AdamW,  # no need to use low-bit optimizer
                fused=True,
                offload_gradients=False,  # does not work with grad accumulation
                lr=cfg.action_lr,
                weight_decay=cfg.action_weight_decay,
            )
        else:
            import bitsandbytes as bnb

            self.action_optimizer = bnb.optim.AdamW8bit(
                model.action_expert_parameters,
                lr=cfg.action_lr,
                weight_decay=cfg.action_weight_decay,
            )
        self.action_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.action_optimizer,
            first_cycle_steps=cfg.action_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.action_lr,
            min_lr=cfg.action_lr_scheduler.min_lr,
            warmup_steps=cfg.action_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        log.info(
            f"Number of trained parameters (Action): {get_num_params_in_billions(self.action_optimizer):.3f}B"
        )
        if not self.train_action_only:
            if cfg.lora:
                vlm_trained_parameters = model.lora_pretrained_parameters
            else:
                vlm_trained_parameters = model.pretrained_parameters
            self.trained_parameters += vlm_trained_parameters
            if cfg.offload_optimizer:
                self.vlm_optimizer = CPUOffloadOptimizer(
                    vlm_trained_parameters,
                    torch.optim.AdamW,
                    fused=True,
                    offload_gradients=False,
                    lr=cfg.vlm_lr,
                    weight_decay=cfg.vlm_weight_decay,
                )
            else:
                self.vlm_optimizer = bnb.optim.AdamW8bit(
                    vlm_trained_parameters,
                    lr=cfg.vlm_lr,
                    weight_decay=cfg.vlm_weight_decay,
                )
            self.vlm_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.vlm_optimizer,
                first_cycle_steps=cfg.vlm_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.vlm_lr,
                min_lr=cfg.vlm_lr_scheduler.min_lr,
                warmup_steps=cfg.vlm_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
            log.info(
                f"Number of trained parameters (VLM): {get_num_params_in_billions(self.vlm_optimizer):.3f}B"
            )

    def run(self):
        timer = Timer()
        epoch = 1
        cnt_batch = 0
        cnt_update = 0
        loss_train_deque = deque(maxlen=self.grad_accumulation_steps)
        if self.multi_gpu:
            import torch.distributed as dist

        for _ in range(self.n_epochs):
            log.info(f"Epoch {epoch}/{self.n_epochs} starts")
            for batch_ind, batch in enumerate(self.dataloader):
                cnt_update_in_epoch = 0
                """
                batch: dict with keys 'observation', 'task', 'action', 'dataset_name', 'action_pad_mask'
                observation: 'image_primary' (torch.Size([bsz, 1, H, W, 3], uint8), 'image_wrist', 'timestep' (torch.Size([bsz, 1])), 'pad_mask_dict', 'timestep_pad_mask', 'task_completed' (torch.Size([bsz, window, 4]), 'proprio' (torch.Size([bsz, window, proprio_dim])
                task: 'language_instruction', 'pad_mask_dict', 'image_primary', 'image_wrist', 'timestep' (torch.Size([bsz]))
                action (torch.Size([bsz, window, horizon, action_dim], float32)
                dataset_name
                action_pad_mask (torch.Size([bsz, window, horizon, action_dim]))
                """
                images = batch["observation"]["image_primary"]
                proprios = batch["observation"]["proprio"]
                actions = batch["action"].squeeze(1)  # remove the time dimension
                texts = [
                    text.decode("utf-8")
                    for text in batch["task"]["language_instruction"]
                ]
                if self.debug and batch_ind == 0:
                    log.info(f"{self.gpu_id} device {self.device}")
                    log.info(f"{self.gpu_id} texts {texts}")
                    log.info(f"{self.gpu_id} images {images.shape}")
                    log.info(
                        f"{self.gpu_id} actions {actions.shape} {actions.mean()} {actions.std()} {actions.max()} {actions.min()}"
                    )
                    log.info(
                        f"{self.gpu_id} proprios {proprios.shape} {proprios.mean()} {proprios.std()} {proprios.max()} {proprios.min()}"
                    )

                    # Save an image for debugging
                    image = images[0, 0].clone().to("cpu")
                    image = Image.fromarray(image.numpy())
                    image.save(os.path.join(self.logdir, f"image_{self.gpu_id}.png"))

                # process image and text
                images = einops.rearrange(
                    images, "B T H W C -> B (T C) H W"
                )  # remove cond_steps dimension
                model_inputs = self.processor(text=texts, images=images)
                pixel_values = model_inputs["pixel_values"]
                input_ids = model_inputs["input_ids"]
                attention_mask = model_inputs[
                    "attention_mask"
                ]  # with padding if bsz > 1

                self.model.train()
                # make sure only syncing when taking gradient steps
                if (cnt_batch + 1) % self.grad_accumulation_steps != 0:
                    with self.model.no_sync():
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                            loss_train = self.model(
                                pixel_values=pixel_values.to(self.device),
                                input_ids=input_ids.to(self.device),
                                proprios=proprios.to(self.device),
                                actions=actions.to(self.device),
                                attention_mask=attention_mask.to(self.device),
                            )
                            loss_train_deque.append(loss_train.item())
                        if self.debug:
                            log_allocated_gpu_memory(log, f"forward batch {batch_ind}")
                        # update -- outside autocast
                        normalized_loss = loss_train / self.grad_accumulation_steps
                        normalized_loss.backward()
                        if self.debug:
                            log_allocated_gpu_memory(log, f"backward batch {batch_ind}")
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        loss_train = self.model(
                            pixel_values=pixel_values.to(self.device),
                            input_ids=input_ids.to(self.device),
                            proprios=proprios.to(self.device),
                            actions=actions.to(self.device),
                            attention_mask=attention_mask.to(self.device),
                        )
                        loss_train_deque.append(loss_train.item())
                    if self.debug:
                        log_allocated_gpu_memory(log, f"forward batch {batch_ind}")
                    # update -- outside autocast
                    normalized_loss = loss_train / self.grad_accumulation_steps
                    normalized_loss.backward()  # now gradients are synced across gpus
                    if self.debug:
                        log_allocated_gpu_memory(log, f"backward batch {batch_ind}")

                    # step
                    torch.nn.utils.clip_grad_norm_(
                        self.trained_parameters,
                        max_norm=self.max_grad_norm,
                    )  # not work any more because of offload? no error thrown
                    self.action_optimizer.step()
                    self.action_lr_scheduler.step()
                    if not self.train_action_only:
                        self.vlm_optimizer.step()
                        self.vlm_lr_scheduler.step()
                    if self.debug:
                        log_allocated_gpu_memory(
                            log, f"optimizer step batch {batch_ind}"
                        )
                    self.action_optimizer.zero_grad(set_to_none=True)
                    if not self.train_action_only:
                        self.vlm_optimizer.zero_grad(set_to_none=True)
                    if self.debug:
                        log_allocated_gpu_memory(
                            log, f"optimizer zero grad batch {batch_ind}"
                        )
                    cnt_update += 1
                    cnt_update_in_epoch += 1

                # TODO: validation with action accuracy
                loss_val = None

                # log loss
                if self.main_rank and cnt_batch % self.log_freq == 0:
                    loss_train_metric = np.mean(loss_train_deque)
                    log_msg = f"Epoch {epoch} Batch {batch_ind}: t:{timer():8.4f} | train loss {loss_train_metric:8.4f} | action lr: {self.action_optimizer.param_groups[0]['lr']:10.8f}"
                    if not self.train_action_only:
                        log_msg += f" | vlm lr: {self.vlm_optimizer.param_groups[0]['lr']:10.8f}"
                    log.info(log_msg)
                    if self.use_wandb:
                        wandb_metrics = {
                            "loss - train": loss_train_metric,
                            "gradient steps": cnt_update,
                            "action lr": self.action_optimizer.param_groups[0]["lr"],
                        }
                        if not self.train_action_only:
                            wandb_metrics["vlm lr"] = self.vlm_optimizer.param_groups[
                                0
                            ]["lr"]
                        if loss_val is not None:
                            wandb_metrics["loss - val"] = loss_val
                        wandb.log(wandb_metrics, step=cnt_batch, commit=True)

                # Count
                cnt_batch += 1

                # save model at batch level
                if self.main_rank and cnt_batch % self.save_model_batch_freq == 0:
                    self.save_model(epoch)

            # Save model at epoch level --- always save at the end
            if self.main_rank and (
                epoch % self.save_model_epoch_freq == 0 or epoch == self.n_epochs
            ):
                self.save_model(epoch)

            # Count
            epoch += 1
            if self.multi_gpu:
                dist.barrier()

    @log_execution_time()
    def save_model(self, epoch):
        data = {
            "epoch": epoch,
            "model": (
                self.model.module.state_dict()
                if self.multi_gpu
                else self.model.state_dict()
            ),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")

    def load_model(self, epoch):
        # TODO
        loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")
        data = torch.load(loadpath, weights_only=True)

        epoch = data["epoch"]
        if self.multi_gpu:
            model = self.model.module
        else:
            model = self.model
        model.load_state_dict(data["model"])
        return epoch

import os
import random
import time

import hydra
import imageio
import numpy as np
import simpler_env
import torch
from omegaconf import OmegaConf

from src.model.vla.pizero import PiZero
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time


@log_execution_time()
def load_checkpoint(model, path):
    """load to cpu first, then move to gpu"""
    data = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(data["model"], strict=True)
    print(f"Loaded model from {path}")


def main(args):
    # seeding
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # devices
    device = torch.device(f"cuda:{args.gpu_id}")

    # load corresponding config
    if "fractal" in args.checkpoint_path:
        cfg = OmegaConf.load(
            "config/eval/fractal_apple.yaml"
        )  # doesn't matter which task
    if "bridge" in args.checkpoint_path:
        cfg = OmegaConf.load("config/eval/bridge.yaml")

    # determine flow matching schedule
    if "uniform" in args.checkpoint_path:
        cfg.flow_schedule = "uniform"
    if "gamma" in args.checkpoint_path:
        cfg.flow_schedule = "gamma"

    # model --- about 16gb with float32
    model = PiZero(cfg, use_ddp=False)
    load_checkpoint(model, args.checkpoint_path)
    model.freeze_all_weights()
    model.to(device)
    model.eval()
    print(f"Using cuda device: {device}")
    log_allocated_gpu_memory(None, "loading model")

    # simpler env
    env = simpler_env.make(args.task)

    # env specifics
    env_adapter = hydra.utils.instantiate(cfg.env.adapter)
    env_adapter.reset()

    # run an episode
    obs, reset_info = env.reset(options={"episode_id": 0})
    instruction = env.get_language_instruction()
    if args.recording:
        os.environ["TOKENIZERS_PARALLELISM"] = (
            "false"  # avoid tokenizer forking warning about deadlock
        )
        video_writer = imageio.get_writer(f"try_{args.task}.mp4")
    print(
        f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
    )
    cnt_step = 0
    inference_times = []
    while 1:
        # infer action chunk
        inputs = env_adapter.preprocess(env, obs, instruction)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        start_inference_time = time.time()
        with torch.no_grad():
            actions = model.infer_action(**inputs)[0]
        if cnt_step > 0:
            inference_times.append(time.time() - start_inference_time)
        env_actions = env_adapter.postprocess(actions.cpu().numpy())

        # environment step
        for env_action in env_actions[: cfg.act_steps]:
            obs, reward, success, truncated, info = env.step(env_action)
            cnt_step += 1
            if truncated:
                break

        # save frame
        if args.recording:
            video_writer.append_data(env_adapter.get_video_frame(env, obs))

        # update instruction in long horizon tasks, e.g., pick apple ---> put in top drawer
        new_instruction = env.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction

        # original octo eval only done when timeout, i.e., not upon success
        if truncated:
            if args.recording:
                video_writer.close()
            break

    # summary
    print("\n\n============ Summary ============")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Action chunk steps (predicted): {cfg.horizon_steps}")
    print(f"Action chunk steps (executed): {cfg.act_steps}")
    print(
        f"Average inference time (skipping first step): {np.mean(inference_times):.3f}s"
    )
    print(f"Peak VRAM usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")
    print(f"Task: {args.task}")
    print(f"Total environment steps: {cnt_step}")
    print(f"Success: {success}")
    if args.recording:
        print(f"Video saved as try_{args.task}.mp4")
    print("======================================\n\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="google_robot_pick_horizontal_coke_can",
        choices=[
            "widowx_carrot_on_plate",
            "widowx_put_eggplant_in_basket",
            "widowx_spoon_on_towel",
            "widowx_stack_cube",
            "google_robot_pick_horizontal_coke_can",
            "google_robot_pick_vertical_coke_can",
            "google_robot_pick_standing_coke_can",
            "google_robot_move_near_v0",
            "google_robot_open_drawer",
            "google_robot_close_drawer",
            "google_robot_place_apple_in_closed_top_drawer",
        ],
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="results/fractal_gamma.pt",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--recording",
        action="store_true",
    )
    args = parser.parse_args()

    # check task
    if "google_robot" in args.task:
        assert "fractal" in args.checkpoint_path
    if "widowx" in args.task:
        assert "bridge" in args.checkpoint_path

    main(parser.parse_args())

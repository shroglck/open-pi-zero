"""
Main eval agent. Only for Simpler for now.

"""

import logging
import os
import random

import hydra
import imageio
import numpy as np
import simpler_env
import torch

from src.model.vla_mixture.model_test import VLA
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time

log = logging.getLogger(__name__)


class EvalAgent:
    def __init__(self, cfg):
        # seeding
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # devices
        self.gpu_id = cfg.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")

        # training params
        self.n_eval_episode = cfg.n_eval_episode
        self.n_video = cfg.n_video
        self.log_dir = cfg.log_dir
        self.video_dir = os.path.join(self.log_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        # didn't see a difference in inference speed with L40
        # torch.set_float32_matmul_precision("medium")  # highest, high
        # torch.backends.cudnn.benchmark = True # for vit conv layers

        # model --- about 16gb with float32
        self.model = VLA(cfg, use_ddp=False)
        self.load_checkpoint(cfg.resume_checkpoint_path)
        self.model.freeze_all_weights()
        self.use_bf16 = cfg.get("use_bf16", False)
        if self.use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model.to(self.device)  # haven't tested quantization
        # self.model = torch.compile(self.model, mode="max-autotune")
        self.model.eval()
        log.info(f"Using cuda device: {self.device}")
        log_allocated_gpu_memory(log, "loading model")
        self.act_steps = (
            cfg.act_steps
        )  # e.g., run first two steps of predicted four steps

        # env --- no parallelized
        self.env = simpler_env.make(cfg.env.task)

        # env specifics
        self.env_adapter = hydra.utils.instantiate(cfg.env.adapter)

    def run(self):
        """
        Roughly following simpler_env/simple_inference_visual_matching_prepackaged_envs.py

        Assume no obs history for now
        """
        env = self.env
        env_adapter = self.env_adapter
        cnt_episode = 0
        successes = []

        # run episodes --- not dealing with subtasks
        env_reset_options = {}
        env_reset_options["obj_init_options"] = {
            "episode_id": cnt_episode,  # this determines the obj inits in bridge
        }
        obs, reset_info = env.reset(options=env_reset_options)
        env_adapter.reset()
        # obs keys: 'agent', 'extra', 'camera_param', 'image'
        # agent: 'qpos', 'qvel', 'eef_pos', 'controller', 'base_pose'
        instruction = env.get_language_instruction()
        recording = self.n_video > 0
        if recording:
            os.environ["TOKENIZERS_PARALLELISM"] = (
                "false"  # avoid tokenizer forking warning about deadlock
            )

            def video_parent_path(x):
                return os.path.join(self.video_dir, f"video_{x}")

            video_writer = imageio.get_writer(video_parent_path(cnt_episode) + ".mp4")
        # is_final_subtask = env.is_final_subtask()
        log.info(
            f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
        )
        # Bridge: {'scene_name': 'bridge_table_1_v1', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': '', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png', 'rgb_overlay_cameras': ['3rd_view_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'episode_model_ids': ['bridge_carrot_generated_modified', 'bridge_plate_objaverse_larger'], 'episode_model_scales': [1.0, 1.0], 'episode_source_obj_name': 'bridge_carrot_generated_modified', 'episode_target_obj_name': 'bridge_plate_objaverse_larger', 'episode_source_obj_init_pose_wrt_robot_base': Pose([0.381995, 0.104536, 0.0175282], [-0.706719, 0.0305475, -0.0305745, -0.706173]), 'episode_target_obj_init_pose_wrt_robot_base': Pose([0.232, -0.047, -0.000468373], [2.00041e-10, -5.10387e-07, -1.6915e-06, -1]), 'episode_id': 5}
        # Fractal: {'scene_name': 'google_pick_coke_can_1_v4', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': 'recolor_tabletop_visual_matching_1', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png', 'rgb_overlay_cameras': ['overhead_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'model_id': 'opened_coke_can', 'model_scale': 1.0, 'distractor_model_ids': None, 'distractor_model_scales': None, 'obj_init_pose_wrt_robot_base': Pose([0.587925, -0.0238302, 0.840576], [0.707052, -0.0081018, -0.01162, -0.70702]), 'orientation': 'laid_vertically'} Instruction: pick coke can Max episode length: 80
        while 1:
            # infer action chunk
            inputs = self.env_adapter.preprocess(env, obs, instruction)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # running amp and bf16 will lead to ~1e-3 difference in action inferred when using vs. not using KV Cache, if starting from the same initial noise.
            # no difference when using float32, also slightly faster on L40
            with torch.no_grad():
                if self.use_bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        actions = self.model.infer_action(**inputs)[0]
                else:
                    actions = self.model.infer_action(**inputs)[0]
                # actions_naive = self.model.infer_action_naive(**inputs)[0]
            env_actions = self.env_adapter.postprocess(actions.cpu().numpy())

            # environment step
            for env_action in env_actions[: self.act_steps]:
                obs, reward, success, truncated, info = env.step(env_action)
                if truncated:
                    break

            # video
            if recording:
                video_writer.append_data(self.env_adapter.get_video_frame(env, obs))

            # update instruction, e.g., pick apple ---> put in top drawer
            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction

            # original octo eval only done when timeout, i.e., not upon success
            if truncated:
                successes.append(success)
                if recording:
                    video_writer.close()
                    if success:  # rename video with success
                        os.rename(
                            video_parent_path(cnt_episode) + ".mp4",
                            video_parent_path(cnt_episode) + "_success.mp4",
                        )
                cnt_episode += 1

                # quit
                if cnt_episode >= self.n_eval_episode:
                    break

                # reset
                env_reset_options["obj_init_options"] = {
                    "episode_id": cnt_episode,
                }
                obs, reset_info = env.reset(options=env_reset_options)
                env_adapter.reset()
                instruction = env.get_language_instruction()
                log.info(
                    f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
                )
                recording = self.n_video > cnt_episode
                if recording:
                    video_writer = imageio.get_writer(
                        video_parent_path(cnt_episode) + ".mp4"
                    )

        # summary
        success_rate = np.mean(successes)
        log.info("============ Evaluation Summary ============")
        log.info(f"Number of episodes: {cnt_episode}")
        log.info(f"Success rate: {success_rate}")
        log.info("============================================")

    @log_execution_time()
    def load_checkpoint(self, path):
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        self.model.load_state_dict(data["model"], strict=True)
        log.info(f"Loaded model from {path}")

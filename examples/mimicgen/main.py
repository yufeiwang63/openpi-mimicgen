"""
MimicGen evaluation with a single pi05 policy.

This script creates a MimicGen/ArticuBot environment (same interface as
third_party/mimicgen/equi_diffpo/env_runner/articubot_pcd_runner.py) and runs
episodes by querying a pi05 policy server (e.g. scripts/serve_policy.py) over
websocket. No high-level or low-level policy split—just one pi05 policy.

Usage:
  # Terminal 1: start the policy server (e.g. your fine-tuned pi05_mimicgen checkpoint)
  uv run scripts/serve_policy.py --policy.config=pi05_mimicgen --policy.dir=gs://openpi-assets/checkpoints/pi05_mimicgen 

  # Terminal 2: run this env script (ensure third_party/mimicgen is on PYTHONPATH)
  cd /path/to/openpi && PYTHONPATH=third_party/mimicgen_pi:$PYTHONPATH uv run examples/mimicgen/main.py \
    --dataset_path third_party/mimicgen_pi/datasets/core/square_d2.hdf5 \
    --video_out_path data/mimicgen/videos
"""

import collections
import dataclasses
import logging
import math
import pathlib
import sys

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from robosuite.utils import transform_utils as T
import time

# Add third_party/mimicgen so we can import env creation from equi_diffpo
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MIMICGEN_PATH = _REPO_ROOT / "third_party" / "mimicgen_pi"
if _MIMICGEN_PATH.is_dir():
    sys.path.insert(0, str(_MIMICGEN_PATH))

# Max steps per task (same as eval_smith.py)
MAX_STEPS_BY_TASK = {
    "stack_d1": 400,
    "stack_three_d1": 400,
    "square_d0": 400,
    "square_d2": 400,
    "threading_d2": 400,
    "coffee_d2": 400,
    "three_piece_assembly_d2": 500,
    "hammer_cleanup_d1": 500,
    "mug_cleanup_d1": 500,
    "kitchen_d1": 800,
    "nut_assembly_d0": 500,
    "pick_place_d0": 1000,
    "coffee_preparation_d1": 800,
    "tool_hang": 700,
    "can": 400,
    "lift": 400,
    "square": 400,
}

def _patch_headless_opencv_teardown() -> None:
    """Robosuite's OpenCVRenderer.close() calls cv2.destroyAllWindows(); headless OpenCV raises.

    Enable when using env.hard_reset (viewer destroyed each reset) on servers without GTK/Qt OpenCV.
    """
    import cv2

    _orig = cv2.destroyAllWindows

    def _safe_destroy_all_windows() -> None:
        try:
            _orig()
        except cv2.error:
            pass

    cv2.destroyAllWindows = _safe_destroy_all_windows


task_name_to_lang_description = {
    'stack_d1': 'Stack the blocks on top of each other',
    'stack_three_d1': 'Stack the three blocks on top of each other',
    'square_d0': 'Place the square block in the square hole',
    'square_d2': 'Place the square object into the pillar',
    'threading_d2': 'Thread the wire through the hole',
    'coffee_d2': 'Pour the coffee into the cup',
}


def _obs_to_policy_element(obs: dict, resize_size: int, task_name: str) -> dict:
    """Build the observation dict expected by pi05 MimicGen (observation/image, wrist, state, prompt)."""
    # MultiStepWrapper returns (n_obs_steps, ...); take latest
    def last(x):
        return x[-1] if x.ndim > 2 else x

    agentview = last(obs["agentview_image"])
    wrist = last(obs["robot0_eye_in_hand_image"])
    # CHW -> HWC if needed, then to uint8 and resize
    # import pdb; pdb.set_trace()
    if agentview.shape[0] == 3:
        agentview = agentview.transpose(1, 2, 0)
    if wrist.shape[0] == 3:
        wrist = wrist.transpose(1, 2, 0)
    if np.issubdtype(agentview.dtype, np.floating):
        agentview = (np.clip(agentview, 0, 1) * 255).astype(np.uint8)
    if np.issubdtype(wrist.dtype, np.floating):
        wrist = (np.clip(wrist, 0, 1) * 255).astype(np.uint8)

    # from matplotlib import pyplot as plt
    # plt.imshow(agentview)
    # plt.savefig("agentview_img.png")

    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(agentview, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, resize_size, resize_size)
    )

    # from matplotlib import pyplot as plt
    # plt.imshow(img)
    # plt.savefig("img.png")

    eef_pos = last(obs["robot0_eef_pos"]).flatten()
    eef_quat = last(obs["robot0_eef_quat"]).flatten()
    gripper_qpos = last(obs["robot0_gripper_qpos"]).flatten()
    eef_axis_angle = T.quat2axisangle(eef_quat)
    # MimicGen state: 8-d = eef_pos (3) + axis_angle (3) + gripper (2)
    state = np.concatenate([eef_pos, eef_axis_angle, gripper_qpos]).astype(np.float32)

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": task_name_to_lang_description[task_name],
    }


def _create_mimicgen_env(
    dataset_path: str,
    max_steps: int,
    seed: int,
    *,
    fps: int = 10,
    crf: int = 22,
    steps_per_render: int = 1,
    postprocess_visual_obs: bool = True,
):
    """Create a single MimicGen env (robomimic + wrappers). Uses same stack as articubot_pcd_runner.

    VideoRecordingWrapper records one frame per env step (steps_per_render=1) so the output
    MP4 is continuous. Set env.env.file_path before each episode to choose where the video is saved.

    postprocess_visual_obs: passed to EnvRobosuite via create_env -> create_env_from_metadata
    (see third_party/mimicgen_pi/external/robomimic/robomimic/envs/env_robosuite.py).
    """
    import collections
    import robomimic.utils.file_utils as FileUtils
    from equi_diffpo.env_runner.articubot_pcd_runner import create_env
    from equi_diffpo.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
    from equi_diffpo.gym_util.multistep_wrapper import MultiStepWrapper
    from equi_diffpo.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

    # Shape meta: same obs as ArticuBot runner; action dim 7 for pi05 (delta pos, axis-angle, gripper)
    shape_meta = {
        "obs": {
            "robot0_eye_in_hand_image": {"shape": [3, 512, 512], "type": "rgb"},
            "agentview_image": {"shape": [3, 512, 512], "type": "rgb"},
            "sideview_image": {"shape": [3, 512, 512], "type": "rgb"},
            "birdview_image": {"shape": [3, 512, 512], "type": "rgb"},
            "point_cloud": {"shape": [4500, 3], "type": "point_cloud"},
            "gripper_pcd": {"shape": [4, 3], "type": "point_cloud"},
            "robot0_eef_pos": {"shape": [3]},
            "robot0_eef_quat": {"shape": [4]},
            "robot0_gripper_qpos": {"shape": [2]},
            "state": {"shape": [10]},
        },
        "action": {"shape": [7]},
    }
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    # import pdb; pdb.set_trace()
    env_meta["env_kwargs"]["use_object_obs"] = False
    robomimic_env = create_env(
        env_meta=env_meta,
        shape_meta=shape_meta,
        is_eval=True,
        postprocess_visual_obs=postprocess_visual_obs,
    )
    
    
    # Extract task name from dataset_path (e.g., "square_d2" from ".../square_d2.hdf5")
    task_name = dataset_path.split("/")[-1].split(".")[0]
    robomimic_env.env.hard_reset = True if task_name in ["square_d2"] else False
    # robomimic_env.env.hard_reset = False 

    # Order: MultiStepWrapper(VideoRecordingWrapper(RobomimicImageWrapper(...)))
    # so that each step records one frame from the inner render (agentview).
    video_recorder = VideoRecorder.create_h264(
        fps=fps,
        codec="h264",
        input_pix_fmt="rgb24",
        crf=crf,
        thread_type="FRAME",
        thread_count=1,
    )

    wrapped = MultiStepWrapper(
        VideoRecordingWrapper(
            RobomimicImageWrapper(
                env=robomimic_env,
                shape_meta=shape_meta,
                init_state=None,
                render_obs_key="agentview_image",
            ),
            video_recoder=video_recorder,
            file_path=None,
            steps_per_render=steps_per_render,
        ),
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=max_steps,
    )
    # wrapped.seed(seed)
    return wrapped


@dataclasses.dataclass
class Args:
    # Policy server
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # MimicGen env
    dataset_path: str = "/data/robogen/smith_mimicgen/datasets/core/square_d2.hdf5"
    num_trials: int = 50
    seed: int = 10000
    # EnvRobosuite: if True, postprocess RGB (e.g. for network inputs); if False, raw images (e.g. dataset style)
    postprocess_visual_obs: bool = True

    # Output
    video_out_path: str = "data/mimicgen/videos"


def eval_mimicgen(args: Args) -> None:
    # np.random.seed(args.seed)
    _patch_headless_opencv_teardown()
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    task_name = pathlib.Path(args.dataset_path).stem
    max_steps = MAX_STEPS_BY_TASK.get(task_name, 400)
    logging.info("Task: %s, max_steps: %s", task_name, max_steps)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env = _create_mimicgen_env(
        args.dataset_path,
        max_steps,
        args.seed,
        postprocess_visual_obs=args.postprocess_visual_obs,
    )

    # Logging like articubot_pcd_runner: per-episode sim_max_reward_{seed}, then mean_score / max_score
    prefix = "test/"
    log_data = {}
    max_rewards = []

    for episode_idx in tqdm.tqdm(range(args.num_trials), desc="Episodes"):
        seed = args.seed + episode_idx
        env.seed(seed)
        out_name = pathlib.Path(args.video_out_path) / f"rollout_{task_name}_ep{episode_idx}.mp4"
        # VideoRecordingWrapper is env.env (inner of MultiStepWrapper). Set path so this episode is recorded.
        env.env.file_path = str(out_name)
        obs = env.reset()
        action_plan = collections.deque()
        done = False
        t = 0

        while t < max_steps:
            # print(f"t: {t}")
            # Latest obs (MultiStepWrapper with n_obs_steps=1 still returns shape (1,...))
            beg_t = time.time()
            if t % 20 == 0:
                print(f"t {t}")
            element = _obs_to_policy_element(obs, args.resize_size, task_name)

            if not action_plan:
                begin_time = time.time()
                response = client.infer(element)
                end_time = time.time()
                # print(f"time taken to infer: {end_time - begin_time}")
                action_chunk = response["actions"]
                # import pdb; pdb.set_trace()
                if len(action_chunk) < args.replan_steps:
                    logging.warning(
                        "Policy returned %d actions; wanted at least %d. Padding or replanning more often.",
                        len(action_chunk), args.replan_steps,
                    )
                action_plan.extend(action_chunk[: args.replan_steps])

            if not action_plan:
                logging.error("No actions in plan; stopping episode.")
                break

            action = action_plan.popleft()
            action_7 = np.asarray(action).flatten()[:7]
            # MultiStepWrapper expects (n_action_steps,) + action_shape = (1, 7)
            env_action = action_7.astype(np.float32).reshape(1, -1)
            # import pdb; pdb.set_trace()
            begin_time = time.time()
            obs, reward, done, info = env.step(env_action)
            end_time = time.time()
            # print(f"time taken to sim step: {end_time - begin_time}")
            t += 1

            end_t = time.time()
            # print(f"time taken to total step: {end_t - beg_t}")
            if done:
                break

        # Flush and close the video file (one frame was written per env step).
        env.render()

        # Success / score like articubot: max over per-step rewards from MultiStepWrapper
        rewards = env.get_rewards()
        max_reward = float(np.max(rewards)) if rewards else 0.0
        max_rewards.append(max_reward)
        log_data[prefix + f"sim_max_reward_{seed}"] = max_reward

        # Rename to include success/failure for easier inspection (success = max_reward >= 1 typically)
        final_name = pathlib.Path(args.video_out_path) / f"rollout_{task_name}_ep{episode_idx}_{'success' if max_reward >= 1.0 else 'failure'}.mp4"
        if out_name.exists() and out_name != final_name:
            out_name.rename(final_name)
        log_data[prefix + f"sim_video_{seed}"] = str(final_name)
        logging.info("Episode %d seed=%d sim_max_reward=%.4f video=%s", episode_idx, seed, max_reward, final_name)

    env.close()

    # Aggregate metrics (same names as articubot_pcd_runner)
    mean_score = float(np.mean(max_rewards)) if max_rewards else 0.0
    log_data[prefix + "mean_score"] = mean_score
    log_data[prefix + "max_score"] = float(np.max(max_rewards)) if max_rewards else 0.0

    logging.info("---------------- Eval Results --------------")
    for key, value in log_data.items():
        if isinstance(value, float):
            logging.info("%s: %.4f", key, value)
        else:
            logging.info("%s: %s", key, value)

    # Write eval_results.txt like eval_smith.py
    results_path = pathlib.Path(args.video_out_path) / "eval_results.txt"
    with open(results_path, "w") as f:
        for key, value in log_data.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    logging.info("Wrote %s", results_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_mimicgen)

"""
Evaluate action reconstruction error of a pi05 MimicGen policy on the training dataset.

This script loads a pi05_mimicgen (or pi05_mimicgen_lora) checkpoint—or the base pi05 model
with --use_base_model—runs inference on mimicgen/square_d2 data, and computes MSE/RMSE
between predicted and ground-truth actions. Use this to check overfitting or to compare
base vs fine-tuned.

Usage:
  # Fine-tuned checkpoint
  uv run examples/mimicgen/eval_action_reconstruction.py \
    --checkpoint_dir ./checkpoints/pi05_mimicgen/exp1/30000 \
    --config_name pi05_mimicgen \
    --max_samples 1000

  # Base pi05 (no fine-tuning). Pass norm stats from a fine-tuned ckpt or compute first:
  #   uv run scripts/compute_norm_stats.py --config-name pi05_mimicgen
  uv run examples/mimicgen/eval_action_reconstruction.py \
    --use_base_model \
    --config_name pi05_mimicgen \
    --norm_stats_dir ./checkpoints/pi05_mimicgen/exp1/30000/assets/mimicgen/square_d2 \
    --max_samples 1000
"""

import logging
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.shared.download as download
import openpi.shared.normalize as normalize
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# Base pi05 checkpoint (params + assets for norm stats); same as used for training pi05_mimicgen.
PI05_BASE_CHECKPOINT_DIR = "gs://openpi-assets/checkpoints/pi05_base"


def main(
    checkpoint_dir: str | None = None,
    config_name: str = "pi05_mimicgen",
    use_base_model: bool = False,
    norm_stats_dir: str | None = None,
    max_samples: int | None = 1000,
    seed: int = 42,
) -> None:
    # force=True so our config applies even if JAX/TF already configured logging
    logging.basicConfig(level=logging.INFO, force=True)
    if use_base_model:
        checkpoint_dir = PI05_BASE_CHECKPOINT_DIR
    elif checkpoint_dir is None:
        raise ValueError("Provide either --checkpoint_dir or --use_base_model")
    rng = np.random.default_rng(seed)

    # Optional: load precomputed norm stats (e.g. from a fine-tuned checkpoint's assets)
    norm_stats = None
    if norm_stats_dir is not None:
        norm_stats_path = pathlib.Path(download.maybe_download(norm_stats_dir))
        norm_stats = normalize.load(norm_stats_path)
        logging.info("Loaded norm stats from %s", norm_stats_path)

    # Load training config and policy
    train_config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(
        train_config, checkpoint_dir, norm_stats=norm_stats
    )
    action_horizon = train_config.model.action_horizon
    action_dim = 7  # MimicGen outputs 7-D actions

    # Build data config and dataset in policy-input format (no normalization; policy normalizes internally)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    base_dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, train_config.model
    )
    # Only repack so we get observation/image, observation/wrist_image, observation/state, prompt, actions (raw)
    dataset = _data_loader.TransformedDataset(
        base_dataset,
        list(data_config.repack_transforms.inputs),
    )

    n_total = len(dataset)
    n_eval = min(max_samples, n_total) if max_samples is not None else n_total
    indices = rng.choice(n_total, size=n_eval, replace=False)

    mse_list = []
    mse_per_step_list = []  # (H,) per sample

    for idx in tqdm.tqdm(indices, desc="Evaluating"):
        sample = dataset[int(idx)]
        # Policy expects observation/image, observation/wrist_image, observation/state, prompt
        obs = {
            "observation/image": sample["observation/image"],
            "observation/wrist_image": sample["observation/wrist_image"],
            "observation/state": sample["observation/state"],
            "prompt": sample["prompt"],
        }



        # Ensure numpy and correct dtypes for policy
        obs = {k: np.asarray(v) for k, v in obs.items()}

        out = policy.infer(obs)
        pred_actions = np.asarray(out["actions"])
        if pred_actions.ndim == 3:
            pred_actions = pred_actions.squeeze(0)
        gt_actions = np.asarray(sample["actions"])

        # Align shapes: both (action_horizon, action_dim)
        H, D = gt_actions.shape
        pred_actions = pred_actions[:H, :D]
        gt_actions = gt_actions[:H, :D]

        diff = pred_actions.astype(np.float64) - gt_actions.astype(np.float64)
        mse = float(np.mean(diff ** 2))
        mse_list.append(mse)
        mse_per_step_list.append(np.mean(diff ** 2, axis=1))

    mse_mean = float(np.mean(mse_list))
    mse_std = float(np.std(mse_list))
    mse_per_step = np.mean(mse_per_step_list, axis=0)

    # Print results so they always appear (logging can be swallowed by JAX/TF)
    lines = [
        "---------------- Action reconstruction (training set) ----------------",
        f"Config: {config_name}",
        f"Checkpoint: {checkpoint_dir}",
        f"Samples: {n_eval} / {n_total}",
        f"MSE (mean ± std): {mse_mean:.6f} ± {mse_std:.6f}",
        f"RMSE: {np.sqrt(mse_mean):.6f}",
        f"Per-step MSE (first 5 steps): {np.array2string(mse_per_step[:5], precision=6)}",
    ]
    if len(mse_per_step) > 5:
        lines.append(f"Per-step MSE (last 5 steps): {np.array2string(mse_per_step[-5:], precision=6)}")
    for line in lines:
        print(line)
        logging.info("%s", line)


if __name__ == "__main__":
    tyro.cli(main)

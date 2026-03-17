"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import os
import shutil

# LeRobot reads HF_LEROBOT_HOME at import time. Set in the shell to choose where datasets go, e.g.:
#   export HF_LEROBOT_HOME="/data/robogen/openpi/data"
# If unset, we use this default (must be set before importing lerobot).
if "HF_LEROBOT_HOME" not in os.environ:
    os.environ["HF_LEROBOT_HOME"] = "/data/robogen/openpi/data"
OUTPUT_ROOT = os.environ["HF_LEROBOT_HOME"]

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import numpy as np

REPO_NAME = "mimicgen/square_d2"  # Name of the output dataset, also used for the Hugging Face Hub

def main(data_dir: str, *, push_to_hub: bool = False):
    # Dataset will be created at OUTPUT_ROOT / REPO_NAME (LeRobot uses HF_LEROBOT_HOME / repo_id)
    output_path = os.path.join(OUTPUT_ROOT, REPO_NAME)

    print(f"output_path: {output_path}")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (512, 512, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (512, 512, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    # for raw_dataset_name in RAW_DATASET_NAMES:
    #     raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
    #     for episode in raw_dataset:
    #         for step in episode["steps"].as_numpy_iterator():
    #             dataset.add_frame(
    #                 {
    #                     "image": step["observation"]["image"],
    #                     "wrist_image": step["observation"]["wrist_image"],
    #                     "state": step["observation"]["state"],
    #                     "actions": step["action"],
    #                     "task": step["language_instruction"].decode(),
    #                 }
    #             )
    #         dataset.save_episode()

    mimicgen_data_path = "/data/robogen/smith_mimicgen/datasets/articubot_format/square_d2_pi"
    for ep_idx in range(100):
        ep_data_path = os.path.join(mimicgen_data_path, f"demo_{ep_idx}")
        num_timesteps = len(os.listdir(ep_data_path))
        for timestep_idx in range(num_timesteps):
            timestep_data_path = os.path.join(ep_data_path, f"{timestep_idx}.npz")
            data = np.load(timestep_data_path)

            action = data['action'][:][0].astype(np.float32)
            state = data['state'][:][0].astype(np.float32)
            image = data['image'][:][0]
            wrist_image = data['wrist_image'][:][0]
            task = str(data['task'])

            # import pdb; pdb.set_trace()

            dataset.add_frame(
                {
                    "image": image,
                    "wrist_image": wrist_image,
                    "state": state,
                    "actions": action,
                    "task": task,
                }
            )

        dataset.save_episode()


    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["mimicgen", "panda"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset

data_dir = "/data/robogen/openpi/data/modified_libero_rlds"
import tensorflow_datasets as tfds

for raw_dataset_name in RAW_DATASET_NAMES:
    raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
    for episode in raw_dataset:
        for step in episode["steps"].as_numpy_iterator():
            # Image is under observation in RLDS
            img = step["observation"]["image"]
            print(f"image dtype: {img.dtype}, shape: {img.shape}")
            break  # only check first step
        break  # only check first episode
    break  # only check first dataset
import os
import yaml

import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "settings.yaml")


def get_nfl_config():
    with open(CONFIG_PATH, mode="r") as f:
        config = yaml.safe_load(f)
    return config


def get_nfl_data_path():
    config = get_nfl_config()
    return config["NFL_DATA_PATH"]


def get_nfl_artifacts_path():
    config = get_nfl_config()
    return config["ARTIFACTS_PATH"]


def get_nfl_models_path():
    config = get_nfl_config()
    return config["MODELS_PATH"]


def get_nfl_output_path():
    config = get_nfl_config()
    return config["OUTPUT_PATH"]


def get_nfl_frames_path():
    artifacts_path = get_nfl_artifacts_path()
    return os.path.join(artifacts_path, "frames")


def get_nfl_images_path():
    data_path = get_nfl_data_path()
    return os.path.join(data_path, "images")


def get_nfl_image_labels_df():
    data_path = get_nfl_data_path()
    image_labels_df = pd.read_csv(os.path.join(data_path, "image_labels.csv"))
    return image_labels_df


def get_nfl_train_labels_df():
    data_path = get_nfl_data_path()
    train_labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    return train_labels_df


def get_nfl_train_images_split_path():
    artifacts_path = get_nfl_artifacts_path()
    return os.path.join(artifacts_path, "train_images_split.txt")


def get_nfl_train_videos_split_path():
    artifacts_path = get_nfl_artifacts_path()
    return os.path.join(artifacts_path, "train_videos_split.txt")


def get_nfl_val_images_split_path():
    artifacts_path = get_nfl_artifacts_path()
    return os.path.join(artifacts_path, "val_images_split.txt")


def get_nfl_val_videos_split_path():
    artifacts_path = get_nfl_artifacts_path()
    return os.path.join(artifacts_path, "val_videos_split.txt")


def get_nfl_train_images_coco_path():
    artifacts_path = get_nfl_artifacts_path()
    return os.path.join(artifacts_path, "train_images_coco.json")


def get_nfl_train_frames_coco_path():
    artifacts_path = get_nfl_artifacts_path()
    return os.path.join(artifacts_path, "train_frames_coco.json")


def get_nfl_val_images_coco_path():
    artifacts_path = get_nfl_artifacts_path()
    return os.path.join(artifacts_path, "val_images_coco.json")


def get_nfl_val_frames_coco_path():
    artifacts_path = get_nfl_artifacts_path()
    return os.path.join(artifacts_path, "val_frames_coco.json")


def get_nfl_rel_path_to_detections_path():
    artifacts_path = get_nfl_artifacts_path()
    return os.path.join(artifacts_path, "rel_path_to_detections.pkl")


if __name__ == "__main__":
    print(get_nfl_data_path())
    print(get_nfl_artifacts_path())
    print(get_nfl_models_path())
    print(get_nfl_frames_path())
    print(get_nfl_train_labels_df())
    print(get_nfl_image_labels_df())

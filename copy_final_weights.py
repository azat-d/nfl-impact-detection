import shutil
import json
import os

from nfl_io import get_nfl_artifacts_path, get_nfl_models_path

HELMET_DETECTOR_SRC_CONFIG_REL_PATH = "helmet_detector/config.yaml"
HELMET_DETECTOR_SRC_WEIGHTS_REL_PATH = "helmet_detector/model_final.pth"
HELMET_DETECTOR_CONFIG_REL_PATH = "helmet_detector/config.yaml"
HELMET_DETECTOR_WEIGHTS_REL_PATH = "helmet_detector/weights.pth"

IMPACT_DETECTOR_SRC_ROOT = "impact_detector"
IMPACT_DETECTOR_CONFIG_REL_PATH = "impact_detector/config.yaml"
IMPACT_DETECTOR_WEIGHTS_REL_PATH = "impact_detector/weights.pth"

IMPACT_CLASSIFIER_SRC_ROOT = "impact_classifier"
IMPACT_CLASSIFIER_CONFIG_REL_PATH = "impact_classifier/config.yaml"
IMPACT_CLASSIFIER_WEIGHTS_REL_PATH = "impact_classifier/weights.pth"

IMPACT_MODEL_CONFIG_FILE_NAME = "config.yaml"
IMPACT_MODEL_LOG_FILE_NAME = "stdout.log"
IMPACT_MODEL_CHECKPOINT_REL_PATH_TEMPLATE = "checkpoints/checkpoint_epoch_{:05d}.pyth"


def copy(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)


def copy_impact_model(model_src_root, dst_config_path, dst_weights_path):
    checkpoints = []
    with open(os.path.join(model_src_root, IMPACT_MODEL_LOG_FILE_NAME), "r") as f:
        for line in f:
            if "\"score\":" in line:
                parts = line.strip().split("json_stats: ")
                json_stats = json.loads(parts[1])
                checkpoint_rel_path = IMPACT_MODEL_CHECKPOINT_REL_PATH_TEMPLATE.format(int(json_stats["cur_epoch"]))
                score = json_stats["score"]
                checkpoints.append((checkpoint_rel_path, score))

    checkpoints = sorted(checkpoints, key=lambda item: item[1])
    best_checkpoint_rel_path, best_score = checkpoints[-1]
    print(f"Checkpoint {best_checkpoint_rel_path} selected with score {best_score}")

    copy(os.path.join(model_src_root, IMPACT_MODEL_CONFIG_FILE_NAME), dst_config_path)
    copy(os.path.join(model_src_root, best_checkpoint_rel_path), dst_weights_path)


def main():
    artifacts_path = get_nfl_artifacts_path()
    models_path = get_nfl_models_path()

    copy(os.path.join(artifacts_path, HELMET_DETECTOR_SRC_CONFIG_REL_PATH),
         os.path.join(models_path, HELMET_DETECTOR_CONFIG_REL_PATH))
    copy(os.path.join(artifacts_path, HELMET_DETECTOR_SRC_WEIGHTS_REL_PATH),
         os.path.join(models_path, HELMET_DETECTOR_WEIGHTS_REL_PATH))

    copy_impact_model(os.path.join(artifacts_path, IMPACT_DETECTOR_SRC_ROOT),
                      os.path.join(models_path, IMPACT_DETECTOR_CONFIG_REL_PATH),
                      os.path.join(models_path, IMPACT_DETECTOR_WEIGHTS_REL_PATH))

    copy_impact_model(os.path.join(artifacts_path, IMPACT_CLASSIFIER_SRC_ROOT),
                      os.path.join(models_path, IMPACT_CLASSIFIER_CONFIG_REL_PATH),
                      os.path.join(models_path, IMPACT_CLASSIFIER_WEIGHTS_REL_PATH))


if __name__ == "__main__":
    main()

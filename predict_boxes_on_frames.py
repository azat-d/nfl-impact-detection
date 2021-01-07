import glob
import os
import tqdm
import pickle
import argparse

import numpy as np

import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor

from nfl_io import get_nfl_frames_path, get_nfl_rel_path_to_detections_path


def main():
    parser = argparse.ArgumentParser(description="Predict boxes on frames")
    parser.add_argument("--cfg", type=str, metavar="FILE", required=True, help="Detector config path")
    parser.add_argument("--weights", type=str, metavar="FILE", required=True, help="Detector weights path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detector score threshold")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(["MODEL.WEIGHTS", args.weights])
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.threshold
    cfg.freeze()

    predictor = DefaultPredictor(cfg)

    frames_path = get_nfl_frames_path()
    content = sorted(glob.iglob(os.path.join(frames_path, "*", "*.jpg")))
    rel_path_to_detections = {}
    for path in tqdm.tqdm(content):
        rel_path = os.path.relpath(path, frames_path)
        img = read_image(path, format="BGR")
        predictions = predictor(img)["instances"].to(torch.device("cpu"))
        boxes = predictions.pred_boxes.tensor.numpy()
        scores = predictions.scores.numpy()
        detections = np.concatenate((boxes, scores[:, np.newaxis]), axis=1)
        rel_path_to_detections[rel_path] = detections

    with open(get_nfl_rel_path_to_detections_path(), "wb") as f:
        pickle.dump(rel_path_to_detections, f)


if __name__ == "__main__":
    main()

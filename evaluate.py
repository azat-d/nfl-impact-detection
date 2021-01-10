import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from nfl_io import get_nfl_train_labels_df
from utils import iou


def precision_calc(gt_boxes, pred_boxes):
    cost_matrix = np.ones((len(gt_boxes), len(pred_boxes)))
    for i, box1 in enumerate(gt_boxes):
        for j, box2 in enumerate(pred_boxes):
            dist = abs(box1[0] - box2[0])
            if dist > 4:
                continue
            iou_score = iou(box1[1:], box2[1:])

            if iou_score < 0.35:
                continue
            else:
                cost_matrix[i, j] = 0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    fn = len(gt_boxes) - row_ind.shape[0]
    fp = len(pred_boxes) - col_ind.shape[0]
    tp = 0
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] == 0:
            tp += 1
        else:
            fp += 1
            fn += 1
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser(description="Evaluates submission")
    parser.add_argument("--csv-path", type=str, metavar="FILE", required=True, help="Path to submission csv file")
    args = parser.parse_args()

    video_to_pred_boxes = defaultdict(list)
    predictions_df = pd.read_csv(args.csv_path)
    for _, _, _, _, video, frame, left, width, top, height in predictions_df.values:
        video_to_pred_boxes[video].append((frame - 1, left, top, left + width, top + height))
    videos = sorted(video_to_pred_boxes)

    labels_df = get_nfl_train_labels_df()
    video_to_gt_boxes = defaultdict(list)
    videos_set = set(videos)
    for _, _, _, video, frame, label, left, width, top, height, impact, _, confidence, visibility in labels_df.values:
        if video in videos_set and frame > 0 and impact == 1 and confidence > 1 and visibility > 0:
            video_to_gt_boxes[video].append((frame - 1, left, top, left + width, top + height))

    video_to_gt_boxes = {video: np.array(sorted(gt_boxes), dtype=np.float32) for video, gt_boxes in
                         video_to_gt_boxes.items()}

    ftp, ffp, ffn = [], [], []
    for video in videos:
        gt_boxes = video_to_gt_boxes[video]
        pred_boxes = video_to_pred_boxes[video]
        tp, fp, fn = precision_calc(gt_boxes, pred_boxes)
        ftp.append(tp)
        ffp.append(fp)
        ffn.append(fn)

    tp = np.sum(ftp)
    fp = np.sum(ffp)
    fn = np.sum(ffn)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, PRECISION: {precision:.4f}, RECALL: {recall:.4f}, F1 SCORE: {f1_score}")


if __name__ == "__main__":
    main()

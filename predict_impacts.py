import glob
import os
import tqdm
from collections import defaultdict
import argparse
import tempfile

import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset

from detectron2.config import get_cfg as get_detectron2_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor

from slowfast.config.defaults import get_cfg as get_slowfast_cfg
from slowfast.datasets import utils, cv2_transform
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model

from impact_data_config import add_impact_data_config
import video_model_builder

from copy_final_weights import HELMET_DETECTOR_CONFIG_REL_PATH, HELMET_DETECTOR_WEIGHTS_REL_PATH
from copy_final_weights import IMPACT_DETECTOR_CONFIG_REL_PATH, IMPACT_DETECTOR_WEIGHTS_REL_PATH
from copy_final_weights import IMPACT_CLASSIFIER_CONFIG_REL_PATH, IMPACT_CLASSIFIER_WEIGHTS_REL_PATH

from nfl_io import get_nfl_models_path
from utils import track_iou, add_center_mask, iou

DETECTOR_THRESHOLD = 0.5
MERGE_IOU_THRESHOLD = 0.95
# These values were tuned on validation
PRED_SIGMA_L = 0.2
PRED_SIGMA_H = 0.4
PRED_SIGMA_IOU = 0.5
PRED_T_MIN = 1


def extract_video_frames(data_root, frames_root):
    video_paths = sorted(glob.iglob(os.path.join(data_root, '*.mp4')))
    for video_path in tqdm.tqdm(video_paths):
        video_rel_path = os.path.relpath(video_path, data_root)
        dst_root = os.path.join(frames_root, video_rel_path)
        os.makedirs(dst_root)
        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            capture.grab()
            ret, frame = capture.retrieve()
            if not ret:
                continue
            cv2.imwrite(os.path.join(dst_root, f'{i}.jpg'), frame)


def predict_boxes_on_frames(frames_root):
    models_path = get_nfl_models_path()
    detector_config_path = os.path.join(models_path, HELMET_DETECTOR_CONFIG_REL_PATH)
    detector_weights_path = os.path.join(models_path, HELMET_DETECTOR_WEIGHTS_REL_PATH)

    cfg = get_detectron2_cfg()
    cfg.merge_from_file(detector_config_path)
    cfg.merge_from_list(['MODEL.WEIGHTS', detector_weights_path])
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = DETECTOR_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTOR_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = DETECTOR_THRESHOLD
    cfg.freeze()

    predictor = DefaultPredictor(cfg)

    content = sorted(glob.iglob(os.path.join(frames_root, '*', '*.jpg')))
    rel_path_to_detections = {}
    for path in tqdm.tqdm(content):
        rel_path = os.path.relpath(path, frames_root)
        img = read_image(path, format='BGR')
        predictions = predictor(img)['instances'].to(torch.device("cpu"))
        boxes = predictions.pred_boxes.tensor.numpy()
        scores = predictions.scores.numpy()
        detections = np.concatenate((boxes, scores[:, np.newaxis]), axis=1)
        rel_path_to_detections[rel_path] = detections

    return rel_path_to_detections


class FramesDataset(Dataset):
    def __init__(self, rel_path_to_detections, frames_root, cfg):
        self._frames_root = frames_root
        self.cfg = cfg
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.IMPACT_DATA.BGR

        self._max_buffer_size = self._video_length
        self._frame_buffer = {}

        video_to_frame_count = defaultdict(int)
        for rel_path in rel_path_to_detections:
            video = os.path.dirname(rel_path)
            video_to_frame_count[video] += 1
        videos = sorted(video_to_frame_count)

        self._image_paths = []
        self._detections = []
        self._video_idx_to_name = videos
        for video in videos:
            image_paths_per_video = []
            detections_per_video = []
            for frame in range(video_to_frame_count[video]):
                rel_path = os.path.join(video, f"{frame}.jpg")
                image_paths_per_video.append(rel_path)
                detections_per_video.append(rel_path_to_detections[rel_path])
            self._image_paths.append(image_paths_per_video)
            self._detections.append(detections_per_video)

        self._keyframe_indices = []

        for video_idx, image_paths_per_video in enumerate(self._image_paths):
            for frame_idx in range(len(image_paths_per_video)):
                self._keyframe_indices.append((video_idx, frame_idx))

    def __len__(self):
        return len(self._keyframe_indices)

    def _image_preprocessing_cv2(self, img):
        # Convert image to CHW keeping BGR order.
        img = cv2_transform.HWC2CHW(img)

        # Image [0, 255] -> [0, 1].
        img = img / 255.0

        img = np.ascontiguousarray(img.reshape((3, img.shape[1], img.shape[2]))).astype(np.float32)
        img = cv2_transform.color_normalization(img, np.array(self._data_mean, dtype=np.float32),
                                                np.array(self._data_std, dtype=np.float32))
        if not self._use_bgr:
            img = img[::-1, ...]
        img = np.ascontiguousarray(img)

        return img

    def _get_frame(self, video_idx, frame_idx):
        if (video_idx, frame_idx) in self._frame_buffer:
            return self._frame_buffer[(video_idx, frame_idx)]

        img_path = os.path.join(self._frames_root, self._image_paths[video_idx][frame_idx])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = self._image_preprocessing_cv2(img)
        self._frame_buffer[(video_idx, frame_idx)] = img
        if len(self._frame_buffer) >= self._max_buffer_size:
            min_key = min(self._frame_buffer)
            del self._frame_buffer[min_key]

        return img

    def __getitem__(self, idx):
        video_idx, center_idx = self._keyframe_indices[idx]
        center_detections = self._detections[video_idx][center_idx]
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )

        imgs = [self._get_frame(video_idx, frame_idx) for frame_idx in seq]
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )
        imgs = torch.from_numpy(imgs)
        imgs = utils.pack_pathway_output(self.cfg, imgs)

        return imgs, self._image_paths[video_idx][center_idx], center_detections


class CropsDataset(Dataset):
    def __init__(self, rel_path_to_detections, frames_root, cfg):
        self._frames_root = frames_root
        self.cfg = cfg
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._crop_size = cfg.DATA.TEST_CROP_SIZE
        self._max_buffer_size = self._video_length
        self._frame_buffer = {}

        video_to_frame_count = defaultdict(int)
        for rel_path in rel_path_to_detections:
            video = os.path.dirname(rel_path)
            video_to_frame_count[video] += 1
        videos = sorted(video_to_frame_count)

        self._image_paths = []
        self._detections = []
        self._video_idx_to_name = videos
        for video in videos:
            image_paths_per_video = []
            detections_per_video = []
            for frame in range(video_to_frame_count[video]):
                rel_path = os.path.join(video, f"{frame}.jpg")
                image_paths_per_video.append(rel_path)
                detections_per_frame = []
                for xmin, ymin, xmax, ymax, score in rel_path_to_detections[rel_path]:
                    detections_per_frame.append({
                        'bbox': (xmin, ymin, xmax, ymax),
                        'score': score
                    })
                detections_per_video.append(detections_per_frame)
            tracks = track_iou(detections_per_video, cfg.IMPACT_DATA.TRACKER.SIGMA_L, cfg.IMPACT_DATA.TRACKER.SIGMA_H,
                               cfg.IMPACT_DATA.TRACKER.SIGMA_IOU, cfg.IMPACT_DATA.TRACKER.T_MIN)
            frame_to_detections = defaultdict(list)
            for track_id, track in enumerate(tracks):
                start_frame = track['start_frame']
                for i, (xmin, ymin, xmax, ymax) in enumerate(track['bboxes']):
                    frame = start_frame + i
                    detection = (xmin, ymin, xmax, ymax, track_id)
                    frame_to_detections[frame].append(detection)

            detections_per_video = []
            for frame in range(video_to_frame_count[video]):
                if frame in frame_to_detections:
                    detections = np.array(frame_to_detections[frame], dtype=np.float32)
                else:
                    detections = np.empty(shape=(0, 5), dtype=np.float32)
                detections_per_video.append(detections)

            self._image_paths.append(image_paths_per_video)
            self._detections.append(detections_per_video)

        self._frame_indices = []

        for video_idx, image_paths_per_video in enumerate(self._image_paths):
            for frame_idx, rel_path in enumerate(image_paths_per_video):
                seq = list(range(frame_idx - self._seq_len // 2, frame_idx + self._seq_len // 2, self._sample_rate))
                if seq[0] < 0 or seq[-1] >= len(image_paths_per_video):
                    continue

                self._frame_indices.append((video_idx, frame_idx))

    def _get_crop_bbox(self, detection, frame_height, frame_width):
        xmin, ymin, xmax, ymax = detection[:4]
        width = xmax - xmin
        height = ymax - ymin
        xcenter = xmin + width / 2
        ycenter = ymin + height / 2
        size = max(width, height)
        crop_scale = self.cfg.IMPACT_DATA.TEST_CROP_SCALE
        width = size * crop_scale
        height = size * crop_scale
        xmin = xcenter - width / 2
        ymin = ycenter - height / 2
        xmax = xmin + width
        ymax = ymin + height

        xmin = max(int(xmin), 0)
        xmax = min(int(xmax), frame_width)
        ymin = max(int(ymin), 0)
        ymax = min(int(ymax), frame_height)

        return xmin, ymin, xmax, ymax

    def _get_crops(self, imgs, crop_bbox):
        xmin, ymin, xmax, ymax = crop_bbox
        crops = [img[ymin:ymax, xmin:xmax] for img in imgs]
        return crops

    def _preprocess(self, src_crops):
        dst_crops = np.empty(shape=(len(src_crops), self._crop_size, self._crop_size, src_crops[0].shape[2]),
                             dtype=np.float32)
        for i, value in enumerate(self._data_mean):
            dst_crops[:, :, :, i] = value

        src_height, src_width, _ = src_crops[0].shape
        src_size = max(src_height, src_width)
        scale = self._crop_size / src_size
        dst_height = min(round(scale * src_height), self._crop_size)
        dst_width = min(round(scale * src_width), self._crop_size)
        dst_xmin = (self._crop_size - dst_width) // 2
        dst_xmax = dst_xmin + dst_width
        dst_ymin = (self._crop_size - dst_height) // 2
        dst_ymax = dst_ymin + dst_height
        for i, src_crop in enumerate(src_crops):
            dst_crop = cv2.resize(src_crop, (dst_width, dst_height), interpolation=cv2.INTER_LINEAR)
            dst_crops[i, dst_ymin:dst_ymax, dst_xmin:dst_xmax] = dst_crop

        data_mean = np.array(self._data_mean, dtype=np.float32) * 255
        data_scale = 1.0 / (np.array(self._data_std, dtype=np.float32) * 255)

        dst_crops -= data_mean
        dst_crops *= data_scale
        dst_crops = dst_crops.transpose(3, 0, 1, 2)
        dst_crops = torch.from_numpy(dst_crops)
        return dst_crops

    def _get_frame(self, video_idx, frame_idx):
        if (video_idx, frame_idx) in self._frame_buffer:
            return self._frame_buffer[(video_idx, frame_idx)]

        img_path = os.path.join(self._frames_root, self._image_paths[video_idx][frame_idx])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.cfg.IMPACT_DATA.CENTER_MASK.ENABLED:
            detections_per_frame = self._detections[video_idx][frame_idx]
            img = add_center_mask([img], [detections_per_frame], self.cfg.IMPACT_DATA.CENTER_MASK.MIN_OVERLAP)[0]

        self._frame_buffer[(video_idx, frame_idx)] = img
        if len(self._frame_buffer) >= self._max_buffer_size:
            min_key = min(self._frame_buffer)
            del self._frame_buffer[min_key]

        return img

    def __len__(self):
        return len(self._frame_indices)

    def __getitem__(self, idx):
        video_idx, center_idx = self._frame_indices[idx]
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )

        imgs = [self._get_frame(video_idx, frame_idx) for frame_idx in seq]
        frame_height, frame_width, _ = imgs[0].shape
        crops = []
        center_detections = self._detections[video_idx][center_idx]
        for detection in center_detections:
            crop_bbox = self._get_crop_bbox(detection, frame_height, frame_width)
            crops_per_detection = self._get_crops(imgs, crop_bbox)
            if self.cfg.IMPACT_DATA.CENTER_MASK.ENABLED:
                track_detections = []
                for frame_idx in seq:
                    detections_per_frame = self._detections[video_idx][frame_idx].copy()
                    detections_per_frame[:, 0] -= crop_bbox[0]
                    detections_per_frame[:, 1] -= crop_bbox[1]
                    detections_per_frame[:, 2] -= crop_bbox[0]
                    detections_per_frame[:, 3] -= crop_bbox[1]
                    track_mask = detections_per_frame[:, 4] == detection[4]
                    track_detections.append(detections_per_frame[track_mask])
                crops_per_detection = add_center_mask(crops_per_detection, track_detections,
                                                      self.cfg.IMPACT_DATA.CENTER_MASK.MIN_OVERLAP)
                crops_per_detection = [crop[:, :, [0, 1, 2, 4, 3]] for crop in crops_per_detection]
            crops_per_detection = self._preprocess(crops_per_detection)
            crops_per_detection = utils.pack_pathway_output(self.cfg, crops_per_detection)
            crops.append(crops_per_detection)

        return crops, self._image_paths[video_idx][center_idx], center_detections


def predict_impacts_on_frames(frames_root, rel_path_to_detections):
    models_path = get_nfl_models_path()
    impact_detector_config_path = os.path.join(models_path, IMPACT_DETECTOR_CONFIG_REL_PATH)
    impact_detector_weights_path = os.path.join(models_path, IMPACT_DETECTOR_WEIGHTS_REL_PATH)

    impact_detector_cfg = get_slowfast_cfg()
    add_impact_data_config(impact_detector_cfg)
    impact_detector_cfg.merge_from_file(impact_detector_config_path)
    impact_detector_cfg.merge_from_list(['TRAIN.ENABLE', False,
                                         'TEST.ENABLE', True,
                                         'TEST.CHECKPOINT_FILE_PATH', impact_detector_weights_path,
                                         'NUM_GPUS', 1])
    impact_detector_cfg.freeze()

    dataset = FramesDataset(rel_path_to_detections, frames_root, impact_detector_cfg)
    impact_detector = build_model(impact_detector_cfg).eval()
    cu.load_test_checkpoint(impact_detector_cfg, impact_detector)

    rel_path_to_impact_detections = {}
    for i in tqdm.tqdm(range(len(dataset))):
        imgs, rel_path, detections = dataset[i]
        if len(detections) > 0:
            boxes = detections[:, :4]
            boxes = np.insert(boxes, 0, 0, 1)
            with torch.no_grad():
                imgs = [pathway_imgs.unsqueeze(0).cuda() for pathway_imgs in imgs]
                impact_scores = impact_detector(imgs, torch.from_numpy(boxes).cuda())
                impact_scores = impact_scores.cpu().numpy().flatten()
            impact_detections = np.concatenate((boxes[:, 1:], impact_scores[:, np.newaxis]), axis=1)
            rel_path_to_impact_detections[rel_path] = impact_detections
    return rel_path_to_impact_detections


def predict_impacts_on_crops(frames_root, rel_path_to_detections):
    models_path = get_nfl_models_path()
    impact_classifier_config_path = os.path.join(models_path, IMPACT_CLASSIFIER_CONFIG_REL_PATH)
    impact_classifier_weights_path = os.path.join(models_path, IMPACT_CLASSIFIER_WEIGHTS_REL_PATH)

    impact_classifier_cfg = get_slowfast_cfg()
    add_impact_data_config(impact_classifier_cfg)
    impact_classifier_cfg.merge_from_file(impact_classifier_config_path)
    impact_classifier_cfg.merge_from_list(['TRAIN.ENABLE', False,
                                           'TEST.ENABLE', True,
                                           'TEST.CHECKPOINT_FILE_PATH', impact_classifier_weights_path,
                                           'NUM_GPUS', 1])
    impact_classifier_cfg.freeze()

    dataset = CropsDataset(rel_path_to_detections, frames_root, impact_classifier_cfg)
    impact_classifier = build_model(impact_classifier_cfg).eval()
    cu.load_test_checkpoint(impact_classifier_cfg, impact_classifier)

    rel_path_to_action_detections = defaultdict(list)
    for i in tqdm.tqdm(range(len(dataset))):
        crops_batch, rel_path, detections = dataset[i]
        if len(crops_batch) == 0:
            continue
        crops_batch_slow = []
        crops_batch_fast = []
        for crops in crops_batch:
            assert len(crops) == 2
            crops_batch_slow.append(crops[0])
            crops_batch_fast.append(crops[1])
        crops_batch_slow = torch.stack(crops_batch_slow)
        crops_batch_fast = torch.stack(crops_batch_fast)
        with torch.no_grad():
            crops_batch_slow = crops_batch_slow.cuda()
            crops_batch_fast = crops_batch_fast.cuda()
            impact_scores = impact_classifier([crops_batch_slow, crops_batch_fast])
            flip_impact_scores = impact_classifier([crops_batch_slow.flip(4), crops_batch_fast.flip(4)])
            impact_scores = impact_scores.cpu().numpy().flatten()
            flip_impact_scores = flip_impact_scores.cpu().numpy().flatten()
        for detection, impact_score, flip_impact_score in zip(detections, impact_scores, flip_impact_scores):
            xmin, ymin, xmax, ymax, track_id = detection
            rel_path_to_action_detections[rel_path].append((xmin, ymin, xmax, ymax, impact_score, flip_impact_score))

    rel_path_to_action_detections = {rel_path: np.array(detections, dtype=np.float32) for rel_path, detections in
                                     rel_path_to_action_detections.items()}

    return rel_path_to_action_detections


def merge_action_detections(rel_path_to_frame_action_detections, rel_path_to_crop_action_detections):
    rel_path_to_merged_detections = defaultdict(list)
    for rel_path, frame_action_detections in rel_path_to_frame_action_detections.items():
        if rel_path in rel_path_to_crop_action_detections:
            for frame_action_detection in frame_action_detections:
                for crop_action_detection in rel_path_to_crop_action_detections[rel_path]:
                    if iou(frame_action_detection, crop_action_detection) > MERGE_IOU_THRESHOLD:
                        xmin, ymin, xmax, ymax, score1, score2 = crop_action_detection
                        score3 = frame_action_detection[4]
                        score = (score1 + score2 + score3) / 3
                        merged_detection = (xmin, ymin, xmax, ymax, score)
                        rel_path_to_merged_detections[rel_path].append(merged_detection)
                        break
    return rel_path_to_merged_detections


def get_video_to_pred_tracks(rel_path_to_detections, rel_path_to_impact_detections):
    video_to_frame_count = defaultdict(int)
    for rel_path in rel_path_to_detections:
        video = os.path.dirname(rel_path)
        video_to_frame_count[video] += 1

    video_to_pred_tracks = {}
    for video, frame_count in video_to_frame_count.items():
        detections = []
        for i in range(frame_count):
            rel_path = os.path.join(video, f'{i}.jpg')
            detections_per_frame = []
            if rel_path in rel_path_to_impact_detections:
                for bbox in rel_path_to_impact_detections[rel_path]:
                    detections_per_frame.append({
                        'bbox': bbox,
                        'score': bbox[4]
                    })
            detections.append(detections_per_frame)
        tracks = track_iou(detections, PRED_SIGMA_L, PRED_SIGMA_H, PRED_SIGMA_IOU, PRED_T_MIN)
        for track in tracks:
            track['bboxes'] = np.array(track['bboxes'], dtype=np.float32)
        video_to_pred_tracks[video] = tracks

    return video_to_pred_tracks


def get_box_from_pred_track(pred_track):
    start_frame = pred_track['start_frame']
    best_idx = np.argmax(pred_track['bboxes'][:, 4])
    xmin, ymin, xmax, ymax = pred_track['bboxes'][best_idx][:4].astype(np.int32)
    width = xmax - xmin
    height = ymax - ymin
    return start_frame + best_idx + 1, xmin, ymin, width, height


def get_predictions_df(video_to_pred_tracks):
    predictions = []
    for video, pred_tracks in video_to_pred_tracks.items():
        parts = video.split('_')
        game_key = int(parts[0])
        play_id = int(parts[1])
        view = parts[2].split('.')[0]
        for pred_track in pred_tracks:
            frame, left, top, width, height = get_box_from_pred_track(pred_track)
            predictions.append((game_key, play_id, view, video, frame, left, width, top, height))

    predictions_df = pd.DataFrame(data=predictions,
                                  columns=["gameKey", "playID", "view", "video", "frame", "left", "width", "top",
                                           "height"])

    return predictions_df


def main():
    parser = argparse.ArgumentParser(description="Predicts impacts on video files")
    parser.add_argument("--videos-root", type=str, metavar="DIR", required=True,
                        help="The path to the folder with video files")
    parser.add_argument("--output-csv-path", type=str, metavar="FILE", required=True,
                        help="Where to save predictions (csv)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as frames_root:
        extract_video_frames(args.videos_root, frames_root)
        rel_path_to_detections = predict_boxes_on_frames(frames_root)
        rel_path_to_frame_impact_detections = predict_impacts_on_frames(frames_root, rel_path_to_detections)
        rel_path_to_crop_impact_detections = predict_impacts_on_crops(frames_root, rel_path_to_detections)
        rel_path_to_impact_detections = merge_action_detections(rel_path_to_frame_impact_detections,
                                                                rel_path_to_crop_impact_detections)
        video_to_pred_tracks = get_video_to_pred_tracks(rel_path_to_detections, rel_path_to_impact_detections)
        predictions_df = get_predictions_df(video_to_pred_tracks)
        predictions_df.to_csv(args.output_csv_path)


if __name__ == '__main__':
    main()

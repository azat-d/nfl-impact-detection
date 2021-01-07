import pickle
import os
import random
from collections import defaultdict

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from slowfast.datasets import utils
from slowfast.datasets.build import DATASET_REGISTRY

from utils import ColorAugmenter, track_iou, iou, add_center_mask
from nfl_io import get_nfl_frames_path
from nfl_io import get_nfl_train_labels_df
from nfl_io import get_nfl_train_videos_split_path, get_nfl_val_videos_split_path
from nfl_io import get_nfl_rel_path_to_detections_path


@DATASET_REGISTRY.register()
class ImpactCropDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._frames_root = get_nfl_frames_path()
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._use_bgr = cfg.IMPACT_DATA.BGR
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._use_color_augmentation = cfg.IMPACT_DATA.TRAIN_USE_COLOR_AUGMENTATION
            if self._use_color_augmentation:
                self._color_augmenter = ColorAugmenter()
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE

        with open(get_nfl_rel_path_to_detections_path(), "rb") as f:
            rel_path_to_detections = pickle.load(f)

        if split == "train":
            split_path = get_nfl_train_videos_split_path()
        else:
            assert split == "val", split
            split_path = get_nfl_val_videos_split_path()
        with open(split_path, "r") as f:
            videos = [line.strip() for line in f]

        video_to_frame_count = defaultdict(int)
        for rel_path in rel_path_to_detections:
            video = os.path.dirname(rel_path)
            video_to_frame_count[video] += 1

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

        labels_df = get_nfl_train_labels_df()
        rel_path_to_impacts = defaultdict(list)
        rel_path_to_label_to_bbox = defaultdict(dict)
        for _, _, _, video, frame, label, left, width, top, height, impact, _, confidence, visibility in labels_df.values:
            rel_path = os.path.join(video, f"{frame}.jpg")
            if frame > 0 and impact == 1 and confidence > 1 and visibility > 0:
                rel_path_to_impacts[rel_path].append((left, top, left + width, top + height, label))
            rel_path_to_label_to_bbox[rel_path][label] = (left, top, left + width, top + height)

        rel_path_to_extended_impacts = defaultdict(list)
        for rel_path, impacts in rel_path_to_impacts.items():
            video = os.path.dirname(rel_path)
            frame = int(os.path.basename(rel_path).split(".")[0])
            for _, _, _, _, label in impacts:
                for i in range(frame - cfg.IMPACT_DATA.IMPACT_EXTENSION_DELTA, frame + cfg.IMPACT_DATA.IMPACT_EXTENSION_DELTA + 1):
                    extended_rel_path = os.path.join(video, f"{i}.jpg")
                    if extended_rel_path in rel_path_to_label_to_bbox:
                        label_to_bbox = rel_path_to_label_to_bbox[extended_rel_path]
                        if label in label_to_bbox:
                            rel_path_to_extended_impacts[extended_rel_path].append(label_to_bbox[label])

        self._crop_indices = []
        self._crop_labels = []

        for video_idx, (image_paths_per_video, detections_per_video) in enumerate(
                zip(self._image_paths, self._detections)):
            for frame_idx, (rel_path, detections_per_frame) in enumerate(
                    zip(image_paths_per_video, detections_per_video)):
                seq = list(range(frame_idx - self._seq_len // 2, frame_idx + self._seq_len // 2, self._sample_rate))
                if seq[0] < 0 or seq[-1] >= len(image_paths_per_video):
                    continue

                for detection_idx, detection in enumerate(detections_per_frame):
                    label = 0
                    if rel_path in rel_path_to_extended_impacts:
                        for impact_bbox in rel_path_to_extended_impacts[rel_path]:
                            if iou(detection[:4], impact_bbox) > cfg.IMPACT_DATA.GT_IOU_THRESH:
                                label = 1
                    self._crop_indices.append((video_idx, frame_idx, detection_idx))
                    self._crop_labels.append(label)

        if split == "train" and cfg.IMPACT_DATA.IMPACT_RATIO is not None:
            num_impacts = sum(self._crop_labels)
            impact_indices = [i for i, label in enumerate(self._crop_labels) if label == 1]
            impact_idx = -1
            while num_impacts / len(self._crop_indices) < cfg.IMPACT_DATA.IMPACT_RATIO:
                impact_idx = (impact_idx + 1) % len(impact_indices)
                self._crop_indices.append(self._crop_indices[impact_indices[impact_idx]])
                self._crop_labels.append(self._crop_labels[impact_indices[impact_idx]])
                num_impacts += 1

    def _get_crop_bbox(self, detection, frame_height, frame_width):
        xmin, ymin, xmax, ymax = detection[:4]
        width = xmax - xmin
        height = ymax - ymin
        xcenter = xmin + width / 2
        ycenter = ymin + height / 2
        size = max(width, height)
        if self._split == "train":
            crop_scale = random.uniform(*self.cfg.IMPACT_DATA.TRAIN_JITTER_CROP_SCALES)
        else:
            crop_scale = self.cfg.IMPACT_DATA.TEST_CROP_SCALE
        width = size * crop_scale
        height = size * crop_scale
        if self._split == "train":
            xcenter += random.uniform(*self.cfg.IMPACT_DATA.TRAIN_JITTER_CROP_CENTER) * width
            ycenter += random.uniform(*self.cfg.IMPACT_DATA.TRAIN_JITTER_CROP_CENTER) * height
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
        if self._split == "train" and self._use_color_augmentation and random.choice([False, True]):
            src_color_crops = [src_crop[:, :, :3].astype(np.uint8) for src_crop in src_crops]
            src_color_crops = self._color_augmenter(src_color_crops)
            for src_crop, src_color_crop in zip(src_crops, src_color_crops):
                src_crop[:, :, :3] = src_color_crop
        for i, src_crop in enumerate(src_crops):
            dst_crop = cv2.resize(src_crop, (dst_width, dst_height), interpolation=cv2.INTER_LINEAR)
            dst_crops[i, dst_ymin:dst_ymax, dst_xmin:dst_xmax] = dst_crop

        if self._split == "train" and self.random_horizontal_flip and random.choice([False, True]):
            dst_crops = dst_crops[:, :, ::-1].copy()

        data_mean = np.array(self._data_mean, dtype=np.float32) * 255
        data_scale = 1.0 / (np.array(self._data_std, dtype=np.float32) * 255)

        dst_crops -= data_mean
        dst_crops *= data_scale
        dst_crops = dst_crops.transpose(3, 0, 1, 2)
        dst_crops = torch.from_numpy(dst_crops)
        return dst_crops

    def __len__(self):
        return len(self._crop_indices)

    def __getitem__(self, idx):
        video_idx, center_idx, detection_idx = self._crop_indices[idx]
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )
        label = self._crop_labels[idx]
        detection = self._detections[video_idx][center_idx][detection_idx]
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        image_paths = [os.path.join(self._frames_root, image_path) for image_path in image_paths]

        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.IMPACT_DATA.IMG_PROC_BACKEND
        )
        if not self._use_bgr:
            imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

        frame_height, frame_width, _ = imgs[0].shape
        crop_bbox = self._get_crop_bbox(detection, frame_height, frame_width)
        crops = self._get_crops(imgs, crop_bbox)

        if self.cfg.IMPACT_DATA.CENTER_MASK.ENABLED:
            detections = []
            track_detections = []
            for frame in seq:
                detections_per_frame = self._detections[video_idx][frame].copy()
                detections_per_frame[:, 0] -= crop_bbox[0]
                detections_per_frame[:, 1] -= crop_bbox[1]
                detections_per_frame[:, 2] -= crop_bbox[0]
                detections_per_frame[:, 3] -= crop_bbox[1]
                detections.append(detections_per_frame)
                track_mask = detections_per_frame[:, 4] == detection[4]
                track_detections.append(detections_per_frame[track_mask])
            crops = add_center_mask(crops, track_detections, self.cfg.IMPACT_DATA.CENTER_MASK.MIN_OVERLAP)
            crops = add_center_mask(crops, detections, self.cfg.IMPACT_DATA.CENTER_MASK.MIN_OVERLAP)

        assert self.cfg.IMPACT_DATA.IMG_PROC_BACKEND == "cv2"
        crops = self._preprocess(crops)
        crops = utils.pack_pathway_output(self.cfg, crops)

        label = np.array([label], dtype=np.float32)

        return crops, label, idx, {}

import pickle
import os
import random
from collections import defaultdict

import numpy as np

import torch
from torch.utils.data import Dataset

from slowfast.datasets import utils, cv2_transform
from slowfast.datasets.build import DATASET_REGISTRY

from utils import ColorAugmenter, iou
from nfl_io import get_nfl_frames_path
from nfl_io import get_nfl_train_labels_df
from nfl_io import get_nfl_train_videos_split_path, get_nfl_val_videos_split_path
from nfl_io import get_nfl_rel_path_to_detections_path


class ImpactDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._frames_root = get_nfl_frames_path()
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.IMPACT_DATA.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
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
                detections_per_video.append(rel_path_to_detections[rel_path])
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

        self._keyframe_indices = []
        self._keyframe_boxes_and_labels = []

        for video_idx, image_paths_per_video in enumerate(self._image_paths):
            for frame_idx, rel_path in enumerate(image_paths_per_video):
                seq = list(range(frame_idx - self._seq_len // 2, frame_idx + self._seq_len // 2, self._sample_rate))
                if seq[0] < 0 or seq[-1] >= len(image_paths_per_video):
                    continue

                detections = rel_path_to_detections[rel_path]
                boxes = []
                labels = []
                for detection in detections:
                    if detection[4] < cfg.IMPACT_DATA.DETECTION_SCORE_THRESH:
                        continue
                    label = 0
                    if rel_path in rel_path_to_extended_impacts:
                        for impact_bbox in rel_path_to_extended_impacts[rel_path]:
                            if iou(detection[:4], impact_bbox) > cfg.IMPACT_DATA.GT_IOU_THRESH:
                                label = 1
                    boxes.append(detection[:4])
                    labels.append(label)

                if len(boxes) == 0:
                    continue

                self._keyframe_indices.append((video_idx, frame_idx))
                self._keyframe_boxes_and_labels.append((boxes, labels))

        if split == "train" and cfg.IMPACT_DATA.IMPACT_RATIO is not None:
            impact_indices = []
            for i, (boxes, labels) in enumerate(self._keyframe_boxes_and_labels):
                if len(labels) > 0 and sum(labels) > 0:
                    impact_indices.append(i)
            num_impacts = len(impact_indices)
            idx = -1
            while num_impacts / len(self._keyframe_indices) < cfg.IMPACT_DATA.IMPACT_RATIO:
                idx = (idx + 1) % len(impact_indices)
                self._keyframe_indices.append(self._keyframe_indices[impact_indices[idx]])
                self._keyframe_boxes_and_labels.append(self._keyframe_boxes_and_labels[impact_indices[idx]])
                num_impacts += 1

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.
        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.
        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "val":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        if self._split == "train" and self._use_color_augmentation and random.choice([False, True]):
            imgs = self._color_augmenter(imgs)

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        return imgs, boxes

    def __len__(self):
        return len(self._keyframe_indices)

    def __getitem__(self, idx):
        video_idx, center_idx = self._keyframe_indices[idx]
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )
        boxes, labels = self._keyframe_boxes_and_labels[idx]
        assert len(boxes) > 0
        boxes = np.array(boxes)
        labels = np.array(labels)[:, np.newaxis]
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        image_paths = [os.path.join(self._frames_root, image_path) for image_path in image_paths]

        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.IMPACT_DATA.IMG_PROC_BACKEND
        )

        height, width, _ = imgs[0].shape
        boxes[:, [0, 2]] /= width
        boxes[:, [1, 3]] /= height
        ori_boxes = boxes.copy()

        assert self.cfg.IMPACT_DATA.IMG_PROC_BACKEND == "cv2"
        # Preprocess images and boxes
        imgs, boxes = self._images_and_boxes_preprocessing_cv2(
            imgs, boxes=boxes
        )

        imgs = utils.pack_pathway_output(self.cfg, imgs)

        metadata = [[video_idx, center_idx]] * len(boxes)

        extra_data = {
            "boxes": boxes,
            "ori_boxes": ori_boxes,
            "metadata": metadata,
        }

        return imgs, labels, idx, extra_data


DATASET_REGISTRY._do_register("Impactdataset", ImpactDataset)

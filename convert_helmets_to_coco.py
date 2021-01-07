import tqdm
import os
import json
from collections import defaultdict
import cv2

from nfl_io import get_nfl_image_labels_df, get_nfl_train_labels_df
from nfl_io import get_nfl_train_images_split_path, get_nfl_train_videos_split_path
from nfl_io import get_nfl_val_images_split_path, get_nfl_val_videos_split_path
from nfl_io import get_nfl_images_path, get_nfl_frames_path
from nfl_io import get_nfl_train_images_coco_path, get_nfl_train_frames_coco_path
from nfl_io import get_nfl_val_images_coco_path, get_nfl_val_frames_coco_path


def load_split(split_path):
    with open(split_path, "r") as f:
        split = set([line.strip() for line in f])
    return split


def save_coco(root, path_to_bboxes, paths, dst_path):
    categories = [{"id": 0, "name": "Helmet", "supercategory": "Helmet"}]
    annotations = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    for rel_path in tqdm.tqdm(sorted(paths)):
        bboxes = path_to_bboxes[rel_path]
        img = cv2.imread(os.path.join(root, rel_path))
        height, width, channels = img.shape

        image_id = len(annotations["images"])
        image = {
            "file_name": rel_path,
            "height": height,
            "width": width,
            "id": image_id
        }
        annotations["images"].append(image)

        for bbox_x, bbox_y, bbox_width, bbox_height in bboxes:
            annotation = {
                "id": len(annotations["annotations"]),
                "image_id": image_id,
                "category_id": 0,
                "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                "segmentation": [
                    [bbox_x, bbox_y, bbox_x + bbox_width, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height, bbox_x,
                     bbox_y + bbox_height]],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            }
            annotations["annotations"].append(annotation)

    with open(dst_path, "w") as f:
        json.dump(annotations, f)


def main():
    train_images_split = load_split(get_nfl_train_images_split_path())
    train_videos_split = load_split(get_nfl_train_videos_split_path())
    val_images_split = load_split(get_nfl_val_images_split_path())
    val_videos_split = load_split(get_nfl_val_videos_split_path())

    image_labels_df = get_nfl_image_labels_df()

    image_to_bboxes = defaultdict(list)
    for image_path, _, left, width, top, height in image_labels_df.values:
        image_to_bboxes[image_path].append((left, top, width, height))

    train_labels_df = get_nfl_train_labels_df()

    train_frames_split = set()
    val_frames_split = set()
    frame_to_bboxes = defaultdict(list)
    for _, _, _, video, frame, _, left, width, top, height, _, _, _, _ in train_labels_df.values:
        if frame > 0:
            frame_path = os.path.join(video, f"{frame - 1}.jpg")
            frame_to_bboxes[frame_path].append((left, top, width, height))
            if video in train_videos_split:
                train_frames_split.add(frame_path)
            else:
                assert video in val_videos_split, video
                val_frames_split.add(frame_path)

    save_coco(get_nfl_images_path(), image_to_bboxes, train_images_split, get_nfl_train_images_coco_path())
    save_coco(get_nfl_images_path(), image_to_bboxes, val_images_split, get_nfl_val_images_coco_path())
    save_coco(get_nfl_frames_path(), frame_to_bboxes, train_frames_split, get_nfl_train_frames_coco_path())
    save_coco(get_nfl_frames_path(), frame_to_bboxes, val_frames_split, get_nfl_val_frames_coco_path())


if __name__ == "__main__":
    main()

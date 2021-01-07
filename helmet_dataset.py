from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

from nfl_io import get_nfl_images_path, get_nfl_frames_path
from nfl_io import get_nfl_train_images_coco_path, get_nfl_train_frames_coco_path
from nfl_io import get_nfl_val_images_coco_path, get_nfl_val_frames_coco_path


def get_helmet_metadata():
    meta = {
        "thing_classes": ["Helmet"],
    }
    return meta


DatasetCatalog.register("helmet_train_images", lambda: load_coco_json(get_nfl_train_images_coco_path(),
                                                                      get_nfl_images_path()))
DatasetCatalog.register("helmet_val_images", lambda: load_coco_json(get_nfl_val_images_coco_path(),
                                                                    get_nfl_images_path()))
DatasetCatalog.register("helmet_train_frames", lambda: load_coco_json(get_nfl_train_frames_coco_path(),
                                                                      get_nfl_frames_path()))
DatasetCatalog.register("helmet_val_frames", lambda: load_coco_json(get_nfl_val_frames_coco_path(),
                                                                    get_nfl_frames_path()))
MetadataCatalog.get("helmet_train_images").set(json_file=get_nfl_train_images_coco_path(),
                                               image_root=get_nfl_images_path(), **get_helmet_metadata())
MetadataCatalog.get("helmet_val_images").set(json_file=get_nfl_val_images_coco_path(), image_root=get_nfl_images_path(),
                                             **get_helmet_metadata())
MetadataCatalog.get("helmet_train_frames").set(json_file=get_nfl_train_frames_coco_path(),
                                               image_root=get_nfl_frames_path(), **get_helmet_metadata())
MetadataCatalog.get("helmet_val_frames").set(json_file=get_nfl_val_frames_coco_path(), image_root=get_nfl_frames_path(),
                                             **get_helmet_metadata())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    args = parser.parse_args()

    import cv2

    from detectron2.utils.visualizer import Visualizer
    from detectron2.utils.logger import setup_logger

    logger = setup_logger(name=__name__)

    dataset = DatasetCatalog.get(args.dataset_name)
    logger.info("Done loading {} samples.".format(len(dataset)))
    meta = MetadataCatalog.get(args.dataset_name)

    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    for image_anno in dataset:
        img = cv2.imread(image_anno["file_name"])
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(image_anno)
        cv2.imshow("demo", vis.get_image())
        cv2.waitKey()

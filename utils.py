import math
import random

from PIL import Image
import numpy as np

from timm.data.transforms_factory import transforms_imagenet_train
from timm.data import auto_augment

AUG_FN_BLACK_LIST = [
    auto_augment.rotate,
    auto_augment.shear_x,
    auto_augment.shear_y,
    auto_augment.translate_x_abs,
    auto_augment.translate_x_rel,
    auto_augment.translate_y_abs,
    auto_augment.translate_y_rel
]


class ColorAugmenter(object):
    def __init__(self):
        _, rand_augment, _ = transforms_imagenet_train(auto_augment="original-mstd0.5", separate=True)
        filtered_policy = []
        # keep only color augmentation
        for sub_policy in rand_augment.transforms[0].policy:
            keep = True
            for op in sub_policy:
                if op.aug_fn in AUG_FN_BLACK_LIST:
                    keep = False
            if keep:
                filtered_policy.append(sub_policy)
        rand_augment.transforms[0].policy = filtered_policy
        self._rand_augment = rand_augment

    def __call__(self, imgs):
        prev_state = random.getstate()
        aug_imgs = []
        for img in imgs:
            random.setstate(prev_state)
            img = Image.fromarray(img.astype(np.uint8))
            img = self._rand_augment(img)
            img = np.array(img).astype(img.dtype)
            aug_imgs.append(img)
        return aug_imgs


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.
    Args:
         detections (list): list of detections per frame, usually generated by util.load_mot
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.
    Returns:
        list: list of tracks.
    """

    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))
                if iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = max(track['max_score'], best_match['score'])

                    updated_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished


def _get_gaussian_radius(width, height, min_overlap):
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def _gaussian2d(radius, sigma=1):
    # m, n = [(s - 1.) / 2. for s in shape]
    m, n = radius
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
    return gauss


def _draw_gaussian(fmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = _gaussian2d((radius, radius), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = fmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_fmap = fmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
        masked_fmap = np.maximum(masked_fmap, masked_gaussian * k)
        fmap[y - top:y + bottom, x - left:x + right] = masked_fmap


# Taken from CenterNet
def add_center_mask(imgs, detections, min_overlap):
    imgs_with_masks = []
    for img, detections_per_img in zip(imgs, detections):
        mask = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.float32)
        for xmin, ymin, xmax, ymax, _ in detections_per_img:
            width = xmax - xmin
            height = ymax - ymin
            xcenter = int((xmin + xmax) / 2)
            ycenter = int((ymin + ymax) / 2)
            radius = _get_gaussian_radius(width, height, min_overlap)
            radius = int(radius)
            _draw_gaussian(mask, (xcenter, ycenter), radius)
        mask = mask * 255
        img_with_mask = np.concatenate((img.astype(np.float32), mask[:, :, np.newaxis]), axis=2)
        imgs_with_masks.append(img_with_mask)
    return imgs_with_masks

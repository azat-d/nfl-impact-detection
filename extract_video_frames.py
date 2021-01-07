import glob
import os
import tqdm

import cv2

from nfl_io import get_nfl_data_path, get_nfl_frames_path


def main():
    nfl_data_path = get_nfl_data_path()
    nfl_frames_path = get_nfl_frames_path()
    video_paths = sorted(glob.iglob(os.path.join(nfl_data_path, "train", "*.mp4")))
    for video_path in tqdm.tqdm(video_paths):
        video_name = os.path.basename(video_path)
        dst_root = os.path.join(nfl_frames_path, video_name)
        os.makedirs(dst_root)
        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            capture.grab()
            ret, frame = capture.retrieve()
            if not ret:
                continue
            cv2.imwrite(os.path.join(dst_root, f"{i}.jpg"), frame)


if __name__ == '__main__':
    main()

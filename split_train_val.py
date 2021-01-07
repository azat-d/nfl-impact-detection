import random
from collections import defaultdict

from nfl_io import get_nfl_image_labels_df, get_nfl_train_labels_df
from nfl_io import get_nfl_train_images_split_path, get_nfl_train_videos_split_path
from nfl_io import get_nfl_val_images_split_path, get_nfl_val_videos_split_path

SPLIT_SEED = 0xDEADFACE
VAL_RATIO = 0.2


def save(game_ids, game_id_to_paths, dst_path):
    with open(dst_path, 'w') as f:
        for game_id in sorted(game_ids):
            for path in sorted(game_id_to_paths[game_id]):
                f.write(f'{path}\n')


def main():
    image_labels_df = get_nfl_image_labels_df()
    images = sorted(set(image_labels_df["image"].values.tolist()))

    game_id_to_images = defaultdict(list)
    for image in images:
        game_id = image.split("_")[0]
        game_id_to_images[game_id].append(image)

    train_labels_df = get_nfl_train_labels_df()
    videos = sorted(set(train_labels_df["video"].values.tolist()))

    game_id_to_videos = defaultdict(list)
    for video in videos:
        game_id = video.split("_")[0]
        game_id_to_videos[game_id].append(video)

    game_ids = sorted(set(game_id_to_images).union(set(game_id_to_videos)))
    val_size = round(len(game_ids) * VAL_RATIO)
    random.seed(SPLIT_SEED)
    random.shuffle(game_ids)
    val_game_ids, train_game_ids = game_ids[:val_size], game_ids[val_size:]

    save(train_game_ids, game_id_to_images, get_nfl_train_images_split_path())
    save(train_game_ids, game_id_to_videos, get_nfl_train_videos_split_path())

    save(val_game_ids, game_id_to_images, get_nfl_val_images_split_path())
    save(val_game_ids, game_id_to_videos, get_nfl_val_videos_split_path())


if __name__ == "__main__":
    main()

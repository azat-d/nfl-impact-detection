# NFL 1st and Future - Impact Detection
4th place solution for the [NFL 1st and Future - Impact Detection](https://www.kaggle.com/c/nfl-impact-detection).  
Private LB score: **0.6355**

## The hardware I used
- CPU: AMD EPYC 7402P 24-Core Processor
- GPU: 4x GeForce RTX 3090
- RAM: 128 GB
- SSD: 3.4 TB

## Prerequisites

### Environment
Use the docker to get an environment close to what was used in the training. Run the following command to build the docker image:
```bash
cd path/to/solution
sudo docker build -t nfl .
```

### Data
Download the [nfl-impact-detection-data](https://www.kaggle.com/c/nfl-impact-detection/data) and extract all files to `/path/to/nfl-data`. This directory must have the following structure:
```
nfl-data/
├── image_labels.csv
├── images
├── nflimpact
├── sample_submission.csv
├── test
├── test_player_tracking.csv
├── train
├── train_labels.csv
└── train_player_tracking.csv
```

### Output directories
Create a directory for training artifacts (extracted frames, checkpoints, logs, etc)
```bash
mkdir -p /path/to/artifacts
```
Create a directory for final weights
```bash
mkdir -p /path/to/models
```
Create a directory for the model output (submissions)
```bash
mkdir -p /path/to/output
```

## How to train the model
Run the docker container with the paths correctly mounted:
```bash
sudo docker run --gpus=all -i -t -d --rm --ipc=host -v /path/to/nfl-data:/kaggle/input/nfl-impact-detection:ro -v /path/to/solution:/kaggle/solution -v /path/to/artifacts:/kaggle/artifacts -v /path/to/models:/kaggle/models -v /path/to/output:/kaggle/output --name nfl nfl
sudo docker exec -it nfl /bin/bash
cd /kaggle/solution
```
Extract frames from video (~12 minutes)
```bash
python3 extract_video_frames.py
```
Split data into training and validation (~3 seconds)
```bash
python3 split_train_val.py
```
Prepare data for detector training in COCO format (~11 minutes)
```bash
python3 convert_helmets_to_coco.py
```
Train the helmet detector (~5 hours)
```bash
python3 train_helmet_net.py --config-file configs/helmet_detector/faster_rcnn_R_50_FPN_1x_syncbn.yaml --num-gpus 4 OUTPUT_DIR /kaggle/artifacts/helmet_detector
```
Predict helmet boxes on frames to generate ROIs. They are needed to train impact classifiers (~1 hour)
```bash
python3 predict_boxes_on_frames.py --cfg configs/helmet_detector/faster_rcnn_R_50_FPN_1x_syncbn.yaml --weights /kaggle/artifacts/helmet_detector/model_final.pth --threshold 0.5
```
Train an impact classifier using an action detection approach (~28 hours)
```bash
python3 run_impact_net.py --cfg configs/impact_detector/SLOWFASTDECONV_20x1_R50_SHORT_V6.yaml OUTPUT_DIR /kaggle/artifacts/impact_detector
```
Train an impact classifier using an action classification approach (~48 hours)
```bash
python3 run_impact_net.py --cfg configs/impact_classifier/SLOWFAST_20x1_R50_CROP_V2.yaml OUTPUT_DIR /kaggle/artifacts/impact_classifier
```
Copy the final weights to the models folder. The weights from the last epoch are copied for the detector. For impact classifiers, the weights with the best validation score are copied (~5 seconds)
```bash
python3 copy_final_weights.py
```

## How to generate submission
### Validation
Copy validation videos to a separate folder (~5 seconds)
```bash
mkdir /kaggle/artifacts/val && for video in `cat /kaggle/artifacts/val_videos_split.txt`; do cp /kaggle/input/nfl-impact-detection/train/${video} /kaggle/artifacts/val; done
```
Run the following command to get predictions in csv format (~3 hours)
```bash
python3 predict_impacts.py --videos-root /kaggle/artifacts/val --output-csv-path /kaggle/artifacts/val_submission.csv
```
Get the result on the validation set (~5 seconds)
```bash
python3 evaluate.py --csv-path /kaggle/artifacts/val_submission.csv
```
### Test
To get predictions on the test set, run the following command (~40 minutes)
```bash
python3 predict_impacts.py --videos-root /kaggle/input/nfl-impact-detection/test --output-csv-path /kaggle/output/submission.csv
```

## `configs/settings.yaml`
This file specifies the path to the train, test, model, and output directories.

##  Serialized copy of the trained model
You can find the weights that I used to generate the final submission in the `models` folder
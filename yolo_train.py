import os
import json
import yaml
import wandb
from src.yolo_preprocessing import process_dataset_to_yolo, train_val_split

with open("./../api-keys.json") as s:
    secrets = json.load(s)

os.environ['WANDB_API_KEY'] = secrets['WANDB_API_KEY']

wandb.login(key=secrets['WANDB_API_KEY'])

from ultralytics import YOLO, settings
settings.update({"wandb": True})


source_img_dir = r"/kaggle/input/tree-canpy-detection/train"
source_ann_file = r"/kaggle/input/tree-canpy-detection/train_annotations_updated.json"
dataset_root = r"/kaggle/working/yolo_data_v1"

process_dataset_to_yolo(
    img_dir=source_img_dir,
    ann_file=source_ann_file,
    output_dir=dataset_root,
    out_file="all_train.txt",
    over_sample=True,
    copy=True
)

all_train_file = os.path.join(dataset_root, "all_train.txt")
output_train_file = os.path.join(dataset_root, "train_final.txt")
output_val_file = os.path.join(dataset_root, "val_final.txt")

# Perform train-validation split and save to separate files
final_train, final_val = train_val_split(all_train_file, val_size=0.2, output_train_file=output_train_file, output_val_file=output_val_file)

yolo_yaml = {'path': dataset_root,
             'train': output_train_file,
             'val': output_val_file,
             'names': {0: 'individual_tree', 1: 'group_of_trees'}}

with open(os.path.join(dataset_root, "dataset.yaml"), "w") as f:
    yaml.safe_dump(yolo_yaml, f)

print("YOLO dataset preparation complete.")

model = YOLO("yolov11s-seg.pt")

results = model.train(
    data=os.path.join(dataset_root, "dataset.yaml"),
    project="yolo_tree_canopy_detection",
    name="yolov11s-seg_experiment_1",

    device=0,
    workers=2,
    batch=16,

    epochs=100,
    patience=10,
    save=True,
    save_period=5,

    imgsz=640,

    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.1,
    cos_lr=True,
    warmup_epochs=5,

    overlap_mask=True,
    mask_ratio=1,

    hsv_h=0.015,
    hsv_s=0.6,
    hsv_v=0.4,

    flipud=0.5,
    fliplr=0.5,
    degrees=90.0,
    translate=0.15,
    scale=0.15,

    mosaic=0.7,
    close_mosaic=10,
    mixup=0.15,
    copy_paste=0.3
)

print("Training complete.")
print("Results:", results)

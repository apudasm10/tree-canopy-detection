import os
import json
import yaml
import wandb
import albumentations as A
from src.yolo_preprocessing import process_dataset_to_yolo, train_val_split

with open("./../api-keys.json") as s:
    secrets = json.load(s)

os.environ['WANDB_API_KEY'] = secrets['WANDB_API_KEY']

wandb.login(key=secrets['WANDB_API_KEY'])

from ultralytics import YOLO, settings
settings.update({"wandb": True})


source_img_dir = r"/kaggle/input/tree-canopy-detection/train"
source_ann_file = r"/kaggle/input/tree-canpy-detection/train_annotations_updated.json"
dataset_root = r"/kaggle/working/yolo_data_v1"
gsd_weight = {"10": 1, "20": 1, "40": 2, "60": 3, "80": 3}

process_dataset_to_yolo(
    img_dir=source_img_dir,
    ann_file=source_ann_file,
    output_dir=dataset_root,
    gsd_weight=gsd_weight,
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

model_name = "yolo11s-seg.pt"
model = YOLO(model_name)

aerial_augments = [
    A.RandomRotate90(p=0.5),
    A.RandomShadow(
        shadow_roi=[0, 0, 1, 1],
        num_shadows_limit=[1, 3],
        shadow_dimension=5,
        shadow_intensity_range=[0.2, 0.8],
        p=0.3
    ),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=0.5),
        A.GaussNoise(
            std_range=[0.05, 0.1],
            mean_range=[0, 0],
            per_channel=True,
            noise_scale_factor=1,
            p=0.5)
    ], p=0.2)
]

results = model.train(
    data=os.path.join(dataset_root, "dataset.yaml"),
    project="yolo_tree_canopy_detection",
    name=model_name.replace(".pt", "_experiment")+"_v1",

    device=0,
    workers=2,
    batch=16,

    epochs=100,
    patience=12,
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

    augmentations=aerial_augments,

    hsv_h=0.015,
    hsv_s=0.6,
    hsv_v=0.4,

    flipud=0.5,
    fliplr=0.5,
    degrees=45.0,
    translate=0.15,
    scale=0.15,

    mosaic=0.7,
    close_mosaic=10,
    mixup=0.15,
    copy_paste=0.3
)

print("Training complete.")
print("Results:", results)

import gc
import os
import json
import torch
import yaml
import wandb
from src.yolo_preprocessing import process_dataset_to_yolo_v2, train_val_split
import albumentations as A

gc.collect()
torch.cuda.empty_cache()

with open("./../api-keys.json") as s:
    secrets = json.load(s)

os.environ['WANDB_API_KEY'] = secrets['WANDB_API_KEY']

wandb.login(key=secrets['WANDB_API_KEY'])

from ultralytics import YOLO, settings
settings.update({"wandb": True})

random_state = 50
source_img_dir = os.path.join(os.getenv("HPCVAULT"), "TCD", "data", "train")
source_ann_file = os.path.join("data", "train_annotations_updated.json")
dataset_root = "yolo_data_v2"
out_file = "all_train.txt"
gsd_weight = {"10": 1, "20": 1, "40": 2, "60": 2, "80": 2}

process_dataset_to_yolo_v2(
    img_dir=source_img_dir,
    ann_file=source_ann_file,
    output_dir=dataset_root,
    gsd_weight=gsd_weight,
    out_file=out_file,
    over_sample=True,
    copy=True
)

all_train_file = os.path.join(dataset_root, "all_train.txt")
output_train_file = os.path.join(dataset_root, "train_final.txt")
output_val_file = os.path.join(dataset_root, "val_final.txt")

# Perform train-validation split and save to separate files
final_train, final_val = train_val_split(all_train_file, val_size=0.2, random_state=random_state, output_train_file=output_train_file, output_val_file=output_val_file)

yolo_yaml = {'path': dataset_root,
             'train': os.path.basename(output_train_file),
             'val': os.path.basename(output_val_file),
             'names': {0: 'individual_tree', 1: 'group_of_trees'}}

with open(os.path.join(dataset_root, "dataset.yaml"), "w") as f:
    yaml.safe_dump(yolo_yaml, f)

print("YOLO dataset preparation complete.")

aerial_augments = [
    A.RandomRotate90(p=0.3),
    A.Blur(blur_limit=(3, 7), p=0.07),
    A.MedianBlur(blur_limit=(3, 7), p=0.07),
    # A.ToGray(p=0.01),
    A.CLAHE(
        clip_limit=(1.0, 3.0), 
        tile_grid_size=(8, 8), 
        p=0.05
    )
]

# model_name = "yolov8m-p2.yaml"
model_name = "yolo11l.pt"
model = YOLO(model_name)
imgsz = 1184

results = model.train(
    data=os.path.join(dataset_root, "dataset.yaml"),
    project="HPC_TCD_GSD_Normalized",
    name=model_name.replace(".pt", f"_sz-{imgsz}_").replace(".yaml", f"_sz-{imgsz}_") + str(random_state),

    device=0,
    workers=16,
    batch=8,

    epochs=200,
    patience=40,
    save=True,
    save_period=0,

    imgsz=imgsz,

    optimizer="SGD",
    lr0=0.005,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=5,
    nbs=16,
    erasing=0.0,

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.2,

    flipud=0.5,
    fliplr=0.5,
    degrees=90.0,
    scale=0.15,
    translate=0.05,

    augmentations=aerial_augments,

    mosaic=0.1,
    close_mosaic=15,

    max_det=2200,

    box=10.0,
    # cls=0.8,
    # dfl=2.5,
)

gc.collect()
torch.cuda.empty_cache()

print("Training complete.")
print("Results:", results)

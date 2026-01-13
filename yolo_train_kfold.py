import os
import json
import random
import yaml
import wandb
import albumentations as A
import pandas as pd
from src.yolo_preprocessing import process_dataset_to_yolo
from sklearn.model_selection import StratifiedKFold

with open("api-keys.json") as s:
    secrets = json.load(s)

os.environ['WANDB_API_KEY'] = secrets['WANDB_API_KEY']

wandb.login(key=secrets['WANDB_API_KEY'])

from ultralytics import YOLO, settings
settings.update({"wandb": True})


source_img_dir = os.path.join("tree-canopy-detection", "train")
source_ann_file = os.path.join("tree-canopy-detection", "train_annotations_updated.json")
dataset_root = "yolo_data_v1"
out_file = "all_train.txt"
gsd_weight = {"10": 1, "20": 1, "40": 2, "60": 3, "80": 3}

process_dataset_to_yolo(
    img_dir=source_img_dir,
    ann_file=source_ann_file,
    output_dir=dataset_root,
    gsd_weight=gsd_weight,
    out_file=out_file,
    over_sample=False,
    copy=True
)

labels = os.listdir(os.path.join(dataset_root, "labels", "train"))
gsds = [label[:2] for label in labels]

df = pd.DataFrame({"path": labels, "gsd": gsds})

random_seed = 42
random.seed(random_seed)
ksplit = 5

kf = StratifiedKFold(n_splits=ksplit, shuffle=True, random_state=random_seed)
kfolds = list(kf.split(df, y=df['gsd']))
print(f"Prepared {ksplit}-Fold cross-validation splits.")
print(kfolds)

results_summary = []
yamls = []

print(f"\nSTARTING {ksplit}-FOLD CROSS VALIDATION")

for fold_idx, (train_idx, val_idx) in enumerate(kfolds):
    print(f"\n⚡ FOLD {fold_idx + 1}/{ksplit}")
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    train_path = os.path.join(dataset_root, "images", "train")
    final_train_paths = []
    for _, row in train_df.iterrows():
        w = gsd_weight.get(row['gsd'], 1)
        row_path = row['path'].replace(".txt", ".tif")
        full_img_path = os.path.abspath(os.path.join(train_path, row_path))
        final_train_paths.extend([full_img_path] * w)
        
    final_val_paths = val_df['path'].tolist()
    final_val_paths = [p.replace(".txt", ".tif") for p in final_val_paths]
    final_val_paths = [os.path.abspath(os.path.join(train_path, p)) for p in final_val_paths]
    
    # 3. Write .txt files
    t_txt = os.path.join(dataset_root, f"train_fold_{fold_idx}.txt")
    v_txt = os.path.join(dataset_root, f"val_fold_{fold_idx}.txt")
    
    with open(t_txt, 'w') as f: f.write('\n'.join(final_train_paths))
    with open(v_txt, 'w') as f: f.write('\n'.join(final_val_paths))
    
    # 4. Create YAML
    yaml_path = os.path.join(dataset_root, f"data_fold_{fold_idx}.yaml")
    yaml_data = {
        'path': os.path.abspath(dataset_root),
        'train': os.path.basename(t_txt),
        'val': os.path.basename(v_txt),
        'names': {0: 'individual_tree', 1: 'group_of_trees'}
    }
    with open(yaml_path, 'w') as f: yaml.dump(yaml_data, f)
    yamls.append(yaml_path)

print("YOLO dataset preparation complete.")

model_name = "yolo11l-seg.pt"

aerial_augments = [
    A.RandomRotate90(p=0.4),
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.25)
]

for fold_idx, yaml_path in enumerate(yamls):
    print(f"\nTraining fold {fold_idx + 1}/{ksplit} with data config: {yaml_path}")
    model = YOLO(model_name)
    project="TCD_YOLO_KFOLD"
    run_name = f"yolo11l-seg-fold{fold_idx+1}-v1"

    results = model.train(
        data=yaml_path,
        project=project,
        name=run_name,

        device=0,
        workers=4,
        batch=4,

        epochs=100,
        patience=20,
        save=True,
        save_period=0,

        imgsz=896,
        cache=True,

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
        scale=0.15,

        mosaic=0.5,
        close_mosaic=15,
        copy_paste=0.15
    )

print("Training complete.")
print("Results:", results)

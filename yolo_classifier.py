import os
import json
import wandb
from src.yolo_preprocessing import prepare_classification_data

with open("api-keys.json") as s:
    secrets = json.load(s)

os.environ['WANDB_API_KEY'] = secrets['WANDB_API_KEY']

wandb.login(key=secrets['WANDB_API_KEY'])

from ultralytics import YOLO, settings
settings.update({"wandb": True})

source_dir = r"tree-canopy-detection\train"
dest_dir = r"classification_data_v1"
json_path = r"tree-canopy-detection\train_annotations_updated.json"
weights = {
    "agriculture_plantation":3.00,
    "urban_area":2.00,
    "rural_area":1.00,
    "industrial_area":2.00,
    "open_field":1.00
}

prepare_classification_data(source_dir, dest_dir, json_path, weights, val_size=0.2, stratify=True)

model_name = "yolo11m-cls.pt"
model = YOLO(model_name)

results = model.train(
    data=dest_dir, 
    
    project="yolo_scene_classification",
    name=model_name.replace(".pt", "_experiment")+"_v4",
    
    device="cpu",
    workers=4,
    
    epochs=50,
    patience=10,
    batch=16,
    imgsz=384,
    
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.1,
    cos_lr=True,
    warmup_epochs=3,
    
    hsv_h=0.015, 
    hsv_s=0.5, 
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.5,
    degrees=45.0, 
    translate=0.1,
    scale=0.2,
)

print("Classification Training Complete.")
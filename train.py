from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
# from src.metrics import *
from src.training_utils import evaluate
import torch
from torch.optim import SGD
from tqdm import tqdm
import time
import json
import os
from torch.amp import GradScaler
from torch.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import timedelta
from src.model_utils import *
from src.dataset import TCDDataset, collate_fn
from sklearn.model_selection import train_test_split
import wandb

torch.cuda.empty_cache()

start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

with open("./../api-keys.json") as s:
    secrets = json.load(s)

os.environ['WANDB_API_KEY'] = secrets['WANDB_API_KEY']

ANNOTATIONS_FILE = "data/train_annotations_updated.json"
img_dir = os.path.join(os.getenv("HPCVAULT"), "TCD/data/train")

train_augments = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=40, val_shift_limit=20, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.ToFloat(max_value=255.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.2))

aug_config = A.to_dict(train_augments)

val_augments = A.Compose([
    A.ToFloat(max_value=255.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

train_ds_aug = TCDDataset(
    img_dir=img_dir,
    ann_file=ANNOTATIONS_FILE,
    augments=train_augments
)
train_ds_eval = TCDDataset(
    img_dir=img_dir,
    ann_file=ANNOTATIONS_FILE,
    augments=val_augments
)
val_ds_eval = TCDDataset(
    img_dir=img_dir,
    ann_file=ANNOTATIONS_FILE,
    augments=val_augments
)

train_idx, val_idx = train_test_split(range(150), train_size=0.8, shuffle=True, random_state=42)

train_dataset = Subset(train_ds_aug,  train_idx)
train_dataset_eval = Subset(train_ds_eval, train_idx[:30])
val_dataset = Subset(val_ds_eval,   val_idx)

print(f"Train: {len(train_dataset)} | Train-eval: {len(train_dataset_eval)} | Val: {len(val_dataset)}")

BATCH_SIZE = 2
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4)
train_loader_eval = DataLoader(train_dataset_eval, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

num_classes = 3
last_level = "LastLevelMaxPool"
model_name = "convnext_large.dinov3_lvd1689m"
backbone = CustomBackbone(model_name, False)

fpn = CustomFPN(backbone.in_channels_list, 256, backbone.feature_module_names, last_level=last_level)

pipe = CustomModelFPN(backbone, fpn)

if last_level == "LastLevelMaxPool":
    featmap_names_box=["p2", "p3", "p4", "p5", "pool"]
    featmap_names_mask=["p2", "p3", "p4", "p5"]
    sizes=((8, 16,), (32, 48), (64, 96), (128, 256), (384, 512))
    aspect_ratios=((0.75, 1.0, 1.33),) * 5
elif last_level == "LastLevelP6P7":
    featmap_names_box=["p2", "p3", "p4", "p5", "p6", "p7"]
    featmap_names_mask=["p2", "p3", "p4", "p5"]
    sizes=((8, 16,), (32, 48), (64, 96), (128, 192), (256, 384), (512, 1024))
    aspect_ratios=((0.75, 1.0, 1.33),) * 6
else:
    print("Define a valid last_level: [LastLevelMaxPool, LastLevelP6P7]")

anchor_generator = AnchorGenerator(
    sizes=sizes,
    aspect_ratios=aspect_ratios
)

box_roi_pool = MultiScaleRoIAlign(
        featmap_names=featmap_names_box,
        output_size=7,
        sampling_ratio=2,
    )

mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=featmap_names_mask,
        output_size=28,
        sampling_ratio=2,
    )

MAX_DETS = 1050
model = MaskRCNN(
    pipe,
    num_classes=3,
    box_roi_pool=box_roi_pool,
    mask_roi_pool=mask_roi_pool,
    rpn_anchor_generator=anchor_generator,
    box_detections_per_img=MAX_DETS,
    min_size=1024,
    max_size=1024,
    rpn_pre_nms_top_n_train=5000,
    rpn_pre_nms_top_n_test=3000,
    rpn_post_nms_top_n_train=3000,
    rpn_post_nms_top_n_test=2000,
)

print("Model created.")

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

EXP_NAME = "Mask_RCNN_Run_1"
save_dir = os.path.join(os.getenv("HPCVAULT"), f"TCD/{EXP_NAME}")

os.makedirs(save_dir, exist_ok=True)

model.to(device)

num_epochs = 50
accumulation_steps = 4
print("Running with accumulation_steps =", accumulation_steps)

params_backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
params_others = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

optimizer = torch.optim.AdamW([
    {'params': params_backbone, 'lr': 5e-6, 'weight_decay': 0.05}, 
    {'params': params_others, 'lr': 2e-4, 'weight_decay': 0.05}  
])

total_steps = num_epochs * (len(train_loader) // accumulation_steps)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, 
    start_factor=0.001, 
    total_iters=5
)

main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-5, eta_min=1e-7)

lr_sched = torch.optim.lr_scheduler.SequentialLR(
    optimizer, 
    schedulers=[warmup_scheduler, main_scheduler], 
    milestones=[5]
)
scaler = GradScaler('cuda') if device.type == 'cuda' else None
clip = 5

wandb.init(
    project=f"tcd-experiments-1",
    name=f"{model_name}-{EXP_NAME}_with_new_augs",
    config={
        "batch_size": BATCH_SIZE,
        "architecture": model_name,
        "epochs": num_epochs,
        "max_dets": MAX_DETS,
        "Exp": EXP_NAME,
        "last_level": last_level,
        "augmentations": aug_config,
        "accumulation_steps": accumulation_steps
    }
)

for epoch in range(num_epochs):
    start_in = time.time()
    model.train()
    running_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
    for i, (imgs, targets) in enumerate(progress_bar):
        imgs = [img.to(device, non_blocking=True) for img in imgs]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        with autocast('cuda'):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            
            if clip:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad(set_to_none=True)

        current_loss = loss.item() * accumulation_steps 
        running_loss += current_loss
        n_batches += 1
        progress_bar.set_postfix(loss=f"{current_loss:.3f}")

    lr_sched.step()
    avg_loss = running_loss / len(train_loader)
    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")
    log_data = {
        'epoch': epoch,
        'train_loss': avg_loss
    }
    print("\nSummarizing training metrics...")
    train_summary = evaluate(model, train_loader_eval, device)
    for k, v in train_summary.items():
        val = v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist()
        log_data[f"train_{k}"] = val

    print("\nSummarizing validation metrics...")
    val_summary = evaluate(model, val_loader, device)
    for k, v in val_summary.items():
        val = v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v.tolist()
        log_data[f"val_{k}"] = val

    wandb.log(log_data)

    model_name = f"maskrcnn_epoch_{epoch+1}.pth"
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    tqdm.write(f"Saved model checkpoint to: {save_path}")

    torch.cuda.empty_cache()
    end_in = time.time()
    diff_in = end_in - start_in
    formatted = str(timedelta(seconds=int(diff_in)))

    print("Time for the epoch:", formatted)


end = time.time()

diff = end - start
formatted = str(timedelta(seconds=int(diff)))

print("Time:", formatted)
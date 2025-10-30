from torch.utils.data import DataLoader, random_split
from torchvision.models.detection.anchor_utils import AnchorGenerator
from src.dataset import *
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from src.metrics import *
import torch
from torch.optim import SGD
from tqdm import tqdm
import time
from datetime import timedelta

torch.cuda.empty_cache()

start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

ALL_ANNOTATIONS_FILE = "data/coco_annotations.json"

# 1) Build a deterministic split
full_ds_for_split = CocoMaskDataset(
    img_dir="/home/vault/iwso/iwso195h/TCD/data/train",
    ann_file=ALL_ANNOTATIONS_FILE,
    resize=(1024, 1024),
    augment=False,          # augment flag irrelevant here; we only take length
)
N = len(full_ds_for_split)
train_size = int(0.8 * N)
val_size   = N - train_size

g = torch.Generator().manual_seed(42)       # reproducible split
perm = torch.randperm(N, generator=g)
train_idx = perm[:train_size].tolist()
val_idx   = perm[train_size:].tolist()

# 2) Create THREE datasets with different transforms
train_ds_aug = CocoMaskDataset(
    img_dir="/home/vault/iwso/iwso195h/TCD/data/train",
    ann_file=ALL_ANNOTATIONS_FILE,
    resize=(1024, 1024),
    augment=True           # training uses augmentation
)
train_ds_eval = CocoMaskDataset(
    img_dir="/home/vault/iwso/iwso195h/TCD/data/train",
    ann_file=ALL_ANNOTATIONS_FILE,
    resize=(1024, 1024),
    augment=False          # training-set evaluation: NO augmentation
)
val_ds_eval = CocoMaskDataset(
    img_dir="/home/vault/iwso/iwso195h/TCD/data/train",
    ann_file=ALL_ANNOTATIONS_FILE,
    resize=(1024, 1024),
    augment=False          # validation/test: NO augmentation
)

# 3) Wrap each with the same indices
train_dataset       = Subset(train_ds_aug,  train_idx)
train_dataset_eval  = Subset(train_ds_eval, train_idx)
val_dataset         = Subset(val_ds_eval,   val_idx)

# 4) Loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
train_loader_eval = DataLoader(train_dataset_eval, batch_size=1, shuffle=False, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

train_metrics = CocoMetrics(annotation_file=ALL_ANNOTATIONS_FILE)
val_metrics = CocoMetrics(annotation_file=ALL_ANNOTATIONS_FILE)

num_classes = 3

anchor_generator = AnchorGenerator(
    sizes=((16,), (32,), (64,), (128,), (256,),),  # tiny → group
    aspect_ratios=((0.5, 0.75, 1.0, 1.33, 2.0),) * 5
)

model = maskrcnn_resnet50_fpn(weights="DEFAULT")

model.rpn.anchor_generator = anchor_generator
num_anchors = anchor_generator.num_anchors_per_location()[0]
model.rpn.head = RPNHead(model.rpn.head.conv[0][0].in_channels, num_anchors)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
model.rpn.pre_nms_top_n_train  = 3000
model.rpn.post_nms_top_n_train = 1500
model.rpn.pre_nms_top_n_test   = 2000
model.rpn.post_nms_top_n_test  = 1024

save_dir = "/home/vault/iwso/iwso195h/TCD/Run 10"
print("Anchors added, epoch 100")

print("Running with accumulation_steps = 4")

os.makedirs(save_dir, exist_ok=True)

model.to(device)

optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[33, 66, 90], gamma=0.2)
accumulation_steps = 4

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_metrics.reset()
    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
    for i, (imgs, targets) in enumerate(progress_bar):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward + loss
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        # backward
        # optimizer.zero_grad()
        losses.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # optimizer.step()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()  # Update parameters
            optimizer.zero_grad()

        epoch_loss += losses.item()
        progress_bar.set_postfix({"batch_loss": f"{losses.item():.4f}"})

    lr_sched.step()
    avg_loss = epoch_loss / len(train_loader)
    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

    print("\nSummarizing training metrics...")
    train_summary = evaluate(model, train_loader_eval, train_metrics, device)
    if train_summary:
        print(f"--- Epoch {epoch + 1} Train Metrics ---")
        print(f"mAP (IoU=0.50:0.95): {train_summary['mAP (IoU=0.50:0.95)']:.4f}")
        print(f"mAP (IoU=0.50): {train_summary['mAP (IoU=0.50)']:.4f}")
        print("---------------------------------")
    else:
        print("Could not compute train metrics.")

    print("\nSummarizing validation metrics...")
    val_summary = evaluate(model, val_loader, val_metrics, device)
    
    if val_summary:
        print(f"--- Epoch {epoch + 1} Validation Metrics ---")
        # Print only the 4 most important metrics
        print(f"mAP (IoU=0.50:0.95): {val_summary['mAP (IoU=0.50:0.95)']:.4f}")
        print(f"mAP (IoU=0.50): {val_summary['mAP (IoU=0.50)']:.4f}")
        print(f"mAP (IoU=0.75): {val_summary['mAP (IoU=0.75)']:.4f}")
        print(f"mAP (Small): {val_summary['mAP (Small)']:.4f}")
        print("---------------------------------")
    else:
        print("Could not compute metrics. (This is expected if dummy file was used)")



    model_name = f"maskrcnn_epoch_{epoch+1}.pth"
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    tqdm.write(f"Saved model checkpoint to: {save_path}")

    torch.cuda.empty_cache()


end = time.time()

diff = end - start
formatted = str(timedelta(seconds=int(diff)))

print("Time:", formatted)
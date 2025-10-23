from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.dataset import *
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
from torch.optim import SGD
from tqdm import tqdm

torch.cuda.empty_cache()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

dataset = CocoMaskDataset(
    img_dir="/home/vault/iwso/iwso195h/TCD/data/train",
    ann_file="data/coco_annotations.json",
    resize=(1024, 1024),
    augment=True
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

num_classes = 3

model = maskrcnn_resnet50_fpn(weights="DEFAULT")

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
model.rpn.post_nms_top_n_train = 1024
model.rpn.post_nms_top_n_test  = 512
model.rpn.pre_nms_top_n_train  = 1024
model.rpn.pre_nms_top_n_test   = 512

save_dir = "/home/vault/iwso/iwso195h/TCD/Run 1"

os.makedirs(save_dir, exist_ok=True)

model.to(device)

optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
    for imgs, targets in progress_bar:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward + loss
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        # backward
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_loss += losses.item()
        progress_bar.set_postfix({"batch_loss": f"{losses.item():.4f}"})

    avg_loss = epoch_loss / len(train_loader)
    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

    model_name = f"maskrcnn_epoch_{epoch+1}.pth"
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    tqdm.write(f"Saved model checkpoint to: {save_path}")

    torch.cuda.empty_cache()

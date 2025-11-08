import json
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchvision import models
import os
from PIL import Image
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2

file_dir = os.path.join(os.getenv("HPCVAULT"), "TCD/data/train_annotations.json")
with open(file_dir, "r") as f:
    data = json.load(f)

records = data["images"]

image_names = []
image_classes = []

for img_id, rec in enumerate(records, start=1):
    image_names.append(rec["file_name"])
    image_classes.append(rec["scene_type"])
    
df = pd.DataFrame({"image_name": image_names, "image_class": image_classes})
print(df.head())

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True, stratify=df["image_class"])

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

print("Train:", len(train_df))
print("Test:", len(test_df))

CLASSES = ['agriculture_plantation','rural_area','urban_area','open_field','industrial_area']
class_to_idx = {c:i for i,c in enumerate(CLASSES)}

IMG_DIR = "/home/vault/iwso/iwso195h/TCD/data/train/"
device = "cuda" if torch.cuda.is_available() else "cpu"

class AerialDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])
        img = np.array(Image.open(img_path).convert("RGB"))
        label_idx = class_to_idx[row["image_class"]]

        # one-hot encode
        # label = torch.eye(len(CLASSES))[label_idx]

        img = self.transform(image=img)["image"]
        return img, label_idx
    
train_ds = AerialDataset(train_df, IMG_DIR, train_transform)
test_ds  = AerialDataset(test_df,  IMG_DIR, test_transform)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False)

print(train_ds[74][1])
print(test_ds[3][1])

total_epochs = 30

model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(device)

criterion = criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

lr_sched = OneCycleLR(optimizer, 
                    max_lr=1e-3,
                    epochs=total_epochs, # Total epochs
                    steps_per_epoch=len(train_loader)) # Batches per epoch

def train_one_epoch():
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    loop = tqdm(train_loader, leave=False)
    for imgs, labels in loop:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_sched.step()

        loss_sum += loss.item() * imgs.size(0)
        preds = outputs.softmax(1).argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        loop.set_description(f"Train loss: {loss.item():.4f}")

    return loss_sum / total, correct / total


@torch.no_grad()
def eval_once():
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    loop = tqdm(test_loader, leave=False)
    for imgs, labels in loop:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss_sum += loss.item() * imgs.size(0)
        preds = outputs.softmax(1).argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        loop.set_description(f"Test loss: {loss.item():.4f}")

    return loss_sum / total, correct / total

save_dir = "/home/vault/iwso/iwso195h/TCD/cls_model_final2"
os.makedirs(save_dir)
# ===== Train =====
for epoch in range(total_epochs):
    tr_loss, tr_acc = train_one_epoch()
    te_loss, te_acc = eval_once()

    # --- Optional: Add LR to see the scheduler work ---
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:02d} | Train {tr_loss:.4f}/{tr_acc:.3f} | Test {te_loss:.4f}/{te_acc:.3f} | LR: {current_lr:.6f}")
    # --- End of change ---

    # print(f"Epoch {epoch+1:02d} | Train {tr_loss:.4f}/{tr_acc:.3f} | Test {te_loss:.4f}/{te_acc:.3f}")
    model_path = os.path.join(save_dir, f"resnet50_aerial_{epoch}.pth")

    torch.save(model.state_dict(), model_path)
    print("Model saved at:", model_path)

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools import mask as coco_mask
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CocoMaskDataset(Dataset):
    def __init__(self, img_dir, ann_file, resize=(224, 224), augment=False):
        self.img_dir = img_dir
        self.resize = resize
        self.augment = augment

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        # group annotations by image id
        self.img2anns = {}
        for ann in self.annotations:
            self.img2anns.setdefault(ann["image_id"], []).append(ann)

        # define Albumentations transforms
        self.transform = self._build_transform()

    def _build_transform(self):
        H, W = self.resize
        if self.augment:
            return A.Compose([
                A.Resize(H, W),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=(0,0,0), std=(1,1,1)),  # ✅ convert to float
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            return A.Compose([
                A.Resize(H, W),
                A.Normalize(mean=(0,0,0), std=(1,1,1)),  # ✅ same here
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = img.shape[:2]

        anns = self.img2anns.get(img_info["id"], [])

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

            seg = ann.get("segmentation", [])
            if seg:
                if isinstance(seg[0], (int, float)):
                    seg = [seg]
                rles = coco_mask.frPyObjects(seg, orig_h, orig_w)
                mask = coco_mask.decode(rles)
                mask = mask.max(axis=2) if mask.ndim == 3 else mask
                masks.append(mask.astype(np.uint8))
            else:
                masks.append(np.zeros((orig_h, orig_w), dtype=np.uint8))

        if len(masks) > 0:
            masks = np.stack(masks)
        else:
            masks = np.zeros((0, orig_h, orig_w), dtype=np.uint8)

        # Apply Albumentations transform (auto-resizes all)
        transformed = self.transform(image=img, masks=list(masks), bboxes=boxes, labels=labels)
        img = transformed["image"]
        masks = torch.stack([m for m in transformed["masks"]]) if len(transformed["masks"]) > 0 else torch.zeros((0, *self.resize), dtype=torch.uint8)
        boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.as_tensor(transformed["labels"], dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([img_info["id"]])

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))

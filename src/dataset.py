import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2


class TCDDataset(Dataset):
    def __init__(self, img_dir, ann_file, augments):
        self.img_dir = img_dir
        self.augments = augments
        self.label2int = {"individual_tree": 1, "group_of_trees": 2}

        with open(ann_file, "r") as f:
            self.all_file = json.load(f)["images"]

    def __len__(self):
        return len(self.all_file)
    
    def __getitem__(self, idx):
        sample = self.all_file[idx]
        img_path = os.path.join(self.img_dir, sample.get("file_name"))
        img = np.array(Image.open(img_path).convert("RGB"))
        annotations = sample.get("annotations")
        width = sample.get("width")
        height = sample.get("height")

        boxes, labels, masks = [], [], []
        for obj in annotations:
            x_coords = obj.get("segmentation")[0::2]
            y_coords = obj.get("segmentation")[1::2]
            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.label2int.get(obj["class"]))
                mask = get_mask(obj.get("segmentation"), height, width)
                masks.append(mask.astype(np.uint8))
            else:
                print("Invalid bbox", [x1, y1, x2, y2])

        if self.augments:
            augmented = self.augments(image=img, masks=masks, bboxes=boxes, class_labels=labels)
            img = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["class_labels"]
            masks = augmented["masks"]

        areas = [np.count_nonzero(mask_i) for mask_i in masks]
        areas = torch.as_tensor(areas, dtype=torch.float32)
        keep = areas > 0

        image_id = torch.tensor([idx], dtype=torch.int64)
        iscrowd = torch.zeros(len(areas), dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes[keep],
            "labels": labels[keep],
            "masks": masks[keep],
            "image_id": image_id,
            "area": areas[keep],
            "iscrowd": iscrowd[keep],
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# def get_polygon_area(xs, ys):
#     """
#     returns polygon area via Shoelace formula
#     """
#     x = np.array(xs)
#     y = np.array(ys)

#     correction = x[-1] * y[0] - x[0] * y[-1]
#     main_area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    
#     return 0.5 * np.abs(main_area + correction)

def get_mask(segmentation, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(segmentation, dtype=np.int32).reshape(-1, 2)], 1)
    return mask
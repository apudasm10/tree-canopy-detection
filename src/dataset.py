import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
from torchvision.ops import masks_to_boxes
import albumentations as A
import pandas as pd


class TCDDataset(Dataset):
    def __init__(self, img_dir, ann_file, augments, target_gsd=20.0, crop_size=444):
        self.img_dir = img_dir
        self.augments = augments
        self.target_gsd = target_gsd
        self.crop_size = crop_size
        self.label2int = {"individual_tree": 1, "group_of_trees": 2}

        with open(ann_file, "r") as f:
            self.all_file = json.load(f)["images"]

    def __len__(self):
        return len(self.all_file)
    
    def __getitem__(self, idx):
        sample = self.all_file[idx]
        img_path = os.path.join(self.img_dir, sample.get("file_name"))
        img = np.array(Image.open(img_path).convert("RGB"))
        
        current_gsd = float(sample.get("cm_resolution"))
        scale = current_gsd / self.target_gsd
        h, w = img.shape[:2]
        
        src_size = int(self.crop_size / scale) 

        # 2. GENERATE SINGLE ID MASK (Memory Safe)
        # 0=Background, 1=Tree1, 2=Tree2...
        id_mask = np.zeros((h, w), dtype=np.int32)
        id_to_label = {}
        
        annotations = sample.get("annotations")
        for i, obj in enumerate(annotations, 1):
            poly = np.array(obj.get("segmentation"), dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(id_mask, [poly], i)
            id_to_label[i] = self.label2int.get(obj["class"], 1)

        # 3. DEFINE CROP TRANSFORM (Albumentations Function) 
        # Randomly pick coordinates
        pad_h, pad_w = max(0, src_size - h), max(0, src_size - w)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            id_mask = cv2.copyMakeBorder(id_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = img.shape[:2]

        x = np.random.randint(0, w - src_size + 1)
        y = np.random.randint(0, h - src_size + 1)

        # The Magic Function: Crops & Resizes Image + ID Mask together
        crop_aug = A.Compose([
            A.Crop(x, y, x_max=x+src_size, y_max=y+src_size),
            A.Resize(height=self.crop_size, width=self.crop_size, 
                     interpolation=cv2.INTER_CUBIC, 
                     mask_interpolation=cv2.INTER_NEAREST) # NEAREST is required for IDs
        ])
        
        augmented_crop = crop_aug(image=img, mask=id_mask)
        img = augmented_crop["image"]
        id_mask = augmented_crop["mask"]

        # 4. RECONSTRUCT BINARY MASKS (Now they are small -> Safe)
        masks, labels = [], []
        visible_ids = np.unique(id_mask)
        
        for oid in visible_ids:
            if oid == 0: continue
            masks.append((id_mask == oid).astype(np.uint8))
            labels.append(id_to_label[oid])

        if self.augments:
            if len(masks) == 0:
                masks_np = np.zeros((0, img.shape[0], img.shape[1]), dtype=np.uint8)
            else:
                masks_np = np.stack(masks, axis=0).astype(np.uint8)

            augmented = self.augments(
                image=img,
                masks=masks_np,
                class_labels=labels
            )

            img = augmented["image"]
            masks = augmented["masks"]   # now a NumPy array (N, H, W)


        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = [np.count_nonzero(mask_i) for mask_i in masks]
        areas = torch.as_tensor(areas, dtype=torch.float32)
        
        if len(masks) > 0:
            boxes = masks_to_boxes(masks)
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            keep = (areas > 0) & (widths >= 1) & (heights >= 1)
            target = {
                "boxes": boxes[keep],
                "labels": labels[keep],
                "masks": masks[keep],
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": areas[keep],
                "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)
            }
        else:
            target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, h, w), dtype=torch.uint8),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64)
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


class TCDClassification(Dataset):
    def __init__(self, img_dir, ann_file, augments):
        self.img_dir = img_dir
        self.augments = augments
        self.label2int = {
            "agriculture_plantation": 0,
            "urban_area": 1,
            "industrial_area": 2,
            "rural_area": 3,
            "open_field": 4
        }

        with open(ann_file, "r") as f:
            data = json.load(f)
    
        self.data = pd.DataFrame([
            {'filename': item['file_name'], 'label': item['scene_type']} 
            for item in data['images']])
        
        print("Class counts:", self.data['label'].value_counts().to_dict())
        
        self.targets = [self.label2int[x] for x in self.data['label']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, sample['filename'])
        img = Image.open(img_path).convert("RGB")
        label = self.label2int.get(sample['label'], 0)

        if self.augments:
            img = self.augments(img)

        return img, label
    
class CustomDataset(Dataset):
    def __init__(self, img_list, img_folder, json_file, processor, augments=None):
        self.img_list = img_list
        self.img_folder = img_folder
        self.processor = processor
        self.augments = augments

        with open(json_file, 'r') as f:
            all_data = json.load(f)["images"]

        self.ann_map = {}
        for item in all_data:
            self.ann_map[item["file_name"]] = {
                "annotations": item.get("annotations", []),
                "file_name": item["file_name"],
                "height": item["height"],
                "width": item["width"],
                "cm_resolution": item["cm_resolution"],
                "scene_type": item["scene_type"]
            }
            
        self.label2int = {"individual_tree": 0, "group_of_trees": 1}
        self.int2label = {idx: name for name, idx in self.label2int.items()}

        print(f"Found {len(self.img_list)} images.")
        print(f"Class Mapping: {self.label2int}")
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        file_name = self.img_list[idx]
        sample = self.ann_map.get(file_name)
        img_path = os.path.join(self.img_folder, sample.get("file_name"))
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]
        
        labels = []
        annotations = sample.get('annotations', [])
        id_mask = np.zeros((h, w), dtype=np.int32)
        id_to_label = {0: 255}  # Background mapped to ignore index
        
        for i, obj in enumerate(annotations, 1):
            poly = np.array(obj.get("segmentation"), dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(id_mask, [poly], i)
            id_to_label[i] = self.label2int.get(obj["class"], 1)
            labels.append(self.label2int.get(obj["class"], 0))

        if self.augments:
            augmented = self.augments(
                image=image,
                mask=id_mask,
                class_labels=labels
            )

            image = augmented["image"]
            id_mask = augmented["mask"]

            unique_ids = np.unique(id_mask)
            id_to_label = {k: v for k, v in id_to_label.items() if k in unique_ids}

        inputs = self.processor(
            images=image,
            segmentation_maps=id_mask,
            instance_id_to_semantic_id=id_to_label,
            return_tensors="pt"
        )
        
        labels = inputs["class_labels"][0] # Shape: (Num_Objects,)
        masks = inputs["mask_labels"][0]   # Shape: (Num_Objects, H, W)
        
        # 2. Find valid objects (Label is NOT 255)
        keep_indices = labels != 255
        
        # 3. Overwrite inputs with filtered data and remove batch dim
        inputs["class_labels"] = labels[keep_indices]
        inputs["mask_labels"] = masks[keep_indices]
        inputs["pixel_values"] = inputs["pixel_values"][0] 
        
        return inputs
    

def collate_fn_seg(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    pixel_mask = torch.stack([x["pixel_mask"] for x in batch]) if "pixel_mask" in batch[0] else None
    
    mask_labels = [x["mask_labels"] for x in batch]
    class_labels = [x["class_labels"] for x in batch]
    
    batch_out = {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }
    if pixel_mask is not None:
        batch_out["pixel_mask"] = pixel_mask
        
    return batch_out
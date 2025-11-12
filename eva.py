"""Infer and prepare output for submission"""
import os, json
import torch.nn as nn
import torch
from torchvision import models
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_DIR = "/home/vault/iwso/iwso195h/TCD/data/val"
IMG_DIR = os.path.join(os.getenv("HPCVAULT"), "TCD/data/val")
CHECKPOINT = os.path.join(os.getenv("HPCVAULT"), "TCD/Train_MASK_RCNN_Run_1.0/maskrcnn_epoch_90.pth")
CLS_CHECKPOINT = os.path.join(os.getenv("HPCVAULT"), "TCD/cls_model_final2/resnet50_aerial_25.pth")
OUT_JSON = "data/preds8.json"
CLASS_NAMES  = ["background", "individual_tree", "group_of_trees"]
CLASSES = ['agriculture_plantation','rural_area','urban_area','open_field','industrial_area']
SCORE_THR = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def mask_to_polygon(bin_mask, max_pts=128):
    m = bin_mask.astype(np.uint8)
    if m.max() == 0: return []
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return []
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True).reshape(-1, 2)
    if len(approx) > max_pts:
        idx = np.linspace(0, len(approx) - 1, max_pts).astype(int)
        approx = approx[idx]
    return approx.astype(float).round(1).flatten().tolist()

if __name__ == "__main__":
    aug = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()])
    
    cls_aug = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    num_classes = 3
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE), strict=True)
    print(f"Loaded weights from {CHECKPOINT}")

    cls_model = models.resnet152(weights=None)
    cls_model.fc = nn.Linear(cls_model.fc.in_features, len(CLASSES))
    cls_model.load_state_dict(torch.load(CLS_CHECKPOINT, map_location=DEVICE))
    print(f"Loaded weights from {CLS_CHECKPOINT}")

    model.eval()
    model.to(DEVICE)

    cls_model.eval()
    cls_model.to(DEVICE)

    images = os.listdir(IMG_DIR)
    print("Total Images:", len(images))
    if not images:
        raise SystemExit(f"No images under {IMG_DIR}")

    out = {"images": []}
    for im_no, p in enumerate(images):
        print("Evaluating Image No:", im_no)
        im = Image.open(p).convert("RGB")
        W, H = im.size
        im = np.array(im)
        x = aug(image=im)['image'].to(DEVICE)
        cls_x = cls_aug(image=im)['image'].unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            y = model([x])[0]
            pred_cls = cls_model(cls_x)
            
            probs = torch.softmax(pred_cls, dim=1)
            pred_idx = int(probs.argmax(dim=1).cpu().item())
            pred_class = CLASSES[pred_idx]
            # print(pred_class)

        boxes  = y.get("boxes", torch.empty(0,4)).detach().cpu().numpy()
        labels = y.get("labels", torch.empty(0,)).detach().cpu().numpy().astype(int)
        scores = y.get("scores", torch.empty(0,)).detach().cpu().numpy()
        masks  = y.get("masks", None)
        masks  = masks.detach().cpu().numpy()[:,0] if masks is not None else None  # [N,H,W]

        anns = []
        for i in range(len(boxes)):
            if scores[i] < SCORE_THR: 
                continue
            cls = labels[i]
            if cls < 0 or cls >= len(CLASS_NAMES):
                continue
            seg = []
            if masks is not None and masks.shape[-2:] == (H, W):
                seg = mask_to_polygon(masks[i] > 0.5)
            if not seg:
                x1, y1, x2, y2 = boxes[i].tolist()
                seg = [x1, y1, x2, y1, x2, y2, x1, y2]

            anns.append({
                "class": CLASS_NAMES[cls],
                "confidence_score": float(scores[i]),
                "segmentation": [float(v) for v in seg],
            })

        out["images"].append({
            "file_name": p,
            "width": W,
            "height": H,
            "cm_resolution": int(p.split("cm")[0]),
            "scene_type": pred_class,
            "annotations": anns
        })

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out['images'])} images -> {OUT_JSON}")

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
from src.model_utils import *
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# --- CONFIG ---
HPC_VAULT = os.getenv("HPCVAULT")
IMG_DIR = os.path.join(HPC_VAULT, "TCD/data/val")
CHECKPOINT = os.path.join(HPC_VAULT, "TCD/Train_MASK_RCNN_Run_1.5/maskrcnn_epoch_74.pth")
CLS_CHECKPOINT = os.path.join(HPC_VAULT, "TCD/cls_model_final2/resnet50_aerial_25.pth")
OUT_JSON = "data/preds_dino.json"
CLASS_NAMES  = ["background", "individual_tree", "group_of_trees"]
CLASSES = ['agriculture_plantation','rural_area','urban_area','open_field','industrial_area']
SCORE_THR = 0.05 # Keep low for Recall
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- FIXED FUNCTION: Returns Single Flat List [x,y,x,y...] ---
def mask_to_polygon(bin_mask, min_area=20.0):
    m = bin_mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts: 
        return []
        
    # 1. Select the largest contour to satisfy the "Single List" requirement
    cnt = max(cnts, key=cv2.contourArea)
    
    if cv2.contourArea(cnt) < min_area:
        return []

    # 2. Use high precision (0.002) to keep the tree shape detailed
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.005 * peri, True).reshape(-1, 2)
    
    # Need at least 3 points (6 coords) for a polygon
    if len(approx) < 3:
        return []
        
    # 3. Flatten to [x1, y1, x2, y2, ...]
    return approx.astype(float).round(1).flatten().tolist()

if __name__ == "__main__":
    aug = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    cls_aug = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    num_classes = 3
    model_name = "timm/convnext_large.dinov3_lvd1689m"
    backbone = DINOBackbone(model_name, True)
    fpn = DINOV3FPN(backbone.in_channels_list, 256, backbone.feature_module_names)
    pipe = CustomModelFPN(backbone, fpn)

    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 0.75, 1.0, 1.33, 2.0),) * 6
    )

    box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5", "p6", "p7"], 
            output_size=7,
            sampling_ratio=2,
        )

    mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5", "p6", "p7"], 
            output_size=14,
            sampling_ratio=2,
        )

    model = MaskRCNN(
        pipe,
        num_classes=3,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        rpn_anchor_generator=anchor_generator
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    print(f"Loading weights from {CHECKPOINT}")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE), strict=True)
    model.eval().to(DEVICE)

    cls_model = models.resnet152(weights=None)
    cls_model.fc = nn.Linear(cls_model.fc.in_features, len(CLASSES))
    cls_model.load_state_dict(torch.load(CLS_CHECKPOINT, map_location=DEVICE))
    print(f"Loaded CLS weights from {CLS_CHECKPOINT}")
    cls_model.eval().to(DEVICE)

    if not os.path.exists(IMG_DIR):
        print("IMG_DIR not found")
        exit()
        
    images = [x for x in os.listdir(IMG_DIR) if x.endswith(('.png', '.jpg', '.tif'))]
    print("Total Images:", len(images))

    out = {"images": []}
    for im_no, p in enumerate(images):
        if im_no % 10 == 0: print(im_no)
        path = os.path.join(IMG_DIR, p)
        im = Image.open(path).convert("RGB")
        W, H = im.size
        im_np = np.array(im)
        x = aug(image=im_np)['image'].to(DEVICE)
        cls_x = cls_aug(image=im_np)['image'].unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            y = model([x])[0]
            pred_cls = cls_model(cls_x)
            probs = torch.softmax(pred_cls, dim=1)
            pred_idx = int(probs.argmax(dim=1).cpu().item())
            pred_class = CLASSES[pred_idx]

        boxes  = y.get("boxes").detach().cpu().numpy()
        labels = y.get("labels").detach().cpu().numpy().astype(int)
        scores = y.get("scores").detach().cpu().numpy()
        masks  = y.get("masks")
        if masks is not None:
            masks = masks.detach().cpu().numpy()[:,0]

        anns = []
        for i in range(len(boxes)):
            if scores[i] < SCORE_THR: 
                continue
            
            cls = labels[i]
            if cls <= 0 or cls >= len(CLASS_NAMES):
                continue
                
            seg = []
            if masks is not None:
                # Calling the fixed Flat-List function
                seg = mask_to_polygon(masks[i] > 0.5)
                
            # Fallback if mask failed or was empty
            if not seg or len(seg) < 6:
                x1, y1, x2, y2 = boxes[i].tolist()
                # Fixed Fallback: FLAT LIST, not list of lists
                seg = [x1, y1, x2, y1, x2, y2, x1, y2]

            anns.append({
                "class": CLASS_NAMES[cls],
                "confidence_score": float(scores[i]),
                "segmentation": seg, # Now guaranteed to be [x,y,x,y...]
            })

        try:
            res = int(p.split("cm")[0])
        except:
            res = 0

        out["images"].append({
            "file_name": p,
            "width": W,
            "height": H,
            "cm_resolution": res,
            "scene_type": pred_class,
            "annotations": anns
        })

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out['images'])} images -> {OUT_JSON}")
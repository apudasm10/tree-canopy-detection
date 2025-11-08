# infer_folder_to_json.py
import os, json
from pathlib import Path
import torch.nn as nn
import torch
from torchvision import models
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection.rpn import RPNHead
# from src.dataset import CocoMaskDataset, collate_fn  # adjust import if needed

# ---------- config ----------
IMG_DIR      = "/home/vault/iwso/iwso195h/TCD/data/val"
CHECKPOINT   = "/home/vault/iwso/iwso195h/TCD/Run 12/maskrcnn_epoch_90.pth"
CLS_CHECKPOINT    = "/home/vault/iwso/iwso195h/TCD/cls_model_final2/resnet50_aerial_25.pth"
OUT_JSON     = "data/preds8.json"
CLASS_NAMES  = ["background", "individual_tree", "group_of_trees"]  # index matches model labels
CLASSES = ['agriculture_plantation','rural_area','urban_area','open_field','industrial_area']
SCORE_THR    = 0.5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- helpers ----------
def list_images(d):
    exts = {".tif",".tiff"}
    return sorted([p for p in Path(d).rglob("*") if p.suffix.lower() in exts])

def to_tensor(pil_img):
    return T.ToTensor()(pil_img)

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

def build_model(num_classes):
    m = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_cls = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_cls, num_classes)
    in_mask = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
    return m

# ---------- run ----------
if __name__ == "__main__":
    aug = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()])
    
    cls_aug = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    anchor_generator = AnchorGenerator(
        sizes=(
            (16, 32),    # P2: Has 2 sizes now
            (32, 64),    # P3: Has 2 sizes
            (64, 128),   # P4: Has 2 sizes
            (128, 256),  # P5: Has 2 sizes
            (256, 512)   # P6: Has 2 sizes
        ),
        aspect_ratios=((0.5, 0.75, 1.0, 1.33, 2.0),) * 5
    )

    num_classes = 3  # background + 2 classes
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.rpn.anchor_generator = anchor_generator
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    model.rpn.head = RPNHead(model.rpn.head.conv[0][0].in_channels, num_anchors)

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
    cls_model = cls_model.to(DEVICE)
    print(f"Loaded weights from {CLS_CHECKPOINT}")


    # model = build_model(num_classes=len(CLASS_NAMES))
    # state = torch.load(CHECKPOINT, map_location="cpu")
    # model.load_state_dict(state.get("model", state), strict=True)
    model.eval()
    model.to(DEVICE)

    cls_model.eval()
    cls_model.to(DEVICE)
    # model.roi_heads.score_thresh = 0.0
    # model.roi_heads.detections_per_img = 300

    images = list_images(IMG_DIR)
    print("Total Images:", len(os.listdir(IMG_DIR)))
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

        boxes  = y.get("boxes",  torch.empty(0,4)).detach().cpu().numpy()
        labels = y.get("labels", torch.empty(0,)).detach().cpu().numpy().astype(int)
        scores = y.get("scores", torch.empty(0,)).detach().cpu().numpy()
        masks  = y.get("masks",  None)
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
            "file_name": Path(p).name,
            "width": W,
            "height": H,
            "cm_resolution": int(Path(p).name.split("cm")[0]),
            "scene_type": pred_class,
            "annotations": anns
        })

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out['images'])} images -> {OUT_JSON}")

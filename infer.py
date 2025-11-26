import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from src.dataset import CocoMaskDataset, collate_fn  # adjust import if needed
import sys

# ---------- config ----------
# color map: index 0 is background, 1..C are classes
CLASS_COLORS = np.array([
    (0,   0,   0),   # bg
    (0, 128, 255),   # class 1: individual_tree (blue-ish)
    (0, 255,   0),   # class 2: group_of_trees (green)
], dtype=np.uint8)

# ---------- helpers ----------
def to_pil(img_tensor):
    """[3,H,W] float in [0,1] -> PIL RGB"""
    arr = (img_tensor.clamp(0, 1).mul(255).byte().cpu().numpy().transpose(1, 2, 0))
    return Image.fromarray(arr)

def class_union_from_instances(masks, labels, out_h, out_w, thr=0.5):
    """
    masks: Tensor [N,1,h,w] or [N,h,w] in [0,1]
    labels: Tensor [N] with class ids (1..C)
    returns uint8 [H,W] class-id mask (union per class)
    """
    if masks.ndim == 4:
        masks = masks.squeeze(1)  # [N,h,w]
    if masks.shape[-2:] != (out_h, out_w):
        masks = F.interpolate(masks.unsqueeze(1), size=(out_h, out_w), mode="nearest").squeeze(1)
    binm = (masks >= thr).detach().cpu().numpy()  # [N,H,W] bool
    labels = labels.detach().cpu().numpy()

    out = np.zeros((out_h, out_w), dtype=np.uint8)
    for cid in np.unique(labels):
        cls_mask = binm[labels == cid]    # [n_c,H,W]
        if cls_mask.size == 0:
            continue
        out[cls_mask.any(axis=0)] = np.uint8(cid)
    return out  # [H,W] values in {0..C}

def colorize_class_mask(label_mask):
    """uint8 [H,W] -> PIL RGB using CLASS_COLORS"""
    return Image.fromarray(CLASS_COLORS[label_mask])

def overlay(pil_img, pil_mask_rgb, alpha=0.45):
    return Image.blend(pil_img.convert("RGB"), pil_mask_rgb.convert("RGB"), alpha)

# ---------- inference & save ----------
def save_gt_pred_vis(model, dataset, idx, out_path,
                     device="cuda", train_size=512, export_size=1024,
                     score_thr=0.5, bin_thr=0.5):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.eval().to(device)

    img, tgt = dataset[idx]                 # img [3,train,train] float in [0,1]
    Ht, Wt = img.shape[1:]
    assert (Ht, Wt) == (train_size, train_size), "Dataset should output resized tensors"

    with torch.no_grad():
        pred = model([img.to(device)])[0]

    # base panels (scaled to export size)
    left  = to_pil(img).resize((export_size, export_size), Image.BILINEAR)
    right = to_pil(img).resize((export_size, export_size), Image.BILINEAR)

    # --- GT: union-per-class from target masks/labels
    if tgt["masks"].numel() > 0:
        gt_label = class_union_from_instances(
            tgt["masks"].float(), tgt["labels"], export_size, export_size, thr=bin_thr
        )
    else:
        gt_label = np.zeros((export_size, export_size), dtype=np.uint8)
    gt_rgb = colorize_class_mask(gt_label)

    # --- PRED: filter by score, then union-per-class
    scores = pred.get("scores", torch.tensor([])).detach().cpu()
    keep = (scores >= score_thr).nonzero(as_tuple=False).flatten()
    if keep.numel() > 0:
        pmasks = pred["masks"][keep]
        plabels = pred["labels"][keep]
        pr_label = class_union_from_instances(pmasks, plabels, export_size, export_size, thr=bin_thr)
    else:
        pr_label = np.zeros((export_size, export_size), dtype=np.uint8)
    pr_rgb = colorize_class_mask(pr_label)

    print(pred)
    print("---------------------------------")
    # overlays
    left_overlay  = overlay(left,  gt_rgb, alpha=0.45)
    right_overlay = overlay(right, pr_rgb, alpha=0.45)

    # side-by-side save
    canvas = Image.new("RGB", (export_size * 2+5, export_size), color="#ffffff")
    canvas.paste(left_overlay,  (0, 0))
    canvas.paste(right_overlay, (export_size+5, 0))
    canvas.save(out_path)
    return out_path

# ---------- run ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset (use same resize as training)
    IMG_SIZE = 1024
    dataset = CocoMaskDataset(
        img_dir="/home/vault/iwso/iwso195h/TCD/data/train",
        ann_file="data/coco_annotations.json",
        resize=(IMG_SIZE, IMG_SIZE),
        augment=False
    )

    # model (+heads) must match training exactly
    num_classes = 3  # background + 2 classes
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    if len(sys.argv)==1:
        idx = 1
        run = 1
        run_model = 50
    elif len(sys.argv)==2:
        idx = sys.argv[1]
        run = 1
        run_model = 50
    elif len(sys.argv)==3:
        idx = sys.argv[1]
        run = sys.argv[2]
        run_model = 50
    else:
        idx = sys.argv[1]
        run = sys.argv[2]
        run_model = sys.argv[3]

    checkpoint = f"/home/vault/iwso/iwso195h/TCD/Run Final2/maskrcnn_epoch_{run_model}.pth"
    out_path = f"viz/gt_pred_{idx}_{run}_{run_model}.png"

    # checkpoint = "/home/vault/iwso/iwso195h/TCD/Run 1/maskrcnn_epoch_50.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    print(f"Loaded weights from {checkpoint}")

    output = save_gt_pred_vis(
        model=model,
        dataset=dataset,
        idx=int(idx),                    # pick an index
        out_path=out_path,
        device=device,
        train_size=IMG_SIZE,
        export_size=1024,          # upsample panels to 1024×1024
        score_thr=0.5,
        bin_thr = 0.5
    )
    print("Saved:", output)

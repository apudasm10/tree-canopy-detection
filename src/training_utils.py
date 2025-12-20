import torch
from torch.amp import autocast
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
import csv
import wandb


def train_one_epoch(model, loader, opt, scaler, device, clip=5.0, desc=""):
    model.train()
    running, n = 0.0, 0
    pbar = tqdm(loader, total=len(loader), desc=desc, leave=False)
    for imgs, targets in pbar:
        if imgs is None:
            continue
        imgs = [im.to(device, non_blocking=True) for im in imgs]

        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        opt.zero_grad(set_to_none=True)
        with autocast('cuda'):
            losses = model(imgs, targets)
            loss = sum(losses.values())

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if clip:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
        scaler.step(opt); scaler.update()

        running += loss.item(); n += 1
        # Optional: show a tiny postfix without spamming
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return running / max(1, n)  # average train loss


def evaluate(model, loader, device, desc=""):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', iou_type="segm", max_detection_thresholds=[1, 100, 1000])
    pbar = tqdm(loader, total=len(loader), desc=desc, leave=False)
    with torch.no_grad():
        for imgs, targets in pbar:
            if imgs is None:
                continue
            imgs = [im.to(device, non_blocking=True) for im in imgs]

            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with autocast('cuda'):
                preds = model(imgs)
                preds_formatted = []
                targets_formatted = []

                for p, t in zip(preds, targets):
                    preds_formatted.append({
                        "boxes": p["boxes"].cpu(),
                        "scores": p["scores"].cpu(),
                        "labels": p["labels"].cpu(),
                        "masks": (p["masks"] > 0.5).squeeze(1).cpu().bool() # <--- CRITICAL
                    })
                    
                    targets_formatted.append({
                        "boxes": t["boxes"].cpu(),
                        "labels": t["labels"].cpu(),
                        "masks": t["masks"].cpu().bool()
                    })
                metric.update(preds_formatted, targets_formatted)

    results = metric.compute()

    print(f"mAP@[.50:.05:.95]: {results['map']}")
    print(f"mAP@.50: {results['map_50']}")
    print(f"mAP [Small]: {results['map_small']}")
    print(f"mAP [Medium]: {results['map_medium']}")
    print(f"mAP [Large]: {results['map_large']}")
    metric.reset()

    return results


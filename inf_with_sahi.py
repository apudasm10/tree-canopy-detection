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
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
from sahi.utils.cv import visualize_object_predictions
import matplotlib.pyplot as plt
from src.utils import padded_img

torch.cuda.empty_cache()
CHECKPOINT = "models/Mask_RCNN_Run_1/maskrcnn_epoch_45.pth"
OUT_JSON = "data/preds_dino_new.json"
CLASS_NAMES  = ["bg", "individual_tree", "group_of_trees"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_SIZE=512
OVERLAP_RATIO = 0.25
OVERLAP_PIXELS = int(CROP_SIZE * OVERLAP_RATIO)
STRIDE = CROP_SIZE - OVERLAP_PIXELS

print(f"crop_size: {CROP_SIZE}, overlap_pixels: {OVERLAP_PIXELS}, stride: {STRIDE}")

aug = A.Compose([
    A.ToFloat(max_value=255.0),
    ToTensorV2()
])

num_classes = 3
last_level = "LastLevelMaxPool"
model_name = "convnext_small.dinov3_lvd1689m"
backbone = CustomBackbone(model_name, False)

fpn = CustomFPN(backbone.in_channels_list, 256, backbone.feature_module_names, last_level=last_level)

pipe = CustomModelFPN(backbone, fpn)

if last_level == "LastLevelMaxPool":
    featmap_names_box=["p2", "p3", "p4", "p5", "pool"]
    featmap_names_mask=["p2", "p3", "p4", "p5"]
    sizes=((8, 16,), (32, 48), (64, 96), (128, 256), (384, 512))
    aspect_ratios=((0.75, 1.0, 1.33),) * 5
elif last_level == "LastLevelP6P7":
    featmap_names_box=["p2", "p3", "p4", "p5", "p6", "p7"]
    featmap_names_mask=["p2", "p3", "p4", "p5"]
    sizes=((8, 16,), (32, 48), (64, 96), (128, 192), (256, 384), (512, 1024))
    aspect_ratios=((0.75, 1.0, 1.33),) * 6
else:
    print("Define a valid last_level: [LastLevelMaxPool, LastLevelP6P7]")

anchor_generator = AnchorGenerator(
    sizes=sizes,
    aspect_ratios=aspect_ratios
)

box_roi_pool = MultiScaleRoIAlign(
        featmap_names=featmap_names_box,
        output_size=7,
        sampling_ratio=2,
    )

mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=featmap_names_mask,
        output_size=28,
        sampling_ratio=2,
    )

MAX_DETS = 300
model = MaskRCNN(
    pipe,
    num_classes=3,
    box_roi_pool=box_roi_pool,
    mask_roi_pool=mask_roi_pool,
    rpn_anchor_generator=anchor_generator,
    box_detections_per_img=MAX_DETS,
    min_size=CROP_SIZE,
    max_size=CROP_SIZE,
    rpn_pre_nms_top_n_train=3000,
    rpn_pre_nms_top_n_test=2000,
    rpn_post_nms_top_n_train=2000,
    rpn_post_nms_top_n_test=1500,
)

print("Model created.")

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

print(f"Loading weights from {CHECKPOINT}")
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE), strict=True)
model.eval().to(DEVICE)

detection_model = AutoDetectionModel.from_pretrained(
        model_type='torchvision',
        model=model,
        confidence_threshold=0.3,
        device=DEVICE,
        category_mapping={str(i): CLASS_NAMES[i] for i in range(1, len(CLASS_NAMES))},
        # category_remapping={CLASS_NAMES[i]: i for i in range(1, len(CLASS_NAMES))}
    )

image_path = "data/10cm_train_13.tif"
img = padded_img(image_path, target_gsd=20.0, crop_size=CROP_SIZE, stride_pixels=STRIDE)

result = get_sliced_prediction(
        img,
        detection_model,
        slice_height=CROP_SIZE,
        slice_width=CROP_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        verbose=2,
    )

print(result)

class_count = {}

for object_prediction in result.object_prediction_list:
    print(f"Label: {object_prediction.category.name}")
    c = class_count.get(object_prediction.category.name, 0)
    class_count[object_prediction.category.name] = c + 1

    print(f"Score: {object_prediction.score.value:.2f}")
    print(f"Box: {object_prediction.bbox.to_xyxy()}") # [x1, y1, x2, y2]
    print("-" * 20)

print("Class counts:", class_count)

visual_result = visualize_object_predictions(
    image=np.array(result.image),
    object_prediction_list=result.object_prediction_list,
    hide_labels=True,  # Crucial: Hides text labels so you can see the trees
    hide_conf=True,    # Crucial: Hides confidence scores
    rect_th=1,         # Thinner boxes
    text_th=1
)

print("Visualization done.")
# 2. Show it
plt.figure(figsize=(15, 15))
plt.imshow(visual_result["image"][0:1024, 0:1024, :])
plt.axis("off")
plt.savefig("prediction.png")
print("Saved prediction visualization to prediction.png")

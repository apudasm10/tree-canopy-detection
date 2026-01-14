#import libraries and define paths
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch
import os
import math
import supervision as sv


def generate_mask(item):
    colors = {'individual_tree': 1, 'group_of_trees': 2}
    polygons = {}

    for annot in item['annotations']:
        cls = annot["class"]
        temp = polygons.setdefault(cls, [])
        temp.append(annot["segmentation"])
        polygons[cls] = temp

    img = Image.new("L", size=(item["width"], item["height"]))
    img1 = ImageDraw.Draw(img)
    for type_of_trees in polygons:
        for polygon in polygons[type_of_trees]:
            img1.polygon(polygon, fill =colors[type_of_trees])

    return img

mask_to_rgb_mapper_3 = np.array([
    (0, 0, 0),  # Background - White
    (255, 127, 0),  # Wound Border
    (255, 255, 255),  # Granulation
], dtype=np.uint8)

def show_img(image, mask_image, overlay=False):
    to_pil = transforms.ToPILImage()
    image = to_pil(image)
    mask_np = np.array(mask_image)
    segmented_rgb = mask_to_rgb_mapper_3[mask_np]
    segmented_rgb = Image.fromarray(segmented_rgb)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis('off')

    if overlay:
        segmented_rgb = Image.blend(image, segmented_rgb, .7)

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_rgb)
    plt.title("Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def padded_img(image_path, target_gsd=20.0, crop_size=512, stride_pixels=384):
    print(f"crop_size: {crop_size}, stride_pixels: {stride_pixels}")
    current_gsd = int(os.path.basename(image_path).split("cm")[0])
    # current_gsd = int(image_path.split("cm")[0])
    img = cv2.imread(image_path)
    scale = current_gsd / target_gsd
    print(f"Current GSD: {current_gsd} cm/pixel, Target GSD: {target_gsd} cm/pixel, Scale: {scale}")
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    h, w = img.shape[:2]
    print(f"Original image size: w={w}, h={h}")

    steps_w = math.ceil(max(0, w - crop_size) / stride_pixels)
    steps_h = math.ceil(max(0, h - crop_size) / stride_pixels)
    print(f"Steps needed - Width: {steps_w}, Height: {steps_h}")
    
    target_w = crop_size + (steps_w * stride_pixels)
    target_h = crop_size + (steps_h * stride_pixels)
    print(f"Target image size: w={target_w}, h={target_h}")
    
    pad_right = int(target_w - w)
    pad_bottom = int(target_h - h)

    print(f"pad_right: {pad_right}, pad_bottom: {pad_bottom}")
    
    # Add black border
    img_padded = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])

    return img_padded

def combine_with_nms(detections_list, iou_threshold=0.45, class_agnostic=True):
    combined = sv.Detections.merge(detections_list)
    combined_nms = combined.with_nms(threshold=iou_threshold, class_agnostic=class_agnostic)
    
    return combined_nms

def get_class(models, image, transform, class_names, device='cpu'):
    img_transformed = transform(image).unsqueeze(0).to(device)

    ensemble_probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(img_transformed)
            probs = torch.softmax(outputs, dim=1)
            ensemble_probs.append(probs)

    avg_probs = torch.stack(ensemble_probs).mean(dim=0)
    _, predicted_class = avg_probs.max(1)
    predicted_class_name = class_names[predicted_class.item()]

    return predicted_class_name
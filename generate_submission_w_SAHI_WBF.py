import os, json, cv2, torch
import torch.nn as nn
import supervision as sv
from ultralytics import YOLO, SAM
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from PIL import Image
from src.utils import get_class 
from tqdm import tqdm
import gc
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from src.utils import sahi_to_sv_detections, run_wbf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ID_TO_CLASS = {0: "individual_tree", 1: "group_of_trees"}
NUM_CLASSES = 5
IMG_DIR = os.path.join(os.getenv("HPCVAULT"), "TCD", "data", "val")
CLASS_NAMES = ["agriculture_plantation", "urban_area", "industrial_area", "rural_area", "open_field"]
output_json = 'submission_wbf_tta_sahi_1216_gsd_fix.json'

pipeline_data = {}
all_img_names = sorted(os.listdir(IMG_DIR))

for name in all_img_names:
    pipeline_data[name] = {"scene": None, "boxes": None}

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# ==============================================================================
# STAGE 1: SCENE CLASSIFICATION
# ==============================================================================
print("\n--- STAGE 1: SCENE CLASSIFICATION ---")
clear_gpu()

classifier_model_path = [
    "cls_models/convnext_run1_best.pth",
    "cls_models/convnext_run2_best.pth",
    "cls_models/convnext_run3_best.pth",
]
classifier_models = []
for path in classifier_model_path:
    m = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(path))
    m.eval().to(DEVICE)
    classifier_models.append(m)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with torch.no_grad():
    for img_name in tqdm(all_img_names, desc="Classifying Scenes"):
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img_rgb)
        
        scene_type = get_class(classifier_models, image_pil, transform, CLASS_NAMES, DEVICE)
        pipeline_data[img_name]["scene"] = scene_type

del classifier_models
del transform
clear_gpu()

# ==============================================================================
# STAGE 2: YOLO BOX DETECTION (SAHI + WBF + GSD SCALING)
# ==============================================================================
print("\n--- STAGE 2: YOLO BOX DETECTION (SAHI + WBF) ---")
clear_gpu()

# these models are trained on crops images for GSD >= 60cm. these images resized to 1792x1792 and then cropped to 1024x1024 with 40% overlap
yolo_model_paths = ['GSD_Normalized/yolov8m-p2.yaml32/weights/best.pt', 'GSD_Normalized/yolo11x_exp48/weights/best.pt']
sahi_models = []

print("Loading SAHI Models...")
for path in yolo_model_paths:
    model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics', 
        model_path=path,
        confidence_threshold=0.3,
        device=str(DEVICE)
    )
    sahi_models.append(model)

for img_name in tqdm(all_img_names, desc="Detecting Boxes"):
    img_path = os.path.join(IMG_DIR, img_name)
    
    img_bgr = cv2.imread(img_path)
    h_orig, w_orig = img_bgr.shape[:2]
    
    gsd = int(img_name[:2])
    
    if gsd >= 60:
        target_size = 1792
        img_inference = cv2.resize(img_bgr, (target_size, target_size))
        scale_factor = target_size / w_orig # e.g. 1.75
    else:
        img_inference = img_bgr
        scale_factor = 1.0

    all_detections = []
    
    for model in sahi_models:
        result = get_sliced_prediction(
            img_inference,
            model,
            slice_height=1024,
            slice_width=1024,
            overlap_height_ratio=0.25,
            overlap_width_ratio=0.25,
            verbose=0, 
            perform_standard_pred=False 
        )
        
        dets = sahi_to_sv_detections(result)
        
        if scale_factor != 1.0:
            dets.xyxy /= scale_factor

        all_detections.append(dets)

    combined_dets = run_wbf(all_detections, image_shape=(h_orig, w_orig), iou_thr=0.55, skip_box_thr=0.25)
    
    combined_dets.mask = None 
    pipeline_data[img_name]["boxes"] = combined_dets

del sahi_models
clear_gpu()

# ==============================================================================
# STAGE 3: SAM SEGMENTATION & SAVE
# ==============================================================================
print("\n--- STAGE 3: SAM SEGMENTATION & SAVE ---")
clear_gpu()

sam_model = SAM('sam2.1_b.pt')

submission_data = {"images": []}

for img_name in tqdm(all_img_names, desc="Segmenting with SAM"):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path) 
    
    scene_type = pipeline_data[img_name]["scene"]
    combined_dets = pipeline_data[img_name]["boxes"]

    image_info = {
        "file_name": img_name,
        "width": img.shape[1],
        "height": img.shape[0],
        "cm_resolution": int(img_name[:2]),
        "scene_type": scene_type,
        "annotations": []
    }

    if len(combined_dets) > 0:
        with torch.no_grad():
            sam_results = sam_model.predict(
                source=img_path,
                bboxes=combined_dets.xyxy,
                verbose=False,
                device=DEVICE
            )
        
        if sam_results[0].masks is not None:
            masks_cpu = sam_results[0].masks.data.cpu().numpy().astype(bool)
            
            num_dets = len(combined_dets)
            num_masks = len(masks_cpu)
            
            combined_dets.mask = masks_cpu
            safe_loop_limit = min(num_dets, num_masks)
            
            for i in range(safe_loop_limit):
                if combined_dets.mask[i] is None: 
                    continue
                
                polygons = sv.mask_to_polygons(combined_dets.mask[i])
                
                if not polygons:
                    continue
                    
                poly_coords = polygons[0].flatten().astype(int).tolist()
                
                if len(poly_coords) < 6: 
                    continue 

                ann = {
                    "class": ID_TO_CLASS[combined_dets.class_id[i]],
                    "confidence_score": round(float(combined_dets.confidence[i]), 4),
                    "segmentation": poly_coords
                }
                image_info["annotations"].append(ann)
    
    submission_data["images"].append(image_info)
    
    clear_gpu()

with open(output_json, 'w') as f:
    json.dump(submission_data, f, indent=4)

print(f"Successfully generated {output_json}")
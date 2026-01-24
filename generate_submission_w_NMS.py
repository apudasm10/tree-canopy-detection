import os, json, cv2, torch
import torch.nn as nn
import supervision as sv
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from PIL import Image
from src.utils import combine_with_nms, get_class
from tqdm import tqdm


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ID_TO_CLASS = {0: "individual_tree", 1: "group_of_trees"}
NUM_CLASSES = 5
IMG_DIR = 'tree-canopy-detection/val'
CLASS_NAMES = ["agriculture_plantation", "urban_area", "industrial_area", "rural_area", "open_field"]
output_json = 'submission.json'
submission_data = {"images": []}

yolo_model_paths = [
    "seg_models/bestf1.pt",
    "seg_models/bestf2.pt"
]

classifier_model_path = [
    "cls_models/convnext_run1_best.pth",
    "cls_models/convnext_run2_best.pth",
    "cls_models/convnext_run3_best.pth",
]

yolo_models = [YOLO(path) for path in yolo_model_paths]

classifier_model = []

for path in classifier_model_path:
    m = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(path))
    m.eval()
    m = m.to(DEVICE)
    classifier_model.append(m)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

all_img_names = os.listdir(IMG_DIR)
for img_name in tqdm(all_img_names, desc="Generating Submission"):
    torch.cuda.empty_cache()
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image_pil = Image.fromarray(img_rgb)
    scene_type = get_class(classifier_model, image_pil, transform, CLASS_NAMES, DEVICE)

    image_info = {
        "file_name": img_name,
        "width": img.shape[1],
        "height": img.shape[0],
        "cm_resolution": int(img_name[:2]),
        "scene_type": scene_type,
        "annotations": []
    }

    all_detections = []
    for model in yolo_models:
        results = model.predict(source=img_path, imgsz=896, conf=0.25, retina_masks=True, max_det=2000, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        all_detections.append(detections)

    combined_dets = combine_with_nms(all_detections, iou_threshold=0.45, class_agnostic=True)

    for i in range(len(combined_dets)):
        polygons = sv.mask_to_polygons(combined_dets.mask[i])
        if not polygons:
            print(f"No polygons found for detection {i} in image {img_name}, skipping annotation.")
            continue
        poly_coords = polygons[0].flatten().astype(int).tolist()

        ann = {
            "class": ID_TO_CLASS[combined_dets.class_id[i]],
            "confidence_score": round(float(combined_dets.confidence[i]), 4),
            "segmentation": poly_coords
        }
        image_info["annotations"].append(ann)

    submission_data["images"].append(image_info)

with open(output_json, 'w') as f:
    json.dump(submission_data, f, indent=4)

print(f"Successfully generated {output_json} with {len(submission_data['images'])} images.")
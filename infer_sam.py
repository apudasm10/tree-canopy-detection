import torch
import matplotlib.pyplot as plt
import cv2
import supervision as sv
from ultralytics import YOLO, SAM
import gc
gc.collect()
torch.cuda.empty_cache()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
m1 = YOLO('GCP_TCD/yolov8x_exp32/weights/best.pt')
m2 = YOLO('GCP_TCD/yolo11l_exp44/weights/best.pt')

img = "tree-canopy-detection/train/10cm_train_32.tif"

detections_list = []
res1 = m1.predict(img, imgsz=1024, conf=0.25, max_det=2000)
d = sv.Detections.from_ultralytics(res1[0])
detections_list.append(d)

res2 = m2.predict(img, imgsz=1024, conf=0.25, max_det=2000)
d = sv.Detections.from_ultralytics(res2[0])
detections_list.append(d)

combined = sv.Detections.merge(detections_list)
combined = combined.with_nms(threshold=0.4, class_agnostic=True)

sam_model = SAM('sam2.1_b.pt')

prompt_boxes = combined.xyxy

sam_results = sam_model.predict(
    source=img, 
    bboxes=prompt_boxes, 
    device=device
)

combined.mask = sam_results[0].masks.data.cpu().numpy().astype(bool)

image = cv2.imread(img)
my_palette = sv.ColorPalette.from_hex(["#FFFF00", "#0000FF"])

mask_annotator = sv.MaskAnnotator(color=my_palette, opacity=0.4)
# box_annotator = sv.BoxAnnotator(color=my_palette, thickness=2)
# label_annotator = sv.LabelAnnotator(color=my_palette, text_scale=0.4)

annotated_image = mask_annotator.annotate(scene=image.copy(), detections=combined)
# annotated_image = box_annotator.annotate(scene=annotated_image, detections=combined)
# annotated_image = label_annotator.annotate(scene=annotated_image, detections=combined)

# 6. Plot
sv.plot_image(annotated_image, size=(12, 12))
plt.savefig("result_plot5.png", bbox_inches='tight', pad_inches=0.0)
print("Saved result_plot5.png")
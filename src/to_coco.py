import json


CLASS2ID = {
    "individual_tree": 1,
    "group_of_trees": 2,
}

def polygon_bbox_and_area(poly):
    """
    poly: flat [x1,y1,x2,y2,...], len>=6, even
    returns (bbox[x,y,w,h], area) in float
    """
    xs = poly[0::2]
    ys = poly[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

    # polygon area via Shoelace formula
    area = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    area = abs(area) / 2.0
    return bbox, float(area)

def to_coco(records, class2id=None):
    """
    records: list of your per-image dicts:
        {
          "file_name": "...tif",
          "width": 1024,            # optional (will still work if missing)
          "height": 1024,           # optional
          "scene_type": "...",      # optional, preserved
          "cm_resolution": 10,      # optional, preserved
          "annotations": [
            {"class": "group_of_trees", "segmentation": [x1,y1,...]},
            ...
          ]
        }
    returns: COCO dict with images/annotations/categories
    """
    if class2id is None:
        class2id = CLASS2ID

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for img_id, rec in enumerate(records, start=1):
        w = rec.get("width", None)
        h = rec.get("height", None)

        # image entry (extra keys are fine to include)
        img_entry = {
            "id": img_id,
            "file_name": rec["file_name"],
        }
        if w is not None: img_entry["width"]  = int(w)
        if h is not None: img_entry["height"] = int(h)
        # preserve your extra metadata (optional for COCO)
        if "scene_type" in rec:     img_entry["scene_type"] = rec["scene_type"]
        if "cm_resolution" in rec:  img_entry["cm_resolution"] = rec["cm_resolution"]

        coco_images.append(img_entry)

        for ann in rec.get("annotations", []):
            cls_name = ann.get("class")
            poly = ann.get("segmentation", [])
            if cls_name not in class2id:
                continue
            if not isinstance(poly, list):
                continue
            if len(poly) < 6 or len(poly) % 2 != 0:
                continue

            bbox, area = polygon_bbox_and_area(poly)
            cat_id = class2id[cls_name]

            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": [poly],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1

    coco_categories = [{"id": cid, "name": name} for name, cid in class2id.items()]
    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories
    }
    return coco


if __name__ == "__main__":
    with open("data/train_annotations.json", "r") as f:
        data = json.load(f)
    records = data["images"]

    coco = to_coco(records, CLASS2ID)
    with open("data/instances_train.json", "w") as f:
        json.dump(coco, f, indent=2)
    print("Wrote COCO to data/instances_train.json")

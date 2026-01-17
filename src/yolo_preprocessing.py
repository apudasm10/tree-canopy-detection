import os
import shutil
from tqdm import tqdm
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def process_dataset_to_yolo_v2(img_dir, ann_file, output_dir, gsd_weight, out_file="all_train.txt", over_sample=True, copy=True):
    dest_labels_dir = os.path.join(output_dir, "labels", "train")
    dest_images_dir = os.path.join(output_dir, "images", "train")

    os.makedirs(dest_labels_dir, exist_ok=True)
    if copy:
        os.makedirs(dest_images_dir, exist_ok=True)

    print(f"Reading: {ann_file}")
    with open(ann_file, 'r') as f:
        data = json.load(f)

    class_map = {"individual_tree": 0, "group_of_trees": 1}
    all_yolo_lines = []

    images_list = data.get('images', [])
    print(f"Found {len(images_list)} images. Starting conversion to YOLO Detection (BBox)...")

    for item in tqdm(images_list, desc="Converting"):
        file_name = item['file_name']
        img_w = item['width']
        img_h = item['height']

        current_image_labels = []

        for ann in item.get('annotations', []):
            if ann['class'] not in class_map:
                continue

            class_id = class_map[ann['class']]
            
            # Convert segmentation list to numpy array [[x,y], [x,y]...]
            segmentation = ann['segmentation']
            polygon = np.array(segmentation).reshape(-1, 2)
            
            polygon[:, 0] = polygon[:, 0] / img_w
            polygon[:, 1] = polygon[:, 1] / img_h
            
            # We clamp inputs so width/height are calculated from valid 0-1 values
            x_min = np.clip(np.min(polygon[:, 0]), 0.0, 1.0)
            y_min = np.clip(np.min(polygon[:, 1]), 0.0, 1.0)
            x_max = np.clip(np.max(polygon[:, 0]), 0.0, 1.0)
            y_max = np.clip(np.max(polygon[:, 1]), 0.0, 1.0)
            
            # 3. Calculate Width/Height from clamped values
            n_width = x_max - x_min
            n_height = y_max - y_min
            
            # Filter Invalid Boxes (Too small OR Zero area)
            if n_width < 0.001 or n_height < 0.001 or n_width > 1.0 or n_height > 1.0:
                print(f"Skipping invalid box: {n_width:.6f}x{n_height:.6f}")
                continue

            n_x_center = x_min + (n_width / 2)
            n_y_center = y_min + (n_height / 2)

            # if n_x_center < 0 or n_x_center > 1.0 or n_y_center < 0 or n_y_center > 1.0:
            #     print(f"Skipping invalid center: {n_x_center:.6f}, {n_y_center:.6f}")

            # # 6. Final Safety Clamp (Just to be paranoid)
            # n_x_center = np.clip(n_x_center, 0.0, 1.0)
            # n_y_center = np.clip(n_y_center, 0.0, 1.0)

            # 7. Add to list
            current_image_labels.append(
                f"{class_id} {n_x_center:.6f} {n_y_center:.6f} {n_width:.6f} {n_height:.6f}"
            )

        if len(current_image_labels) == 0:
            print(f"Warning: No valid labels for image {file_name}, skipping label file creation.")
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        with open(os.path.join(dest_labels_dir, txt_name), "w") as f:
            if current_image_labels:
                f.write("\n".join(current_image_labels))

        src_image_path = os.path.join(img_dir, file_name)

        if copy:
            final_image_path = os.path.join(dest_images_dir, file_name)
            if not os.path.exists(final_image_path):
                shutil.copy2(src_image_path, final_image_path)
        else:
            final_image_path = src_image_path

        repeat_count = 1
        if over_sample:
            repeat_count = gsd_weight.get(file_name[:2], 1)

        abs_path = os.path.abspath(final_image_path)
        for _ in range(repeat_count):
            all_yolo_lines.append(abs_path)

    # Write the master training list file
    out_file_path = os.path.join(output_dir, out_file)
    with open(out_file_path, "w") as f:
        f.write("\n".join(all_yolo_lines))

    print("Done!")
    print(f"All Train samples saved to: {out_file_path} ({len(all_yolo_lines)} samples)")


def train_val_split(all_train_file, val_size=0.2, random_state=42, output_train_file="train_final.txt", output_val_file="val_final.txt"):
    print(f"Reading {all_train_file}...")
    with open(all_train_file, "r") as f:
        all_lines = [line.strip() for line in f if line.strip()]

    unique_paths = list(set(all_lines))
    print(f"Total lines: {len(all_lines)} | Unique images: {len(unique_paths)}")

    gsd_map = []
    for p in unique_paths:
        g = os.path.basename(p)[:2]
        gsd_map.append(g)

    _, val_imgs = train_test_split(unique_paths, test_size=val_size, random_state=random_state, shuffle=True, stratify=gsd_map)

    final_train = []
    final_val = val_imgs

    for line in all_lines:
        if line not in final_val:
            final_train.append(line)

    with open(output_train_file, "w") as f:
        f.write("\n".join(final_train))

    with open(output_val_file, "w") as f:
        f.write("\n".join(final_val))

    return final_train, final_val

def process_dataset_to_yolo(img_dir, ann_file, output_dir, gsd_weight, out_file="all_train.txt", over_sample=True, copy=True):
    dest_labels_dir = os.path.join(output_dir, "labels", "train")
    dest_images_dir = os.path.join(output_dir, "images", "train")

    os.makedirs(dest_labels_dir, exist_ok=True)
    if copy:
        os.makedirs(dest_images_dir, exist_ok=True)

    print(f"Reading: {ann_file}")
    with open(ann_file, 'r') as f:
        data = json.load(f)

    class_map = {"individual_tree": 0, "group_of_trees": 1}
    all_yolo_lines = []

    images_list = data.get('images', [])
    print(f"Found {len(images_list)} images. Starting conversion...")

    for item in tqdm(images_list, desc="Converting"):
        file_name = item['file_name']
        width = item['width']
        height = item['height']

        current_image_labels = []

        for ann in item.get('annotations', []):
            if ann['class'] not in class_map:
                continue

            class_id = class_map[ann['class']]
            segmentation = ann['segmentation']

            norm_coords = []
            for i in range(0, len(segmentation), 2):
                x = min(max(segmentation[i] / width, 0.0), 1.0)
                y = min(max(segmentation[i + 1] / height, 0.0), 1.0)
                norm_coords.append(f"{x:.6f} {y:.6f}")

            current_image_labels.append(f"{class_id} {' '.join(norm_coords)}")

        txt_name = os.path.splitext(file_name)[0] + ".txt"
        with open(os.path.join(dest_labels_dir, txt_name), "w") as f:
            if current_image_labels:
                f.write("\n".join(current_image_labels))

        src_image_path = os.path.join(img_dir, file_name)

        if copy:
            final_image_path = os.path.join(dest_images_dir, file_name)
            if not os.path.exists(final_image_path):
                shutil.copy2(src_image_path, final_image_path)
        else:
            final_image_path = src_image_path

        repeat_count = 1
        if over_sample:
            repeat_count = gsd_weight.get(file_name[:2], 1)

        abs_path = os.path.abspath(final_image_path)

        for _ in range(repeat_count):
            all_yolo_lines.append(abs_path)

    out_file_path = os.path.join(output_dir, out_file)
    with open(out_file_path, "w") as f:
        f.write("\n".join(all_yolo_lines))

    print("Done!")
    print(f"All Train samples saved to: {out_file_path} ({len(all_yolo_lines)} samples)")


def prepare_classification_data(source_dir, dest_dir, json_path, weights, val_size=0.2, stratify=True):
    """Parses JSON labels and organizes images into YOLO structure using os."""
    
    train_check = os.path.join(dest_dir, 'train')
    if os.path.exists(train_check) and len(os.listdir(train_check)) > 0:
        print(f"Data exists in {dest_dir}. Skipping.")
        return

    print(f"Reading labels from {json_path}...")
    with open(json_path) as f:
        data = json.load(f)
    
    df = pd.DataFrame([
        {'filename': item['file_name'], 'label': item['scene_type']} 
        for item in data['images']
    ])

    print(f"Splitting data (Stratify={stratify})...")
    strat_col = df['label'] if stratify else None
    train_df, val_df = train_test_split(df, test_size=val_size, stratify=strat_col, random_state=42)

    for split, subset in [('train', train_df), ('val', val_df)]:
        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Processing {split}"):
            label = row['label']
            
            target_dir = os.path.join(dest_dir, split, label)
            os.makedirs(target_dir, exist_ok=True)
            
            src = os.path.join(source_dir, row['filename'])
            
            if not os.path.exists(src):
                continue
            
            num_copies = 1
            if split == 'train':
                num_copies = int(weights.get(label, 1))
            
            for i in range(num_copies):
                new_name = row['filename'] if i == 0 else f"copy_{i}_{row['filename']}"
                dst = os.path.join(target_dir, new_name)
                
                shutil.copy2(src, dst)

    print("\nDone! Dataset Stats:")
    for split in ['train', 'val']:
        count = sum([len(files) for r, d, files in os.walk(os.path.join(dest_dir, split))])
        print(f" - {split.upper()}: {count} images total.")
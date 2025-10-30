# Requires: 
# pip install numpy pycocotools

import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

class CocoMetrics:
    """
    A wrapper class to calculate COCO-style mAP metrics for
    instance segmentation.
    """
    def __init__(self, annotation_file):
        """
        Args:
            annotation_file (str): Path to the COCO-style ground truth
                                   annotations JSON file.
        """
        # Load ground truth annotations
        self.coco_gt = COCO(annotation_file)
        # Initialize an empty list to store prediction results
        self.results = []
        
        print(f"Loaded ground truth annotations from {annotation_file}")

    def reset(self):
        """
        Clears the results list for a new evaluation epoch.
        Call this before starting your validation loop.
        """
        self.results = []

    def update(self, image_ids, model_outputs):
        """
        Updates the results list with predictions from a batch.
        
        Args:
            image_ids (list[int]): A list of image IDs from the batch.
            model_outputs (list[dict]): A list of dictionaries, one per image,
                                        from your Mask R-CNN model.
                                        
            Expected format for each dict in `model_outputs`:
            {
                'boxes': <Tensor[N, 4] (x1, y1, x2, y2)>,
                'labels': <Tensor[N]>,
                'scores': <Tensor[N]>,
                'masks': <Tensor[N, 1, H, W] or [N, H, W] (float, >0.5 for binary)>
            }
        """
        # Iterate over each image's predictions in the batch
        for img_id, output in zip(image_ids, model_outputs):
            
            # Move data to CPU and convert to NumPy
            # (Assumes model_outputs are PyTorch tensors)
            try:
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                masks = output['masks'].cpu().numpy()
            except Exception as e:
                print(f"Error converting model output to numpy: {e}")
                print("Make sure your model_outputs are lists of dicts with PyTorch tensors.")
                continue

            # Ensure masks are binary (0 or 1)
            # Model output is usually [N, 1, H, W] or [N, H, W] float
            if masks.ndim == 4:
                masks = masks.squeeze(1) # -> [N, H, W]
            binary_masks = (masks > 0.5).astype(np.uint8)

            # Iterate over each detected instance in the image
            for i in range(len(boxes)):
                box = boxes[i]
                label = labels[i]
                score = scores[i]
                mask = binary_masks[i]
                
                # --- 1. Convert Bounding Box Format ---
                # Model outputs [x1, y1, x2, y2]
                # COCO expects [x, y, width, height]
                x1, y1, x2, y2 = box
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                
                # --- 2. Convert Mask to RLE ---
                # The `maskUtils.encode` function requires a F-contiguous array
                # (column-major), which is the opposite of NumPy's C-contiguous
                # (row-major) default.
                mask_f_contiguous = np.asfortranarray(mask)
                # Encode the binary mask to RLE
                rle = maskUtils.encode(mask_f_contiguous)
                
                # The RLE 'counts' value can be bytes, which is not JSON serializable
                # We need to decode it to a string.
                rle['counts'] = rle['counts'].decode('utf-8')

                # --- 3. Store the result in COCO format ---
                result = {
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": float(score),
                    "segmentation": rle  # Store the RLE
                }
                self.results.append(result)

    def summarize(self, image_ids):
        """
        Computes and prints the COCO mAP metrics.
        Call this *after* your validation loop is complete.
        
        Args:
            image_ids (list[int]): The list of image IDs to evaluate against.
                                   This is crucial to ensure you only evaluate
                                   on the 'train' or 'val' split.
        
        Returns:
            dict: A dictionary of the primary COCO metrics.
        """
        if not self.results:
            print("No results to summarize. Did you call update()?")
            return {}

        print("Summarizing results...")

        if 'info' not in self.coco_gt.dataset:
            # print("Warning: 'info' key not found in COCO GT. Adding a dummy 'info' key to prevent crash.")
            self.coco_gt.dataset['info'] = {}
        
        # Load our predictions into a COCO API object
        coco_dt = self.coco_gt.loadRes(self.results)
        
        # Create a COCOeval object
        # IMPORTANT: Set iouType='segm' for instance segmentation (masks)
        # Use iouType='bbox' for object detection (boxes)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='segm')
        
        # Configure evaluation to run on ALL images in the provided list
        coco_eval.params.imgIds = image_ids
        
        # Run evaluation
        # print("Running evaluation...")
        coco_eval.evaluate()
        
        # Accumulate results
        # print("Accumulating results...")
        coco_eval.accumulate()
        
        # Print the standard COCO summary
        print("\n--- COCO mAP (Mask) Summary ---")
        coco_eval.summarize()
        print("---------------------------------")
        
        # `stats` is a 12-element numpy array
        # We can map them to a more readable dictionary
        metrics = {
            "mAP (IoU=0.50:0.95)": coco_eval.stats[0],
            "mAP (IoU=0.50)": coco_eval.stats[1],
            "mAP (IoU=0.75)": coco_eval.stats[2],
            "mAP (Small)": coco_eval.stats[3],
            "mAP (Medium)": coco_eval.stats[4],
            "mAP (Large)": coco_eval.stats[5],
            "AR (max=1)": coco_eval.stats[6],
            "AR (max=10)": coco_eval.stats[7],
            "AR (max=100)": coco_eval.stats[8],
            "AR (Small)": coco_eval.stats[9],
            "AR (Medium)": coco_eval.stats[10],
            "AR (Large)": coco_eval.stats[11],
        }
        
        return metrics

def evaluate(model, dataloader, metrics_object, device):
    """
    Runs a full evaluation loop for a given model, dataloader, and metrics object.
    This version is designed to work with a DataLoader wrapping a torch.utils.data.Subset.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader (wrapping a Subset) to evaluate on.
        metrics_object (CocoMetrics): The metrics wrapper to update.
        device (torch.device): The device to run on.
        
    Returns:
        dict: A dictionary of the computed metrics.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Reset the metrics object for this new evaluation
    metrics_object.reset()
    
    # --- This is the key change ---
    # Get the *correct* list of image IDs for this specific dataset split
    all_image_ids = []
    try:
        # 1. Get the Subset object from the dataloader
        subset_dataset = dataloader.dataset
        
        # 2. Get the indices used by this Subset
        subset_indices = subset_dataset.indices
        
        # 3. Get the *original* full dataset
        full_dataset = subset_dataset.dataset
        
        # 4. Get the full list of image info from the original dataset
        all_image_info = full_dataset.images
        
        # 5. Filter this list to get only the image_info dicts for our split
        subset_image_info = [all_image_info[i] for i in subset_indices]
        
        # 6. Extract the 'id' from each
        all_image_ids = [info['id'] for info in subset_image_info]
        
        if not all_image_ids:
            raise ValueError("No image IDs found.")
            
    except Exception as e:
        print(f"Error: Could not get image IDs from dataloader: {e}")
        print("Please ensure your dataloader wraps a Subset and the original dataset has a `.images` attribute (list of dicts).")
        return {}
    
    print(f"Evaluating on {len(all_image_ids)} images...")
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, targets in progress_bar:
            # Move images to device
            images = [img.to(device) for img in images]
            
            # Get model predictions
            outputs = model(images)
            
            # Get the image_ids from the targets
            # This is essential for matching predictions to ground truth
            image_ids_batch = [int(t['image_id'][0]) for t in targets]
            
            # Update the metrics object with the batch results
            metrics_object.update(image_ids_batch, outputs)

    # --- Summarize metrics *after* the loop is done ---
    # We pass the *full* list of image IDs for this dataset split
    # print("Summarizing results...")
    metrics_summary = metrics_object.summarize(image_ids=all_image_ids)
    
    return metrics_summary
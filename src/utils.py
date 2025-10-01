#import libraries and define paths
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


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
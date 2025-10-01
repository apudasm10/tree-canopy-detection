#import libraries and define paths
import os
import json
from src.data_utils import  *
from src.dataset import *


with open(os.path.join("data", "train_annotations.json"), 'r') as f:
    output = json.load(f)

train_transform = get_train_transform()
labeled_data = SegDataset(output["images"], "data", train_transform)
print(labeled_data[1][0].shape)

im, msk = labeled_data[94]
show_img(im, msk, True)

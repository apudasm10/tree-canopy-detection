#import libraries and define paths
from torch.utils.data import Dataset
from src.utils import  *

class SegDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = annotations_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img_path = self.data[idx]
        file_name = self.data[idx]["file_name"]
        image_path = os.path.join(self.img_dir, "train", file_name)
        rgb_image = Image.open(image_path).convert("RGB")
        mask_image = generate_mask(self.data[idx])

        rgb_image = np.array(rgb_image)
        mask_image = np.array(mask_image)

        if self.transform:
            augmented = self.transform(image=rgb_image, mask=mask_image)
            rgb_image = augmented['image']
            mask_image = augmented['mask']  # ensure it's a LongTensor for CrossEntropyLoss

        return rgb_image, mask_image
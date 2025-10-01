import os
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform():
    return A.Compose([
    A.Resize(512, 512),
    ToTensorV2()
])

def get_eval_transform():
    return A.Compose([
    A.Resize(512, 512),
    ToTensorV2()
])

def get_loaders(train_dataset, val_dataset, batch_size):
    num_cpu = os.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(8, num_cpu//2), pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(8, num_cpu//2), pin_memory=True, prefetch_factor=2)

    return train_loader, val_loader

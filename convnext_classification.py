import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, convnext_small, ConvNeXt_Small_Weights
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import TCDClassification
from tqdm import tqdm

IMG_DIR = 'tree-canopy-detection/train'
JSON_PATH = 'tree-canopy-detection/train_annotations_updated.json'
NUM_CLASSES = 5 
BATCH_SIZE = 4
EPOCHS = 7
NUM_RUNS = 5
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NAMES = ["agriculture_plantation", "urban_area", "industrial_area", "rural_area", "open_field"]

SCORING_WEIGHTS = torch.tensor([2.00, 1.50, 1.25, 1.00, 1.00]).to(DEVICE)

def create_balanced_sampler(subset):
    targets = [subset.dataset.targets[i] for i in subset.indices]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    class_weights = 1.0/(class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = TCDClassification(IMG_DIR, JSON_PATH, train_transform)
val_dataset = TCDClassification(IMG_DIR, JSON_PATH, val_transform)

sss = StratifiedShuffleSplit(n_splits=NUM_RUNS, test_size=0.25, random_state=42)

run_results = []
print("Total Class Counts:", np.unique(train_dataset.targets, return_counts=True))

for run_idx, (train_ids, val_ids) in enumerate(sss.split(np.zeros(len(train_dataset)), train_dataset.targets)):
    print(f'\n--- Run {run_idx + 1}/{NUM_RUNS} ---')
    
    train_subset = Subset(train_dataset, train_ids)
    val_subset = Subset(val_dataset, val_ids)

    print("Train Class Counts:", np.unique([train_dataset.targets[i] for i in train_ids], return_counts=True))
    print("Val Class Counts:", np.unique([val_dataset.targets[i] for i in val_ids], return_counts=True))
    
    train_sampler = create_balanced_sampler(train_subset)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=SCORING_WEIGHTS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    best_val_acc = 0
    best_model_path = f'convnext_run{run_idx+1}_best.pth'
    best_epoch = -1
    
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
        
        train_acc = 100. * train_correct / train_total
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            best_epoch = epoch + 1
        
        print(f'Epoch {epoch+1} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%')

    run_results.append(best_val_acc)
    print(f'Run {run_idx+1} Best: {best_val_acc:.2f}%')



    print(f"\nLoading Best Model ({best_model_path}) for Evaluation...")
    print("Best Epoch:", best_epoch)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Run {run_idx+1})')
    plt.savefig(f'confusion_matrix_run{run_idx+1}.png')


print("Training Complete.")
import re
import pandas as pd
import matplotlib.pyplot as plt

# CONFIGURATION
log_file_path = 'train_job.slurm.o1409496'  # Replace with your actual log filename

# 1. PARSE THE FILE
data = {}
current_epoch = None
context = None  # 'train' or 'val'

# Regex patterns
loss_pattern = re.compile(r"Epoch \[(\d+)/\d+\] - Avg Loss: ([\d\.]+)")
map_pattern = re.compile(r"mAP \(IoU=0.50:0.95\): ([\d\.]+)")

try:
    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Extract Epoch and Loss
            loss_match = loss_pattern.search(line)
            if loss_match:
                epoch = int(loss_match.group(1))
                loss = float(loss_match.group(2))
                if epoch not in data:
                    data[epoch] = {'Epoch': epoch, 'Train_mAP': None, 'Val_mAP': None}
                data[epoch]['Loss'] = loss
                current_epoch = epoch
            
            # Detect Context (Train vs Validation headers)
            if "Train Metrics" in line:
                context = 'train'
            elif "Validation Metrics" in line:
                context = 'val'
                
            # Extract mAP Score
            map_match = map_pattern.search(line)
            if map_match and context and current_epoch:
                value = float(map_match.group(1))
                if context == 'train':
                    data[current_epoch]['Train_mAP'] = value
                elif context == 'val':
                    data[current_epoch]['Val_mAP'] = value

except FileNotFoundError:
    print(f"Error: Could not find file '{log_file_path}'")
    exit()

# 2. CREATE DATAFRAME
df = pd.DataFrame(data.values())
if not df.empty:
    df = df.sort_values('Epoch')
    print("Extracted Data:")
    print(df.head())
    
    # Save to CSV
    df.to_csv('training_metrics.csv', index=False)
    print("\nData saved to 'training_metrics.csv'")

    # 3. PLOT
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Loss (Left Y-Axis)
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(df['Epoch'], df['Loss'], color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot mAP (Right Y-Axis)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('mAP Score', color=color)
    ax2.plot(df['Epoch'], df['Train_mAP'], color=color, marker='s', linestyle='--', label='Train mAP')
    ax2.plot(df['Epoch'], df['Val_mAP'], color='tab:green', marker='^', linestyle='--', label='Val mAP')
    ax2.tick_params(axis='y', labelcolor=color)

    # Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.title('Training Progress')
    plt.tight_layout()
    file_name = log_file_path + '_plot.png'
    plt.savefig(file_name)
    plt.show()
    print(f"Plot saved to {file_name}")
else:
    print("No data found. Please check the log file format.")
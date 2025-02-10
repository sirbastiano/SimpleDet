import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import os

# Import the model definition from resnet18_allB.py (assuming it is placed in the same directory or accessible module)
from s2_B_1 import ResNet18Classifier, BANDS, RESULTS_DIR, TRAIN_DIR
# Set Seaborn style
sns.set_style("whitegrid")

# Define a function to set the styling for plots
def set_style(scale_factor=1.5, font_family='STIXGeneral', dpi=500, fig_width=7.5, fig_height=5, fontsize=13):
    """
    Sets the styling parameters for Matplotlib and Seaborn plots, including figure dimensions.

    Parameters:
    scale_factor (float, optional): Factor to scale the font sizes. Default is 1.5.
    font_family (str, optional): The font family to use for the plots. Default is 'STIXGeneral'.
    dpi (int, optional): Dots per inch (DPI) setting for the figure. Default is 500.
    fig_width (float, optional): Width of the figure in inches. Default is 7.5.
    fig_height (float, optional): Height of the figure in inches. Default is 5.
    fontsize (int, optional): Base font size. Default is 13.
    """
    plt.rcParams['font.family'] = font_family
    plt.rcParams['figure.dpi'] = dpi

    font_size = fontsize * scale_factor
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size,
        'figure.figsize': (fig_width * scale_factor, fig_height * scale_factor)
    })

    sns.set_context("paper", rc={
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size
    })

# Apply the style settings
set_style()

# Create the parser
parser = argparse.ArgumentParser(description='Aggregate results from multiple seeds')
parser.add_argument('--batch-size', type=int, required=True,
                    help='Batch size for training')
parser.add_argument('--learning-rate', type=float, required=True,
                    help='Learning rate for the optimizer')
parser.add_argument('--seeds', type=int, nargs='+', required=True,
                    help='Random seeds used for training')
parser.add_argument('--precision', type=str, required=True,
                    help='Precision used for training')

args = parser.parse_args()
BS = args.batch_size
LR = args.learning_rate
seeds = args.seeds
PRECISION = args.precision
sat = "s2"

#band_str = "-".join(map(str, BANDS))
#result_dir_base = os.path.join(RESULTS_DIR, band_str)


# Define a reverse mapping for directory naming purposes
band_mapping_reverse = {1: 2, 2: 3, 3: 4, 4: 8}
# Map the internal BANDS [1, 2, 3, 4] to the original [2, 3, 4, 8] for directory naming
mapped_band_str = "_".join(map(str, [band_mapping_reverse[b] for b in BANDS]))
# Update the result directory base path with the correct band string
result_dir_base = os.path.join(RESULTS_DIR, mapped_band_str)


# Load the category names
image_dir = TRAIN_DIR
categories = [x.name for x in Path(image_dir).glob("*") if (x.is_dir()) and ('pycache' not in x.name)]
categories.sort()  # Ensure the category names are in the correct order

if not categories:
    raise ValueError("No categories found in the specified image directory.")

print(f"Loaded categories: {categories}")

# Check if GPU is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Collect results from all seeds
all_test_results = []
all_preds = []
all_targets = []

# Function to load model
def load_model(seed, BS, LR):
    model = ResNet18Classifier(image_dir=TRAIN_DIR, batch_size=BS, lr=LR)
    model_path = os.path.join(result_dir_base, 'model', f'fp{PRECISION}_{sat}_seed{seed}_classifier_{BS}_{LR}_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

for seed in seeds:
    # Load test results
    file_path = os.path.join(result_dir_base, 'csv', f'fp{PRECISION}_{sat}_seed{seed}_classifier_{BS}_{LR}.csv')
    df = pd.read_csv(file_path)
    all_test_results.append(df.to_dict(orient='records')[0])
    
    # Load predictions and targets
    preds_file_path = os.path.join(result_dir_base, 'pred', f'fp{PRECISION}_{sat}_seed{seed}_classifier_{BS}_{LR}_preds_targets.csv')
    preds_targets_df = pd.read_csv(preds_file_path)
    all_preds.extend(preds_targets_df['preds'].tolist())
    all_targets.extend(preds_targets_df['targets'].tolist())

# Aggregate results (compute both mean and standard deviation)
aggregated_results_mean = {key: np.mean([result[key] for result in all_test_results]) for key in all_test_results[0].keys()}
aggregated_results_std = {key: np.std([result[key] for result in all_test_results]) for key in all_test_results[0].keys()}

# Combine mean and standard deviation into a single DataFrame
aggregated_df = pd.DataFrame([aggregated_results_mean, aggregated_results_std], index=["mean", "std"])

# Compute confusion matrix
conf_matrix = confusion_matrix(all_targets, all_preds)
conf_matrix_df = pd.DataFrame(conf_matrix, index=categories, columns=categories)

# Compute normalized confusion matrix
conf_matrix_normalized = confusion_matrix(all_targets, all_preds, normalize='true')
conf_matrix_normalized_df = pd.DataFrame(conf_matrix_normalized, index=categories, columns=categories)

# Create the necessary directories
os.makedirs(result_dir_base, exist_ok=True)

# Save aggregated results (mean and std)
aggregated_df.to_csv(os.path.join(result_dir_base, 'csv', f'fp{PRECISION}_{sat}_aggregated_classifier_{BS}_{LR}.csv'))

# Save confusion matrix as CSV
conf_matrix_df.to_csv(os.path.join(result_dir_base, 'csv', f'fp{PRECISION}_{sat}_aggregated_confusion_matrix_{BS}_{LR}.csv'))

# Save normalized confusion matrix as CSV
conf_matrix_normalized_df.to_csv(os.path.join(result_dir_base, 'csv', f'fp{PRECISION}_{sat}_aggregated_confusion_matrix_normalized_{BS}_{LR}.csv'))
# Generate band-related title dynamically based on BANDS
band_title = '-'.join([f'{{{band_mapping_reverse[b]}}}$' for b in BANDS])
# Plot and save confusion matrix with the corrected style
# Plot and save confusion matrix with the corrected style
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title(f'({band_title})', loc='left', pad=20, fontdict={'fontsize': 14})  # Ensures title is clearly visible and placed on the left
plt.xlabel('Predicted Label')
plt.ylabel('True Label', rotation=90, labelpad=20, ha='right')  # Adjust y-label rotation and positioning
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)  # Set y-tick labels to be horizontal
plt.tight_layout()
plt.savefig(os.path.join(result_dir_base, 'csv', f'fp{PRECISION}_{sat}_aggregated_confusion_matrix_{BS}_{LR}.png'))
plt.close()

# Plot and save normalized confusion matrix with the corrected style
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title(f'({band_title})', loc='left', pad=20, fontdict={'fontsize': 14})  # Ensures title is clearly visible and placed on the left
plt.xlabel('Predicted Label')
plt.ylabel('True Label', rotation=0, labelpad=20, ha='right')  # Adjust y-label rotation and positioning
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)  # Set y-tick labels to be horizontal
plt.tight_layout()
plt.savefig(os.path.join(result_dir_base, 'csv', f'fp{PRECISION}_{sat}_aggregated_confusion_matrix_normalized_{BS}_{LR}.png'))
plt.close()



print(f'Aggregated results, including mean and std, and confusion matrices saved for batch size {BS}, learning rate {LR}, and precision {PRECISION}.')









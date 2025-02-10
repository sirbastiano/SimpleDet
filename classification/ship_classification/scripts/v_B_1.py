import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchmetrics
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import os
from pathlib import Path
import rasterio as rio
import numpy as np
import timm
import kornia.augmentation as K
import kornia.filters as filters  # Import GaussianBlur from filters
import argparse
from flops_counter import get_model_complexity_info
from visualisation_utils import plot_csv_values, save_metrics_to_csv, plot_all_data

# Define the maximum number of training epochs
sat = "venus"
MAX_EPOCHS = 200
BANDS = [1]

# Directories for train, validation, and test
TRAIN_DIR = '/Data_large/marine/Dom/Projects/E2E_data_analysis/classification/notebooks/Categories_V_2cat_80Tr20TestR/Train'
TEST_DIR = '/Data_large/marine/Dom/Projects/E2E_data_analysis/classification/notebooks/Categories_V_2cat_80Tr20TestR/Test'
# RESULTS_DIR = f'/Data_large/marine/Dom/Projects/E2E_data_analysis/classification/results/{sat}/64x64'
RESULTS_DIR = f'/Data_large/marine/Dom/Projects/E2E_data_analysis/classification/results_roberto_2cat_80Tr20TestR/{sat}/64x64'

os.makedirs(RESULTS_DIR, exist_ok=True)

# Create the parser to allow for command-line arguments to be passed into the script
parser = argparse.ArgumentParser(description='Parser for training with BatchSize, LearningRate, and Seed')
parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate for the optimizer (default: 0.01)')
parser.add_argument('--precision', type=str, default=32, help='Precision for training the model (default: 32 floating point numbers)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for training (default: 42)')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training.")

# Precision setting for matrix multiplication operations in PyTorch
torch.set_float32_matmul_precision('medium')

# Function to calculate and print model size
def print_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    param_size = 2 if next(model.parameters()).dtype == torch.float16 else 4
    size_mb = num_params * param_size / (1024**2)
    print(f"Model size: {size_mb:.2f} MB")

# Custom Dataset class
class StratifiedImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, loader=self.loader, transform=transform)
        self.filepaths = self.samples

    @staticmethod
    def loader(x):
        try:
            with rio.open(x) as src:
                image_data = [src.read(idx) for idx in BANDS]
                image_data = np.array(image_data)
                tensor = torch.from_numpy(image_data.astype(np.float32))
                if tensor.max() > 1.0:
                    tensor /= 65535
                return tensor
        except rio.RasterioIOError as e:
            print(f"RasterioIOError: Could not open {x}: {e}")
            raise e
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            raise e

    def __getitem__(self, index):
        path, target = self.filepaths[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def get_strong_augmentation():
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=30, p=0.5),
        K.RandomPerspective(distortion_scale=0.4, p=0.5),
        K.RandomElasticTransform(kernel_size=(63, 63), sigma=(7.0, 13.0), alpha=(1.0, 2.0), p=0.5),
        same_on_batch=False
    )

# Define the ResNet18 Classifier
class ResNet18Classifier(pl.LightningModule):
    def __init__(self, image_dir=TRAIN_DIR, batch_size=32, lr=1e-3, image_size=(64, 64), train_split=0.9, val_split=0.1):
        super().__init__()
        self.num_workers = 12
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_split = train_split
        self.val_split = val_split
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize num_classes by counting categories in the dataset folder
        categories = [x.name for x in Path(self.image_dir).glob("*") if (x.is_dir()) and ('pycache' not in x.name)]
        self.num_classes = len(categories)

        # Metrics
        self.precision = torchmetrics.Precision(num_classes=self.num_classes, average='macro', task='multiclass')
        self.recall = torchmetrics.Recall(num_classes=self.num_classes, average='macro', task='multiclass')
        self.f1_score = torchmetrics.F1Score(num_classes=self.num_classes, average='macro', task='multiclass')
        self.cohen_kappa = torchmetrics.CohenKappa(num_classes=self.num_classes, task='multiclass')
        self.mcc = torchmetrics.MatthewsCorrCoef(num_classes=self.num_classes, task='multiclass')

        # ResNet18 model from timm
        resnet18_full = timm.create_model('resnet18', pretrained=True, in_chans=len(BANDS), num_classes=self.num_classes)
        self.neck = nn.Sequential(
            resnet18_full.conv1,
            resnet18_full.bn1,
            resnet18_full.act1,
            resnet18_full.maxpool,
            resnet18_full.layer1,
            resnet18_full.layer2,
            resnet18_full.layer3,
            resnet18_full.layer4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),            
            nn.Dropout(p=0.2),  # Dropout for regularization

            nn.Linear(512, self.num_classes)
        )

        # Kornia for strong augmentation (GPU-accelerated)
        self.strong_augmentations = get_strong_augmentation()

        # Transform (resize)
        self.transform = transforms.Compose([
            transforms.Resize(image_size, antialias=True),
        ])

        # Initialize to store predictions and targets for test set
        self.all_preds = []
        self.all_targets = []

    def setup(self, stage=None):
        full_dataset = StratifiedImageDataset(root=self.image_dir, transform=self.transform)
        train_indices, val_indices, _ = [], [], []
        classes = full_dataset.classes
        class_to_idx = full_dataset.class_to_idx

        for class_name in classes:
            class_indices = [i for i, (_, class_id) in enumerate(full_dataset.samples) if class_id == class_to_idx[class_name]]
            train_size = int(self.train_split * len(class_indices))
            val_size = int(self.val_split * len(class_indices))
            class_train_indices, class_val_indices = train_test_split(class_indices, train_size=train_size, stratify=None)
            train_indices.extend(class_train_indices)
            val_indices.extend(class_val_indices)

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        test_dataset = StratifiedImageDataset(root=TEST_DIR, transform=self.transform)
        return DataLoader(test_dataset, batch_size=1, num_workers=self.num_workers)

    def configure_optimizers(self):
        # Using AdamW optimizer with weight decay of 0.01
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
        
        # Monitor validation loss or MCC instead of Cohen's Kappa
        return {
            'optimizer': optimizer,
            #'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.strong_augmentations(x)
        features = self.neck(x)
        y_hat = self.head(features)
        loss = self.loss_fn(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        metrics = {'train_loss': loss, 'train_acc': acc, 'lr': self.lr}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.neck(x)
        y_hat = self.head(features)
        loss = self.loss_fn(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.precision.update(preds, y)
        self.recall.update(preds, y)
        self.f1_score.update(preds, y)
        self.cohen_kappa.update(preds, y)
        self.mcc.update(preds, y)
        metrics = {
            'val_loss': loss,
            'val_acc': acc,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'cohen_kappa': self.cohen_kappa,
            'mcc': self.mcc
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def test_step(self, batch, batch_idx):
        x, y = batch
        features = self.neck(x)
        y_hat = self.head(features)
        loss = self.loss_fn(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.precision.update(preds, y)
        self.recall.update(preds, y)
        self.f1_score.update(preds, y)
        self.cohen_kappa.update(preds, y)
        self.mcc.update(preds, y)
        metrics = {
            'test_loss': loss,
            'test_acc': acc,
            'test_precision': self.precision,
            'test_recall': self.recall,
            'test_f1_score': self.f1_score,
            'test_cohen_kappa': self.cohen_kappa,
            'test_mcc': self.mcc
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Collect predictions and targets
        self.all_preds.extend(preds.cpu().numpy())  # Convert to numpy for storage
        self.all_targets.extend(y.cpu().numpy())  # Convert to numpy for storage

        return metrics

    def forward(self, x):
        features = self.neck(x)
        return self.head(features)

# Add your training loop below
if __name__ == '__main__':
    args = parser.parse_args()

    BS = args.batch_size
    LR = args.learning_rate
    Pr = args.precision
    seed = args.seed

    pl.seed_everything(seed, workers=True)

    # Ensure reproducibility in PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define a reverse mapping for logging purposes
    band_mapping_reverse = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}
    # Map the internal BANDS [1, 2, 3, 4] to the original [2, 3, 4, 8] for naming
    mapped_band_str = "_".join(map(str, [band_mapping_reverse[b] for b in BANDS]))    
    csv_logger = CSVLogger(save_dir=RESULTS_DIR, name=f'lightning_logs/{mapped_band_str}/fp{Pr}_bands_{mapped_band_str}_seed_{seed}')

    resnet_classifier = ResNet18Classifier(image_dir=TRAIN_DIR, batch_size=BS, image_size=(64, 64), train_split=0.9, val_split=0.1, lr=LR)

    # No early stopping, just the logging and monitoring
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=10,
        logger=csv_logger,
        precision=Pr
    )

    trainer.fit(resnet_classifier)
    test_results = trainer.test(resnet_classifier)

    # Save test results and model
    result_dir = os.path.join(RESULTS_DIR, mapped_band_str, 'csv')
    result_prefix = f'fp{Pr}_{sat}_seed{seed}_classifier_{BS}_{LR}'
    os.makedirs(result_dir, exist_ok=True)

    save_metrics_to_csv(test_results, f'{result_dir}/{result_prefix}.csv')

    preds_dir = os.path.join(RESULTS_DIR, mapped_band_str, 'pred')
    os.makedirs(preds_dir, exist_ok=True)
    preds_targets_df = pd.DataFrame({'preds': resnet_classifier.all_preds, 'targets': resnet_classifier.all_targets})
    preds_targets_df.to_csv(f'{preds_dir}/{result_prefix}_preds_targets.csv', index=False)

    model_dir = os.path.join(RESULTS_DIR, mapped_band_str, 'model')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(resnet_classifier.state_dict(), f'{model_dir}/{result_prefix}_model_statedict.pth')
    torch.save(resnet_classifier, f'{model_dir}/{result_prefix}_model.pth')


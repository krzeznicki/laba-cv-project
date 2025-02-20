import sys
from pathlib import Path
import importlib
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
import argparse
from typing import Tuple, List, Optional
from DefDETR_model import DeformableDETRConfig, DeformableDETR

# --------------------------------------DEFINE PATHS--------------------------------------
# Get the absolute path of the FOOTBALLAI_ENDPROJECT root directory
ROOT_DIR = Path(__file__).resolve().parent.parent  # FOOTBALLAI_ENDPROJECT
sys.path.append(str(ROOT_DIR))

# Reload config and dataset modules to load the latest version
import config.paths
importlib.reload(config.paths)
from config.paths import paths

import utils.load_dataset
importlib.reload(utils.load_dataset)
from utils.load_dataset import LoadDataset

# Paths to directories containing train images and corresponding JSON file 
TRAIN_DIR = paths['HOME'] / 'data' / 'images' / 'train'
COCO_TRAIN_PATH = paths['HOME'] / 'data' / 'coco_annotations' / 'train.json'

# Paths to directories containing valid images and corresponding JSON file 
VALID_DIR = paths['HOME'] / 'data' / 'images' / 'valid'
COCO_VALID_PATH = paths['HOME'] / 'data' / 'coco_annotations' / 'valid.json'

# Paths to directories containing test images and corresponding JSON file 
TEST_DIR = paths['HOME'] / 'data' / 'images' / 'test'
COCO_TEST_PATH = paths['HOME'] / 'data' / 'coco_annotations' / 'test.json'

# Paths to directories containing model checkpoints
CHECKPOINTS_DIR = paths['HOME'] / 'Models' / 'player_detection_model_checkpoints'
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------DEFINE PATHS--------------------------------------

# --------------------------------------CUSTOM COLLATE FUNCTION--------------------------------------
def collate_fn(batch: List[Tuple[np.ndarray, dict]]) -> Tuple[List[np.ndarray], List[dict]]:
    '''
    Custom collate function to format batch and targets fot the DataLoader.
    Converts images to np.float32 and returns them along with their targets.

    Args:
        batch (List[Tuple[np.ndarray, dict]]): List of tuples (image, target) from the dataset.

    Returns:
        Tuple[List[np.ndarray], List[dict]]: List of images and corresponding target annotations.
    '''

    images, targets = zip(*batch)
    images = [img.astype(np.float32) for img in images]
    return images, list(targets)
# --------------------------------------CUSTOM COLLATE FUNCTION--------------------------------------

# --------------------------------------LOAD DATASET LOADERS FOR TRAIN, VALID AND TEST IMAGES--------------------------------------
def dataset_loaders(set_ratio: Optional[float], batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    Loads train, validation and test datasets and initializes DataLoaders for each.

    Args:
        set_ratio (Optional[float]): Ratio for selecting a subset of the dataset (0.0 - 1.0].
        batch_size (int): Batch size for the DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for train, validation and test datasets.
    '''

    # Load train dataset and initialize DataLoader
    train_dataset = LoadDataset(TRAIN_DIR, COCO_TRAIN_PATH, set_ratio=set_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Load valid dataset and initialize DataLoader
    valid_dataset = LoadDataset(VALID_DIR, COCO_VALID_PATH, set_ratio=set_ratio)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Load test dataset and initialize DataLoader
    test_dataset = LoadDataset(TEST_DIR, COCO_TEST_PATH, set_ratio=set_ratio)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader, test_dataloader
# --------------------------------------LOAD DATASET LOADERS FOR TRAIN, VALID AND TEST IMAGES--------------------------------------


if __name__ == '__main__':
    # Parse command line arguments
    # Config arguments:
        # --num_epochs
        # --batch_size
        # --lr
        # --val_frequency
        # --checkpoints
        # --max_checkpoints
        # --use_scheduler
        # --T_max
        # --eta_min
        
    # dataset loaders arguments:
        # --set_ratio
        # --batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')
    parser.add_argument('--set_ratio', type=float, default=None, help='Ratio of the datasets (0.0 - 1.0] or an integer for number of images.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--val_frequency', type=float, default=50, help='Frequency of model validation.')
    parser.add_argument('--checkpoints', type=bool, default=False, help='If True then the model saves checkpoints.')
    parser.add_argument('--max_checkpoints', type=int, default=10, help='Maximum number of checkpoints contained in a folder.')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='If True the use learning rate scheduler (CosineAnnealingLR).')
    parser.add_argument('--T_max', type=int, default=5, help='Number of epochs to reach minimum learning rate -> if use_scheduler=True.')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate after T_max epochs -> if use_scheduler=True.')
    args = parser.parse_args()

    train_dataloader, valid_datalaoder, test_dataloader = dataset_loaders(args.set_ratio, args.batch_size)

    config = DeformableDETRConfig(
        pretrained_model = "SenseTime/deformable-detr",
        num_epochs = args.num_epochs,
        batch_size = args.batch_size,
        optimizer = AdamW,
        learning_rate = args.lr,
        checkpoints_dir = CHECKPOINTS_DIR if args.checkpoints else None,
        max_checkpoints = args.max_checkpoints,
        val_frequency = int(args.val_frequency),
        use_scheduler=args.use_scheduler,
        T_max=args.T_max,
        eta_min=args.eta_min
    )

    model = DeformableDETR(config)
    model.train_model(train_dataloader, valid_datalaoder)
    
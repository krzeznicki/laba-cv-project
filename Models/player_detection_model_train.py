import os
import sys
from pathlib import Path
import importlib

# Get the absolute path of the FOOTBALLAI_ENDPROJECT root directory
ROOT_DIR = Path(__file__).resolve().parent.parent  # FOOTBALLAI_ENDPROJECT
sys.path.append(str(ROOT_DIR))

# Reload config and dataset modules to load the latest version
import config.paths
from torch.utils.data import DataLoader
importlib.reload(config.paths)
from config.paths import paths

import utils.load_dataset
importlib.reload(utils.load_dataset)
from utils.load_dataset import LoadDataset

import numpy as np
import torch
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from tqdm import tqdm
from torch.optim import Adam
import argparse
import wandb

# --------------------------------------DEFINE PATHS--------------------------------------
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
CHECKPOINT_DIR = paths['HOME'] / 'Models' / 'player_detection_model_checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------DEFINE PATHS--------------------------------------

# --------------------------------------CUSTOM COLLATE FUNCTION--------------------------------------
def collate_fn(batch):
    '''
    Custom collate function to format batch and targets fot the DataLoader.
    Converts images to np.float32 and returns them along with their targets.

    Args:
        batch: List of tuples (image, target) from the dataset.

    Returns:
        images (list): List of images as numpy arrays with dtype float32.
        targets (list): List of corresponding target annotations.
    '''
    images, targets = zip(*batch)
    images = [img.astype(np.float32) for img in images]
    return images, list(targets)
# --------------------------------------CUSTOM COLLATE FUNCTION--------------------------------------

# --------------------------------------SAVE CHECKPOINTS--------------------------------------
def save_checkpoints(model, optimizer, epoch, iteration):
    '''
    Save dictionaries of the model and optimizer state at a specific epoch and iteration.

    Args:
        model (torch.nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epoch (int): Current epoch number.
        iteration (int): Current iteration number.
    '''

    checkpoint_path = f'{CHECKPOINT_DIR}/model_{epoch}_{iteration}.pth'
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, 
        checkpoint_path
    )
    print(f'Checkpoints saved: {checkpoint_path}')
# --------------------------------------SAVE CHECKPOINTS--------------------------------------

if __name__ == '__main__':
    # Parse command line arguments for epochs, batc size and learning rate
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Device to be used for training
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Load validation dataset and initialize DataLoader
    valid_dataset = LoadDataset(VALID_DIR, COCO_VALID_PATH)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Load pre-trained Deformable DETR model and image processor
    processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
    model.to(DEVICE)  # Move the model to the specified device

    # Initialize Adam optimizer with the specified learning rate
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Login and initialize Weights & Biases (wandb) for logging
    wandb.login()
    wandb.init(
        project='player_detection_model',
        name='player_detections_logs',
        config={
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        }
    )

    # Initialize variables for tracking losses per epoch
    train_loss_per_epoch = 0.0
    valid_loss_per_epoch = 0.0

    # Watch the model with wandb to track gradients and parameters
    wandb.watch(model, log='all')

    # Training loop over the specified number epochs
    model.train()
    for epoch in range(args.epochs):
        running_train_loss = 0.0  # Accumulate training loss for the epoch

        # Iterate over the train DataLoader
        for iteration, (images, targets) in enumerate(tqdm(valid_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}'), start=1):
            torch.cuda.empty_cache()  # Clear cached memory on GPU to avoid out-of-memory issues

            # Preprocesses images using processor and move to the specified device
            inputs = processor(images=images, return_tensors='pt')
            inputs = inputs['pixel_values'].to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Zero the gradients before the backward pass
            optimizer.zero_grad()

            # Forward passthrough the model
            outputs = model(pixel_values=inputs, labels=targets)
            loss = outputs.loss  # Get the loss from the model output

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Save checkpoint at the each 10 iterations
            if iteration % 10 == 0:
                save_checkpoints(model, optimizer, epoch, iteration)

            # Accumulate training loss for this epoch
            running_train_loss += loss.item()

            # Calculate and log average training loss per iteration
            avg_running_loss_iteration = running_train_loss / iteration
            wandb.log(
                {
                    'train/train_loss_per_iteration': avg_running_loss_iteration
                }
            )

        # Calculate and log average training loss for the entire epoch
        avg_running_loss_per_epoch = train_loss_per_epoch / len(valid_dataloader)
        wandb.log(
            {
                'epoch': epoch + 1
            }
        )
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
from typing import Optional, List, Tuple
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import wandb
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from collections import deque
import os


@dataclass
class CheckpointManager:
    max_checkpoints: int

    def __post_init__(self):
        self.checkpoint_paths = deque(maxlen=self.max_checkpoints)

    def save_checkpoint(self, checkpoint, checkpoint_path):
        if len(self.checkpoint_paths) == self.max_checkpoints:
            self.delete_last_checkpoint()

        torch.save(checkpoint, checkpoint_path)
        self.add_new_checkpoint(checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')

    def delete_last_checkpoint(self):
        oldest_checkpoint = self.checkpoint_paths.popleft()
        os.remove(oldest_checkpoint)

    def add_new_checkpoint(self, checkpoint_path: Path):
        self.checkpoint_paths.append(checkpoint_path)


@dataclass
class DeformableDETRConfig:
    pretrained_model: str = "SenseTime/deformable-detr"
    num_epochs: int = 5
    batch_size: int = 2
    optimizer: Optimizer = AdamW
    learning_rate: float = 0.0001
    checkpoints_dir: Optional[Path] = None
    max_checkpoints: int = 10
    val_frequency: int = 10
    use_scheduler: bool = False  # If True the use CosineAnnealingLR (the method for learning rate scheduler)
    T_max: int = 5  # Number of epochs to reach minimum learning rate
    eta_min: float = 1e-6  # Minimum learning rate after T_max epochs


class DeformableDETR(nn.Module):
    def __init__(self, config: DeformableDETRConfig):
        super(DeformableDETR, self).__init__()

        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained(self.config.pretrained_model, use_fast=True)
        self.model = DeformableDetrForObjectDetection.from_pretrained(self.config.pretrained_model)
        self.model.to(self.device)

        self.optimizer = self.config.optimizer(self.model.parameters(), lr=self.config.learning_rate)
        if self.config.use_scheduler:
            self.scheduler = CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.config.T_max,
                eta_min=self.config.eta_min
            )

        if self.config.checkpoints_dir is not None:
            self.checkpoint_manager = CheckpointManager(self.config.max_checkpoints)
        
        self.map_metric = MeanAveragePrecision(iou_thresholds=[0.3, 0.5, 0.75])
        self.train_loss = []
        self.valid_loss = []


    def forward(self, images: List[np.ndarray], targets: Optional[list[dict]]) -> dict:
        inputs = self.processor(images=images, return_tensors='pt')
        inputs = inputs['pixel_values'].to(self.device)

        if targets is not None:
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            outputs = self.model(pixel_values=inputs, labels=targets)
        else:
            outputs = self.model(pixel_values=inputs)

        return outputs


    def train_model(self, train_dataloader: DataLoader, valid_dataloader: DataLoader):       
        # Initialize Weights & Biases (wandb) for logging
        wandb.init(
            project='player_detection_model',
            name='player_detections_logs',
            config={
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size
            }
        )

        # Watch the model with wandb to track gradients and parameters
        wandb.watch(self.model, log='all')

        # Training loop over the specified number epochs
        for epoch in range(self.config.num_epochs):

            if self.config.use_scheduler:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f'Epoch: {epoch + 1}: Current learning rate: {current_lr}')
                wandb.log(
                    {
                        'learning_rate': current_lr
                    }
                )

            # Pass through one epoch
            train_loss_per_epoch = self._train_one_epoch(train_dataloader, valid_dataloader, epoch)
            wandb.log(
                {
                    'train/train_loss': train_loss_per_epoch
                }
            )

            # Evaluate model after each epoch
            print(f'Epoch: {epoch + 1}. Evaluating model on valid dataset.')
            self.model.eval()
            eval_results = self.evaluate(valid_dataloader, on_dataset='valid')
            self._log_metrics(train_loss_per_epoch, eval_results, epoch=epoch)

            # Tracking train and valid loss
            self.train_loss.append(train_loss_per_epoch)      # Tracking train_loss
            self.valid_loss.append(eval_results['avg_loss'])  # Tracking valid_loss

            # If the path to save checkpoint is defined then save it after each epoch
            if self.config.checkpoints_dir is not None:
                checkpoint_file_name = f'model_epoch{epoch + 1}'
                self._save_model(checkpoint_name=checkpoint_file_name)

            if self.config.use_scheduler:
                self.scheduler.step()

        wandb.finish()
        
        # If the path to save checkpoint is defined then save final model
        if self.config.checkpoints_dir is not None:
            checkpoint_file_name = f'final_player_detection_model'
            self._save_model(checkpoint_name=checkpoint_file_name)


    def _train_one_epoch(self, train_dataloader: DataLoader, valid_dataloader, epoch: int) -> float:
        self.model.train()
        running_train_loss = 0.0

        for iteration, (images, targets) in enumerate(tqdm(train_dataloader, desc=f'Epoch: {epoch + 1}/{self.config.num_epochs}')):
            torch.cuda.empty_cache()  # Clear cached memory on GPU to avoid out-of-memory issues

            # Zero the gradients before the backward pass
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self(images, targets)
            loss = outputs.loss  # Compute loss from the model output
            # Backward pass
            loss.backward()
            self.optimizer.step()
            # Accumulate training loss for this epoch
            running_train_loss += loss.item()
            wandb.log(
                {
                    'iteration': iteration + 1,
                    'train/train_loss_per_iteration': loss.item()
                }
            )

            # After each 10 iterations calculate validation and train loss 
            if (iteration + 1) % self.config.val_frequency == 0:
                train_loss_n_iteration = running_train_loss / (iteration + 1)

                print(f'Iteration: {iteration + 1}. Evaluating model on valid dataset.')
                self.model.eval()
                eval_results = self.evaluate(valid_dataloader, on_dataset='valid')
                self._log_metrics(train_loss_n_iteration, eval_results, iteration=iteration)

                # If the path to save checkpoint is defined then save it after n _iteration
                if self.config.checkpoints_dir is not None:
                    checkpoint_file_name = f'model_epoch{epoch + 1}_iteration{iteration + 1}'
                    self._save_model(checkpoint_name=checkpoint_file_name)
        
        return running_train_loss  / len(train_dataloader)


    def evaluate(self, dataloader: DataLoader, on_dataset: str) -> dict:
        total_loss = 0.0
        self.map_metric.reset()

        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc=f'Evaluating on {on_dataset} dataset.'):
                outputs = self(images, targets)
                total_loss += outputs.loss.item()
                preds, target_boxes = self._process_predictions(outputs, images, targets)
                self.map_metric.update(preds, target_boxes)

        map_results = self.map_metric.compute()
        avg_map_30 = map_results['map'].mean().item()
        avg_loss = total_loss / len(dataloader)

        return {
            'avg_loss': avg_loss,
            'map_30': avg_map_30,
            'map_50': map_results['map_50'].item(),
            'map_75': map_results['map_75'].item(),
        }


    def _log_metrics(self, train_loss, eval_results, iteration=None, epoch=None):
        # If the function was used to evaluate n iteration
        if iteration is not None and epoch is None:
            avg_running_train_loss_n_iteration = train_loss
            wandb.log(
                    {
                        'iteration': iteration + 1,
                        f'train/train_loss_per_{self.config.val_frequency}_iteration': avg_running_train_loss_n_iteration,
                        f'valid/valid_loss_per_{self.config.val_frequency}_iteration': eval_results['avg_loss'],
                        f'mAP/mAP@30_per_{self.config.val_frequency}_iteration': eval_results['map_30'],
                        f'mAP/mAP@50_per_{self.config.val_frequency}_iteration': eval_results['map_50'],
                        f'mAP/mAP@75_per_{self.config.val_frequency}_iteration': eval_results['map_75']
                    }
            )
        
        # If the functions was used to evaluate the epoch
        elif epoch is not None and iteration is None:
            # Train loss 
            avg_train_loss_per_epoch = train_loss
            # Tracking losses
            wandb.log(
                {
                    'epoch': epoch + 1,
                    'train/train_loss_per_epoch': avg_train_loss_per_epoch,
                    'valid/valid_loss_per_epoch': eval_results['avg_loss'],
                    'mAP/mAP@50_per_epoch': eval_results['map_30'],
                    'mAP/mAP@50_per_epoch': eval_results['map_50'],
                    'mAP/mAP@75_per_epoch': eval_results['map_75']
                }
            )


    def _save_model(self, checkpoint_name: str, extension: str='.pth') -> None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = self.config.checkpoints_dir / f'{checkpoint_name}_{timestamp}{extension}'

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        if self.config.use_scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            checkpoint['current_lr'] = self.scheduler.get_last_lr()[0]

        self.checkpoint_manager.save_checkpoint(checkpoint, checkpoint_path)


    def _process_predictions(self, outputs, images, targets) -> Tuple[dict, dict]:
        image_height, image_width = images[0].shape[-2:]

        processed_preds = []
        for i in range(len(outputs.logits)):
            pred_boxes = self.convert_to_xyxy(outputs.pred_boxes[i].detach().cpu().numpy())
            pred_boxes[:, [0, 2]] *= image_width  # Scale x_min, x_max to pixels
            pred_boxes[:, [1, 3]] *= image_height  # Scale y_min,  y_max to pixels

            scores, labels = outputs.logits[i].softmax(-1).detach().cpu().max(dim=-1)
            high_conf_indices = scores > 0.5

            filtered_boxes = pred_boxes[high_conf_indices]
            filtered_scores = scores[high_conf_indices]
            filtered_labels = labels[high_conf_indices]

            processed_preds.append(
                {
                    'boxes': torch.tensor(filtered_boxes),
                    'scores': filtered_scores,
                    'labels': filtered_labels
                }
            )

        processed_targets = []
        for t in targets:
            gt_boxes = self.convert_to_xyxy(t['boxes'].cpu().numpy())
            gt_boxes[:, [0, 2]] *= image_width
            gt_boxes[:, [1, 3]] *= image_height
            processed_targets.append(
                {
                    'boxes': torch.tensor(gt_boxes),
                    'labels': t['class_labels'].cpu()
                }
            )

        return processed_preds, processed_targets
    

    @staticmethod
    def convert_to_xyxy(boxes) -> np.ndarray:
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return boxes

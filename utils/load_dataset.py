import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class LoadDataset(Dataset):
    root_dir: Path
    coco_json_file: Path
    transform: Optional[A.Compose] = None
    
    def __post_init__(self) -> None:
        self.images_file_names = np.array(os.listdir(self.root_dir))

        with open(self.coco_json_file, 'r') as f:
            self.coco_data = json.load(f)

    def __len__(self) -> None:
        return self.images_file_names.shape[0]
    
    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        # Image
        img_filename = self.images_file_names[index]
        img_path = self.root_dir / img_filename  # Absolute path to the image 
        image = cv2.imread(img_path)  # Load an image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Change BGR format to RGB format

        # Annotations - Get annotations to particular image
            # Find image_id for the corresponding image
        image_id = None
        for img_data in self.coco_data['images']:
            if img_data['file_name'] == img_filename:
                image_id = img_data['id']
                break
        
            # Find bboxes and labels for the corresponding image
        bboxes = []
        labels = []
        for annotation in self.coco_data['annotations']:
            if annotation['image_id'] == image_id:
                # Deformable DETR reqiures coordinates in (x_min, y_min, x_max, y_max) format
                x_min = annotation['bbox'][0]
                y_min = annotation['bbox'][1]
                x_max = x_min + annotation['bbox'][2]
                y_max = y_min + annotation['bbox'][3]
                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(annotation['category_id'])

        # Normalization of bounding boxes to 0-1
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes[:, [0, 2]] /= image.shape[1]
        bboxes[:, [1, 3]] /= image.shape[0]
        labels = torch.tensor(labels)

        target = {
            'boxes': bboxes,
            'class_labels': labels
        }

        return image, target
    
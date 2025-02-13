import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class LoadDataset(Dataset):
    '''
    Custom Dataset for loading images and annotations from COCO JSON file.

    Attributes:
        root_dir (Path): Path to the directory containing images.
        coco_json_file (Path): Path to the COCO format JSON file with annotations.
        transform (Optional[A.Compose]): Optional image transformations using Albumentations. Defaults to None
        set_ratio (Optional[Union[int, float]]): Ratio for splitting the dataset. Can be a float (percentage of the dataset). Defaults to 1.0 -> 100%.
    '''

    root_dir: Path
    coco_json_file: Path
    transform: Optional[A.Compose] = None
    set_ratio: Optional[Union[float]] = None
    
    def __post_init__(self) -> None:
        '''
        Initializes the dataset by loading image file names and the COCO JSON file.
        Validates and applies set_ratio to limit the number of samples if specified.

        Raises:
            ValueError: If set_ratio is out of the valid range.
            TypeError: If set_ratio is not an integer or float.
        '''
        # Load file names of images
        self.image_file_names = np.array(os.listdir(self.root_dir))
        # Shuffle the list of file names
        np.random.seed(42)
        np.random.shuffle(self.image_file_names)

        # Check the assumptions
        if self.set_ratio is not None:
            # If type of the set_ratio is float and the ratio is out of the valid range (0.0 - 1.0] then raise ValueError
            if isinstance(self.set_ratio, float):
                if not (0.0 < self.set_ratio <= 1.0):
                    raise ValueError(f'set_ratio as float must be in the range (0.0 - 1.0]')
                num_images = int(len(self.image_file_names) * self.set_ratio)
            
            # If type of the set_ratio is not int or float then raise TypeError
            else:
                raise TypeError('set_ratio must be an float.')
            
            # Select the specified number of images.
            self.image_file_names = self.image_file_names[:num_images]

        print(f'Loaded {len(self.image_file_names)} images from {self.root_dir}')

        # Load COCO ANNOTATIONS from JSON file
        with open(self.coco_json_file, 'r') as f:
            self.coco_data = json.load(f)
            # If set ratio is given then select appropriate image_file_names and annotations to these images
            if self.set_ratio is not None:
                selected_image_file_names = [img_data 
                                             for img_data in self.coco_data['images'] 
                                             if img_data['file_name'] in self.image_file_names]
                
                selected_image_ids = set(img_data['id'] 
                                         for img_data in selected_image_file_names)

                selected_image_annotations = [ann 
                                              for ann in self.coco_data['annotations'] 
                                              if ann['image_id'] in selected_image_ids]
                
                self.coco_data['images'] = selected_image_file_names
                self.coco_data['annotations'] = selected_image_annotations


                
                
    def __len__(self) -> None:
        '''
        Returns the total number of images after in the datasets after applying set_ratio (if specified)

        Returns:
            int: Number of images in the dataset.
        '''
        return self.image_file_names.shape[0]
    
    def __getitem__(self, index: int) -> dict[np.ndarray, torch.Tensor]:
        '''
        Retrieves an images its corresponding annotations (bounding boxes and labels) for a given index.

        Args:
            Index (int): Indedx of the iamge to retrieve.

        Returns:
            dict: A dictionary containing the image and ts annotations:
                - 'image' (numpy.ndarray): The image in RGB format.
                - 'target' (dict): Dictionary with bounding boxes and class labels:
                    - "boxes" (torch.Tensor): Normalized bounding boxes in (x_min, y_min, x_max, y_max) format.
                    - "class_labels" (torch.Tensor): Class labels for each bounding box.
        '''
        # Load image
        img_filename = self.image_file_names[index]
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
                width = annotation['bbox'][2]
                height = annotation['bbox'][3]
                bboxes.append([x_min, y_min, width, height])
                labels.append(annotation['category_id'])

        # Normalize bounding boxes to the range [0 - 1]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes[:, [0, 2]] /= image.shape[1]
        bboxes[:, [1, 3]] /= image.shape[0]
        labels = torch.tensor(labels)

        # Prepare the target dictionary
        target = {
            'boxes': bboxes,
            'class_labels': labels
        }

        return image, target
    
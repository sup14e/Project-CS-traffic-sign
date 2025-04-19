import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ObjectDataset(Dataset):
    def __init__(self, label_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Traffic sign classes
        self.class_to_idx = {"stop": 0, "yield": 1, "yieldAhead": 2, "merge": 3, "signalAhead": 4, "pedestrianCrossing": 5, "keepRight": 6, "speedLimit35": 7, "speedLimit25": 8}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Parse the label file
        self.samples = []
        
        # Check if label_file is a string (path to file) or already parsed data
        if isinstance(label_file, str):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:  # Ensure we have all needed parts
                        img_name = parts[0]
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        class_name = parts[5]
                        
                        # Skip if class is not in our mapping
                        if class_name not in self.class_to_idx:
                            continue
                            
                        img_path = os.path.join(img_dir, img_name)
                        # Convert from (x1, y1, x2, y2) to (x, y, width, height)
                        x = x1
                        y = y1
                        w = x2 - x1
                        h = y2 - y1
                        
                        self.samples.append({
                            'img_path': img_path,
                            'bbox': [x, y, w, h],
                            'class': class_name
                        })
        else:
            # Assuming label_file is a dict with filename as key and label info as value
            for img_name, metadata in label_file.items():
                if 'class' in metadata and metadata['class'] in self.class_to_idx:
                    img_path = os.path.join(img_dir, img_name)
                    self.samples.append({
                        'img_path': img_path,
                        'bbox': metadata['bbox'],
                        'class': metadata['class']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        img = Image.open(img_path).convert("RGB")
        
        # Get image dimensions for normalization
        img_width, img_height = img.size
        
        # Get class label
        class_name = sample['class']
        label = self.class_to_idx[class_name]
        
        # Get bounding box and normalize to [0, 1] range
        x, y, w, h = sample['bbox']
        
        # Normalize bbox coordinates
        normalized_bbox = [
            x / img_width,
            y / img_height,
            w / img_width,
            h / img_height
        ]
        bbox = torch.tensor(normalized_bbox, dtype=torch.float32)
        
        # Convert bbox to absolute pixel values for transformations
        bbox_abs = BoundingBoxes(
            [x, y, w, h], 
            format="xywh", 
            canvas_size=(img_height, img_width)  # (height, width) expected
        )

        img = TVImage(img)  # Convert to torchvision tensor image format

        # Apply transformations (image + bbox)
        if self.transform:
            transformed = self.transform(img, bbox_abs)
            img, bbox_abs = transformed
        
        # Remove extra dimension
        bbox_abs = bbox_abs[0]

        # Convert bbox back to normalized format
        _, img_h, img_w = img.shape
        bbox_norm = torch.tensor([
            bbox_abs[0] / img_w, 
            bbox_abs[1] / img_h, 
            bbox_abs[2] / img_w, 
            bbox_abs[3] / img_h
        ], dtype=torch.float32)

        return img, torch.tensor(label, dtype=torch.long), bbox_norm

class YOLODataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        img_size=640,
        augment=True,
        cache_images=False,
        num_classes=36
    ):
        """
        Enhanced dataset loader for YOLO format data with traffic signs.
        
        Args:
            data_dir (str): Root directory containing train/test/valid folders
            split (str): Dataset split to use ('train', 'test', or 'valid')
            img_size (int): Size to resize images to
            augment (bool): Whether to apply data augmentation (only on train)
            cache_images (bool): Whether to cache images in memory
            num_classes (int): Number of classes in the dataset
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"  # Only augment training data
        self.num_classes = num_classes
        self.cache_images = cache_images
        
        # Set paths for images and labels based on split
        self.img_dir = self.data_dir / split / "images"
        self.label_dir = self.data_dir / split / "labels"
        
        if not self.img_dir.exists() or not self.label_dir.exists():
            raise ValueError(f"Dataset directories not found at {self.img_dir} or {self.label_dir}")
        
        # Valid image extensions
        self.img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Get all image files with corresponding label files
        self.img_files = self._get_img_files()
        
        # Initialize cache
        self.img_cache = {}
        self.label_cache = {}
        
        # Create transforms
        self._create_transforms()
        
        print(f"[{split}] Loaded {len(self.img_files)} images with labels")
    
    def _get_img_files(self):
        """Find all image files that have corresponding label files"""
        img_files = []
        
        for img_path in self.img_dir.iterdir():
            if any(img_path.suffix.lower() == ext for ext in self.img_extensions):
                # Check if corresponding label exists
                label_path = self.label_dir / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    img_files.append(img_path.name)
        
        return sorted(img_files)  # Sort for reproducibility
    
    def _create_transforms(self):
        """Create image transformation pipelines"""
        # Base transforms for both train and validation
        base_transform = [
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        # Augmentations for training
        if self.augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ColorJitter(p=0.2),
                    A.ShiftScaleRotate(
                        shift_limit=0.1, 
                        scale_limit=0.1, 
                        rotate_limit=10, 
                        p=0.5
                    ),
                ] + base_transform,
                bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels']
                )
            )
        else:
            self.transform = A.Compose(
                base_transform,
                bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels']
                )
            )
    
    def __len__(self):
        return len(self.img_files)
    
    def _load_image(self, idx):
        """Load and cache image"""
        if idx in self.img_cache:
            return self.img_cache[idx]
            
        img_name = self.img_files[idx]
        img_path = self.img_dir / img_name
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.cache_images:
            self.img_cache[idx] = image
            
        return image
    
    def _load_labels(self, idx):
        """Load and cache labels"""
        if idx in self.label_cache:
            return self.label_cache[idx]
            
        img_name = self.img_files[idx]
        base_name = os.path.splitext(img_name)[0]
        label_path = self.label_dir / f"{base_name}.txt"
        
        boxes = []
        class_labels = []
        
        # Read the label file (YOLO format)
        with open(label_path, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    data = line.strip().split()
                    class_id = int(data[0])
                    # YOLO format: class_id, x_center, y_center, width, height
                    bbox = list(map(float, data[1:5]))
                    
                    class_labels.append(class_id)
                    boxes.append(bbox)
        
        # Default to empty tensors if no annotations
        if not boxes:
            boxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.zeros(0, dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            class_labels = np.array(class_labels, dtype=np.int64)
        
        if self.cache_images:
            self.label_cache[idx] = (boxes, class_labels)
            
        return boxes, class_labels
    
    def __getitem__(self, idx):
        # Load image
        image = self._load_image(idx)
        
        # Load labels
        boxes, class_labels = self._load_labels(idx)
        
        # Store original image dimensions
        original_height, original_width = image.shape[:2]
        
        # Apply transformations with bounding boxes
        if boxes.shape[0] > 0:  # If we have annotations
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )
            
            image = transformed['image']
            transformed_boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            class_labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
        else:  # No annotations
            # Just transform the image
            transformed = self.transform(
                image=image,
                bboxes=[],
                class_labels=[]
            )
            
            image = transformed['image']
            transformed_boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros(0, dtype=torch.long)
        
        # Create dictionary output with everything needed
        return {
            'image': image,
            'boxes': transformed_boxes,
            'labels': class_labels,
            'img_path': str(self.img_dir / self.img_files[idx]),
            'img_size': (original_height, original_width),
            'img_id': idx
        }
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle batches with different numbers of objects"""
        images = torch.stack([item['image'] for item in batch])
        img_paths = [item['img_path'] for item in batch]
        img_sizes = [item['img_size'] for item in batch]
        img_ids = [item['img_id'] for item in batch]
        
        # Since each image may have a different number of objects,
        # we need to create lists of tensors
        boxes = [item['boxes'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Return as dictionary
        return {
            'images': images,
            'boxes': boxes,
            'labels': labels,
            'img_paths': img_paths,
            'img_sizes': img_sizes,
            'img_ids': img_ids
        }
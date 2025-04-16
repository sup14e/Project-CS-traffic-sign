import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage


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
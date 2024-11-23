# src/data/dataset.py

import torch
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Any
import numpy as np
from datasets import Dataset as HFDataset

class DPODataset:
    def __init__(
        self, 
        image_dir: str,
        human_feedback_path: str,
        resolution: int = 1024,
        use_fp16: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.use_fp16 = use_fp16
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Load human preferences
        with open(human_feedback_path, 'r') as f:
            self.preferences = json.load(f)
            
        # Get all image paths and sort them
        image_paths = sorted(list(self.image_dir.glob("*.png")))
        
        # Group images into pairs (even, odd)
        self.image_pairs = []
        self.valid_preferences = []
        
        # Create valid pairs and preferences
        for idx, pref in enumerate(self.preferences):
            even_num = f"{(idx * 2):05d}"
            odd_num = f"{(idx * 2 + 1):05d}"
            
            even_path = self.image_dir / f"{even_num}.png"
            odd_path = self.image_dir / f"{odd_num}.png"
            
            if even_path.exists() and odd_path.exists():
                self.image_pairs.append((even_path, odd_path))
                self.valid_preferences.append(pref)
        
        print(f"Found {len(self.image_pairs)} valid image pairs out of {len(self.preferences)} preferences")
    
    def __len__(self) -> int:
        return len(self.valid_preferences)
        
    def create_hf_dataset(self):
        dataset_dict = {
            "preferred_image": [],
            "rejected_image": [],
        }
        
        for idx in range(len(self)):
            pref = self.valid_preferences[idx]
            even_path, odd_path = self.image_pairs[idx]
            
            try:
                # Load images
                even_img = Image.open(even_path).convert('RGB')
                odd_img = Image.open(odd_path).convert('RGB')
                
                # If preference is [1, 0], odd is preferred
                if pref == [1, 0]:
                    preferred_img = even_img
                    rejected_img = odd_img
                else:  # [0, 1] case
                    preferred_img = odd_img
                    rejected_img = even_img
                
                # Apply transforms
                preferred_tensor = self.transform(preferred_img)
                rejected_tensor = self.transform(rejected_img)
                
                # Convert to numpy before adding to dataset
                dataset_dict["preferred_image"].append(preferred_tensor.numpy())
                dataset_dict["rejected_image"].append(rejected_tensor.numpy())
                
            except Exception as e:
                print(f"Error loading images at index {idx}: {e}")
                continue
        
        # Convert lists to numpy arrays
        dataset_dict["preferred_image"] = np.stack(dataset_dict["preferred_image"])
        dataset_dict["rejected_image"] = np.stack(dataset_dict["rejected_image"])
            
        dataset = HFDataset.from_dict(dataset_dict)
        
        # Add tensor conversion to features
        dataset.set_format(type="torch", columns=["preferred_image", "rejected_image"])
        
        return dataset

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with memory efficient settings"""

    def collate_fn(batch):
        preferred_images = torch.stack([b["preferred_image"] for b in batch])
        rejected_images = torch.stack([b["rejected_image"] for b in batch])
        return {
            "preferred_image": preferred_images,
            "rejected_image": rejected_images
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,  # Drop incomplete batches
    )


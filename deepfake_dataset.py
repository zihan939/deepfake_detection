import json
import random
from pathlib import Path
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
import io
import numpy as np


class DeepfakeDataset(Dataset):
    
    def __init__(
        self,
        json_path: str,
        split: str = "train",
        num_frames: int = 16,
        image_size: int = 224,
        min_frames: int = 8,
        sampling: str = "uniform",  # "uniform", "random", "consecutive"
    ):
    
        self.num_frames = num_frames
        self.image_size = image_size
        self.sampling = sampling
        self.split = split
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
       
        self.samples = [
            s for s in data[split] 
            if s["num_frames"] >= min_frames
        ]
        
        print(f"[{split}] 加载 {len(self.samples)} 个序列 (过滤掉帧数<{min_frames}的)")
        
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
       
    
    def __len__(self):
        return len(self.samples)

    
    def _sample_augmentation_params(self): 
        params = {
            'hflip': random.random() < 0.5,
            
            'crop_scale': random.uniform(0.8, 1.0),
            'crop_ratio_offset': (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)),
            
            # ColorJitter 
            'brightness': random.uniform(0.9, 1.1),
            'contrast': random.uniform(0.8, 1.2),
            'saturation': random.uniform(0.8, 1.2),
            'hue': random.uniform(-0.05, 0.05),
            
            # JPEG compress
            'jpeg_quality': random.randint(70, 100),
            'apply_jpeg': random.random() < 0.3,  # 30% 概率应用
            
            'blur_radius': random.uniform(0, 0.7),
            'apply_blur': random.random() < 0.2,  # 20% 概率应用

            'grayscale': random.random() < 0.1,
            
            'apply_noise': random.random() < 0.2,
            'noise_std': random.uniform(5, 20),
        }
        return params

    def _apply_consistent_augmentation(self, img, params):
        

        w, h = img.size
        new_size = int(min(w, h) * params['crop_scale'])
        left = int((w - new_size) / 2 + params['crop_ratio_offset'][0] * (w - new_size))
        top = int((h - new_size) / 2 + params['crop_ratio_offset'][1] * (h - new_size))
        left = max(0, min(left, w - new_size))
        top = max(0, min(top, h - new_size))
        img = TF.crop(img, top, left, new_size, new_size)
        

        img = TF.resize(img, (self.image_size, self.image_size))
        

        if params['hflip']:
            img = TF.hflip(img)
        

        img = TF.adjust_brightness(img, params['brightness'])
        img = TF.adjust_contrast(img, params['contrast'])
        img = TF.adjust_saturation(img, params['saturation'])
        img = TF.adjust_hue(img, params['hue'])
        

        if params['apply_jpeg']:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=params['jpeg_quality'])
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')
        

        if params['apply_blur']:
            img = img.filter(ImageFilter.GaussianBlur(radius=params['blur_radius']))

        if params.get('apply_noise', False):
            img = self._add_gaussian_noise(img, params.get('noise_std', 10))

        if params.get('grayscale', False):
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        
        return img

    def _add_gaussian_noise(self, img, std=10):

        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _sample_frames(self, frames: list) -> list:

        n = len(frames)
        T = self.num_frames
        
        if n == T:
            return frames
        
        if n < T:

            indices = list(range(n))
            while len(indices) < T:
                indices.extend(list(range(n)))
            indices = sorted(indices[:T])
            return [frames[i] for i in indices]
        

        if self.sampling == "uniform":
            indices = [int(i * n / T) for i in range(T)]
        elif self.sampling == "random":
            indices = sorted(random.sample(range(n), T))
        elif self.sampling == "consecutive":
            start = random.randint(0, n - T)
            if self.split != "train":
                start = 0
            indices = list(range(start, start + T))
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling}")
        
        return [frames[i] for i in indices]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        frame_paths = self._sample_frames(sample["frames"])
        
        if self.split == "train":
            aug_params = self._sample_augmentation_params()
        else:
            aug_params = None
        
        images = []
        for fp in frame_paths:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception as e:
                return self.__getitem__((idx + 1) % len(self.samples))
            
            if self.split == "train":
                img = self._apply_consistent_augmentation(img, aug_params)
            else:
                img = TF.resize(img, (self.image_size, self.image_size))
            
            img = TF.to_tensor(img)
            img = self.normalize(img)
            images.append(img)
        
        video = torch.stack(images, dim=0)
        
        label = sample["label"]
        
        return video, label
    
    def get_sample_info(self, idx):
        return self.samples[idx]


def create_dataloaders(
    json_path: str,
    batch_size: int = 8,
    num_frames: int = 16,
    image_size: int = 224,
    num_workers: int = 4,
    min_frames: int = 8,
    sampling: str = "uniform",
):

    train_dataset = DeepfakeDataset(
        json_path=json_path,
        split="train",
        num_frames=num_frames,
        image_size=image_size,
        min_frames=min_frames,
        sampling=sampling,
    )
    
    test_dataset = DeepfakeDataset(
        json_path=json_path,
        split="test",
        num_frames=num_frames,
        image_size=image_size,
        min_frames=min_frames,
        sampling="consecutive",  
    )
    val_dataset = DeepfakeDataset(
        json_path=json_path,
        split="val",
        num_frames=num_frames,
        image_size=image_size,
        min_frames=min_frames,
        sampling="consecutive",  
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    
    return train_loader, test_loader, val_loader


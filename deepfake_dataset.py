import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DeepfakeDataset(Dataset):
    """
    Deepfake 视频序列数据集
    每个样本是一个 face sequence，采样 T 帧
    """
    
    def __init__(
        self,
        json_path: str,
        split: str = "train",
        num_frames: int = 16,
        image_size: int = 224,
        min_frames: int = 8,
        sampling: str = "uniform",  # "uniform", "random", "consecutive"
    ):
        """
        Args:
            json_path: dataset.json 路径
            split: "train" 或 "test"
            num_frames: 每个序列采样的帧数 T
            image_size: 图片大小
            min_frames: 序列最少需要多少帧才会被使用
            sampling: 采样策略
                - "uniform": 均匀采样
                - "random": 随机采样
                - "consecutive": 连续采样（随机起点）
        """
        self.num_frames = num_frames
        self.image_size = image_size
        self.sampling = sampling
        self.split = split
        
        # 加载数据集索引
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # 过滤掉帧数太少的序列
        self.samples = [
            s for s in data[split] 
            if s["num_frames"] >= min_frames
        ]
        
        print(f"[{split}] 加载 {len(self.samples)} 个序列 (过滤掉帧数<{min_frames}的)")
        
        # 数据增强
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def _sample_frames(self, frames: list) -> list:
        """根据采样策略选择 T 帧"""
        n = len(frames)
        T = self.num_frames
        
        if n == T:
            return frames
        
        if n < T:
            # 帧数不够，重复填充
            indices = list(range(n))
            while len(indices) < T:
                indices.extend(list(range(n)))
            indices = sorted(indices[:T])
            return [frames[i] for i in indices]
        
        # n > T，需要采样
        if self.sampling == "uniform":
            # 均匀采样
            indices = [int(i * n / T) for i in range(T)]
        elif self.sampling == "random":
            # 随机采样（保持顺序）
            indices = sorted(random.sample(range(n), T))
        elif self.sampling == "consecutive":
            # 连续采样，随机起点
            # start = random.randint(0, n - T)
            start = 0
            indices = list(range(start, start + T))
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling}")
        
        return [frames[i] for i in indices]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 采样 T 帧
        frame_paths = self._sample_frames(sample["frames"])
        
        # 加载并处理图片
        images = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            img = self.transform(img)
            images.append(img)
        
        # 堆叠成 (T, C, H, W)
        video = torch.stack(images, dim=0)
        
        label = sample["label"]
        
        return video, label
    
    def get_sample_info(self, idx):
        """获取样本的元信息（用于调试）"""
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
    """
    创建训练和测试的 DataLoader
    """
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
        sampling="uniform",  # 测试时用均匀采样保证一致性
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
    
    return train_loader, test_loader


# 测试代码
if __name__ == "__main__":
    # 测试数据集
    json_path = "./dataset/dataset.json"
    
    print("=" * 50)
    print("测试 Dataset")
    print("=" * 50)
    
    dataset = DeepfakeDataset(
        json_path=json_path,
        split="train",
        num_frames=16,
        image_size=224,
    )
    
    # 取一个样本
    video, label = dataset[0]
    print(f"\n单个样本:")
    print(f"  video shape: {video.shape}")  # 期望 (16, 3, 224, 224)
    print(f"  label: {label}")
    print(f"  样本信息: {dataset.get_sample_info(0)}")
    
    print("\n" + "=" * 50)
    print("测试 DataLoader")
    print("=" * 50)
    
    train_loader, test_loader = create_dataloaders(
        json_path=json_path,
        batch_size=4,
        num_frames=16,
        num_workers=0,  # 测试时用 0
    )
    
    # 取一个 batch
    for videos, labels in train_loader:
        print(f"\n一个 batch:")
        print(f"  videos shape: {videos.shape}")  # 期望 (4, 16, 3, 224, 224)
        print(f"  labels: {labels}")
        break
    
    print("\n数据集准备完成！")
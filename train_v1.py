# import os
# import time
# import argparse
# import json
# from pathlib import Path
# from datetime import datetime

# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
# from torch.cuda.amp import GradScaler, autocast
# from tqdm import tqdm

# from deepfake_dataset import create_dataloaders
# from model import create_model


# class Trainer:
#     """
#     Deepfake 检测模型训练器
#     """
    
#     def __init__(
#         self,
#         model,
#         train_loader,
#         test_loader,
#         device,
#         output_dir: str = "./checkpoints",
#         # 优化器参数
#         lr: float = 1e-4,
#         weight_decay: float = 0.05,
#         # 训练参数
#         epochs: int = 30,
#         warmup_epochs: int = 3,
#         # 其他
#         use_amp: bool = True,  # 混合精度训练
#         grad_clip: float = 1.0,
#         log_interval: int = 10,
#     ):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.device = device
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         self.epochs = epochs
#         self.use_amp = use_amp
#         self.grad_clip = grad_clip
#         self.log_interval = log_interval
        
#         # 损失函数
#         self.criterion = nn.CrossEntropyLoss()
        
#         # 优化器
#         self.optimizer = AdamW(
#             model.parameters(),
#             lr=lr,
#             weight_decay=weight_decay,
#             betas=(0.9, 0.999),
#         )
        
#         # 学习率调度器
#         total_steps = len(train_loader) * epochs
#         warmup_steps = len(train_loader) * warmup_epochs
        
#         self.scheduler = OneCycleLR(
#             self.optimizer,
#             max_lr=lr,
#             total_steps=total_steps,
#             pct_start=warmup_steps / total_steps,
#             anneal_strategy='cos',
#         )
        
#         # 混合精度
#         self.scaler = GradScaler() if use_amp else None
        
#         # 记录
#         self.history = {
#             "train_loss": [],
#             "train_acc": [],
#             "test_loss": [],
#             "test_acc": [],
#             "lr": [],
#         }
#         self.best_acc = 0.0
        
#     def train_one_epoch(self, epoch):
#         """训练一个 epoch"""
#         self.model.train()
        
#         total_loss = 0.0
#         correct = 0
#         total = 0
        
#         pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
#         for batch_idx, (videos, labels) in enumerate(pbar):
#             videos = videos.to(self.device)  # (B, T, C, H, W)
#             labels = labels.to(self.device)  # (B,)
            
#             self.optimizer.zero_grad()
            
#             # 前向传播（混合精度）
#             if self.use_amp:
#                 with autocast():
#                     logits = self.model(videos)
#                     loss = self.criterion(logits, labels)
                
#                 # 反向传播
#                 self.scaler.scale(loss).backward()
                
#                 # 梯度裁剪
#                 if self.grad_clip > 0:
#                     self.scaler.unscale_(self.optimizer)
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
#             else:
#                 logits = self.model(videos)
#                 loss = self.criterion(logits, labels)
#                 loss.backward()
                
#                 if self.grad_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
#                 self.optimizer.step()
            
#             self.scheduler.step()
            
#             # 统计
#             total_loss += loss.item()
#             preds = logits.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
            
#             # 更新进度条
#             if (batch_idx + 1) % self.log_interval == 0:
#                 avg_loss = total_loss / (batch_idx + 1)
#                 acc = 100.0 * correct / total
#                 lr = self.scheduler.get_last_lr()[0]
#                 pbar.set_postfix({
#                     "loss": f"{avg_loss:.4f}",
#                     "acc": f"{acc:.2f}%",
#                     "lr": f"{lr:.2e}",
#                 })
        
#         epoch_loss = total_loss / len(self.train_loader)
#         epoch_acc = 100.0 * correct / total
        
#         return epoch_loss, epoch_acc
    
#     @torch.no_grad()
#     def evaluate(self, epoch):
#         """评估模型"""
#         self.model.eval()
        
#         total_loss = 0.0
#         correct = 0
#         total = 0
        
#         # 用于计算更详细的指标
#         all_preds = []
#         all_labels = []
        
#         pbar = tqdm(self.test_loader, desc=f"Epoch {epoch+1} [Test]")
        
#         for videos, labels in pbar:
#             videos = videos.to(self.device)
#             labels = labels.to(self.device)
            
#             if self.use_amp:
#                 with autocast():
#                     logits = self.model(videos)
#                     loss = self.criterion(logits, labels)
#             else:
#                 logits = self.model(videos)
#                 loss = self.criterion(logits, labels)
            
#             total_loss += loss.item()
#             preds = logits.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
        
#         epoch_loss = total_loss / len(self.test_loader)
#         epoch_acc = 100.0 * correct / total
        
#         # 计算各类别准确率
#         all_preds = torch.tensor(all_preds)
#         all_labels = torch.tensor(all_labels)
        
#         real_mask = all_labels == 0
#         fake_mask = all_labels == 1
        
#         real_acc = 100.0 * (all_preds[real_mask] == 0).sum().item() / real_mask.sum().item() if real_mask.sum() > 0 else 0
#         fake_acc = 100.0 * (all_preds[fake_mask] == 1).sum().item() / fake_mask.sum().item() if fake_mask.sum() > 0 else 0
        
#         print(f"  Real 准确率: {real_acc:.2f}%")
#         print(f"  Fake 准确率: {fake_acc:.2f}%")
        
#         return epoch_loss, epoch_acc
    
#     def save_checkpoint(self, epoch, is_best=False):
#         """保存检查点"""
#         checkpoint = {
#             "epoch": epoch,
#             "model_state_dict": self.model.state_dict(),
#             "optimizer_state_dict": self.optimizer.state_dict(),
#             "scheduler_state_dict": self.scheduler.state_dict(),
#             "best_acc": self.best_acc,
#             "history": self.history,
#         }
        
#         # 保存最新的
#         path = self.output_dir / "latest.pth"
#         torch.save(checkpoint, path)
        
#         # 保存最佳的
#         if is_best:
#             best_path = self.output_dir / "best.pth"
#             torch.save(checkpoint, best_path)
#             print(f"  保存最佳模型: {best_path}")
        
#         # 每10个epoch保存一次
#         if (epoch + 1) % 10 == 0:
#             epoch_path = self.output_dir / f"epoch_{epoch+1}.pth"
#             torch.save(checkpoint, epoch_path)
    
#     def train(self):
#         """完整训练流程"""
#         print("=" * 60)
#         print("开始训练")
#         print("=" * 60)
#         print(f"设备: {self.device}")
#         print(f"训练集大小: {len(self.train_loader.dataset)}")
#         print(f"测试集大小: {len(self.test_loader.dataset)}")
#         print(f"Batch size: {self.train_loader.batch_size}")
#         print(f"Epochs: {self.epochs}")
#         print(f"混合精度: {self.use_amp}")
#         print(f"输出目录: {self.output_dir}")
#         print("=" * 60)
        
#         start_time = time.time()
        
#         for epoch in range(self.epochs):
#             epoch_start = time.time()
            
#             # 训练
#             train_loss, train_acc = self.train_one_epoch(epoch)
            
#             # 评估
#             test_loss, test_acc = self.evaluate(epoch)
            
#             # 记录
#             self.history["train_loss"].append(train_loss)
#             self.history["train_acc"].append(train_acc)
#             self.history["test_loss"].append(test_loss)
#             self.history["test_acc"].append(test_acc)
#             self.history["lr"].append(self.scheduler.get_last_lr()[0])
            
#             # 检查是否最佳
#             is_best = test_acc > self.best_acc
#             if is_best:
#                 self.best_acc = test_acc
            
#             # 保存
#             self.save_checkpoint(epoch, is_best)
            
#             # 打印摘要
#             epoch_time = time.time() - epoch_start
#             print(f"\nEpoch {epoch+1}/{self.epochs} 完成 ({epoch_time:.1f}s)")
#             print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
#             print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
#             print(f"  Best Acc: {self.best_acc:.2f}%")
#             print("-" * 60)
        
#         total_time = time.time() - start_time
#         print(f"\n训练完成！总耗时: {total_time/3600:.2f}小时")
#         print(f"最佳测试准确率: {self.best_acc:.2f}%")
        
#         # 保存训练历史
#         history_path = self.output_dir / "history.json"
#         with open(history_path, "w") as f:
#             json.dump(self.history, f, indent=2)
        
#         return self.history

import os
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from deepfake_dataset import create_dataloaders
from model import create_model


class FocalLoss(nn.Module):
    """
    Focal Loss: 让模型更关注难分类的样本
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    - 当样本被正确分类且置信度高时，(1-p_t)^gamma 很小，loss 贡献小
    - 当样本被错误分类或置信度低时，(1-p_t)^gamma 大，loss 贡献大
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 类别权重，可以是:
                   - None: 不使用类别权重
                   - float: 二分类时正类的权重
                   - list/tensor: 每个类别的权重
            gamma: focusing parameter，越大越关注难分类样本
                   - gamma=0: 等价于交叉熵
                   - gamma=2: 常用值
            reduction: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                # 二分类: alpha 是正类权重，负类权重是 1-alpha
                self.alpha = torch.tensor([1 - alpha, alpha])
            else:
                self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) 类别标签
        """
        # 计算交叉熵（不做 reduction）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算 p_t
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # 取正确类别的概率
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # 应用 focal weight
        focal_loss = focal_weight * ce_loss
        
        # 应用 alpha (类别权重)
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Trainer:
    """
    Deepfake 检测模型训练器
    """
    
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device,
        output_dir: str = "./checkpoints",
        # 优化器参数
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        # 训练参数
        epochs: int = 30,
        warmup_epochs: int = 3,
        # 其他
        use_amp: bool = True,  # 混合精度训练
        grad_clip: float = 1.0,
        log_interval: int = 10,
        # Loss 设置
        loss_type: str = "focal",  # "ce", "focal", "weighted_ce"
        focal_gamma: float = 2.0,  # Focal Loss 的 gamma 参数
        class_weights: str = "balanced",  # "balanced", "none", 或自定义权重
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.epochs = epochs
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        
        # 计算类别权重
        labels = [s["label"] for s in train_loader.dataset.samples]
        num_real = sum(1 for l in labels if l == 0)
        num_fake = sum(1 for l in labels if l == 1)
        total = num_real + num_fake
        
        print(f"\n类别分布: Real={num_real}, Fake={num_fake}")
        
        if class_weights == "balanced":
            weight_real = total / (2 * num_real) if num_real > 0 else 1.0
            weight_fake = total / (2 * num_fake) if num_fake > 0 else 1.0
            weights = [weight_real, weight_fake]
            print(f"类别权重: Real={weight_real:.3f}, Fake={weight_fake:.3f}")
        elif class_weights == "none":
            weights = None
        else:
            weights = class_weights
        
        # 选择损失函数
        print(f"Loss 类型: {loss_type}")
        if loss_type == "focal":
            self.criterion = FocalLoss(alpha=weights, gamma=focal_gamma)
            print(f"Focal Loss gamma={focal_gamma}")
        elif loss_type == "weighted_ce":
            weight_tensor = torch.tensor(weights).to(device) if weights else None
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:  # "ce"
            self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # 学习率调度器
        total_steps = len(train_loader) * epochs
        warmup_steps = len(train_loader) * warmup_epochs
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
        )
        
        # 混合精度
        self.scaler = GradScaler() if use_amp else None
        
        # 记录
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "balanced_acc": [],
            "real_acc": [],
            "fake_acc": [],
            "lr": [],
        }
        self.best_balanced_acc = 0.0
        
    def train_one_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(self.device)  # (B, T, C, H, W)
            labels = labels.to(self.device)  # (B,)
            
            self.optimizer.zero_grad()
            
            # 前向传播（混合精度）
            if self.use_amp:
                with autocast():
                    logits = self.model(videos)
                    loss = self.criterion(logits, labels)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(videos)
                loss = self.criterion(logits, labels)
                loss.backward()
                
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{acc:.2f}%",
                    "lr": f"{lr:.2e}",
                })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def evaluate(self, epoch):
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 用于计算更详细的指标
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.test_loader, desc=f"Epoch {epoch+1} [Test]")
        
        for videos, labels in pbar:
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            if self.use_amp:
                with autocast():
                    logits = self.model(videos)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(videos)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(self.test_loader)
        epoch_acc = 100.0 * correct / total
        
        # 计算各类别准确率
        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)
        
        real_mask = all_labels == 0
        fake_mask = all_labels == 1
        
        real_acc = 100.0 * (all_preds[real_mask] == 0).sum().item() / real_mask.sum().item() if real_mask.sum() > 0 else 0
        fake_acc = 100.0 * (all_preds[fake_mask] == 1).sum().item() / fake_mask.sum().item() if fake_mask.sum() > 0 else 0
        
        # Balanced Accuracy (两类准确率的平均)
        balanced_acc = (real_acc + fake_acc) / 2
        
        print(f"  Real 准确率: {real_acc:.2f}%")
        print(f"  Fake 准确率: {fake_acc:.2f}%")
        print(f"  Balanced Acc: {balanced_acc:.2f}%")
        
        return epoch_loss, epoch_acc, balanced_acc, real_acc, fake_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_balanced_acc": self.best_balanced_acc,
            "history": self.history,
        }
        
        # 保存最新的
        path = self.output_dir / "latest.pth"
        torch.save(checkpoint, path)
        
        # 保存最佳的
        if is_best:
            best_path = self.output_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"  保存最佳模型: {best_path}")
        
        # 每10个epoch保存一次
        if (epoch + 1) % 10 == 0:
            epoch_path = self.output_dir / f"epoch_{epoch+1}.pth"
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """完整训练流程"""
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"测试集大小: {len(self.test_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"混合精度: {self.use_amp}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # 评估
            test_loss, test_acc, balanced_acc, real_acc, fake_acc = self.evaluate(epoch)
            
            # 记录
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            self.history["balanced_acc"].append(balanced_acc)
            self.history["real_acc"].append(real_acc)
            self.history["fake_acc"].append(fake_acc)
            self.history["lr"].append(self.scheduler.get_last_lr()[0])
            
            # 用 balanced_acc 作为最佳模型标准
            is_best = balanced_acc > self.best_balanced_acc
            if is_best:
                self.best_balanced_acc = balanced_acc
            
            # 保存
            self.save_checkpoint(epoch, is_best)
            
            # 打印摘要
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.epochs} 完成 ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"  Balanced Acc: {balanced_acc:.2f}% (Best: {self.best_balanced_acc:.2f}%)")
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {total_time/3600:.2f}小时")
        print(f"最佳 Balanced Accuracy: {self.best_balanced_acc:.2f}%")
        
        # 保存训练历史
        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


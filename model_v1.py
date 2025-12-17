import torch
import torch.nn as nn
from einops import rearrange
import timm


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer: 学习帧之间的时序关系
    输入: (B, T, D) - T帧的特征向量
    输出: (B, D) - 聚合后的时序特征
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 32,
    ):
        super().__init__()
        
        self.dim = dim
        
        # 可学习的时序位置编码
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_frames, dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        # 可学习的 CLS token（用于聚合时序信息）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm 架构
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, D) - T帧的特征
        Returns:
            (B, D) - 聚合后的特征
        """
        B, T, D = x.shape
        
        # 添加时序位置编码
        x = x + self.temporal_pos_embed[:, :T, :]
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # 取 CLS token 作为输出
        return x[:, 0]


class ViTTemporalModel(nn.Module):
    """
    ViT + Temporal Transformer 端到端模型
    
    流程:
    1. ViT 处理每一帧，提取空间特征
    2. Temporal Transformer 建模帧间时序关系
    3. 分类头输出预测
    """
    
    def __init__(
        self,
        # ViT 参数
        vit_model: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_vit: bool = False,
        # Temporal Transformer 参数
        temporal_num_heads: int = 8,
        temporal_num_layers: int = 4,
        temporal_dropout: float = 0.5,
        max_frames: int = 64,
        # 分类参数
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # ViT 主干网络（使用 timm 加载预训练模型）
        self.vit = timm.create_model(
            vit_model,
            pretrained=pretrained,
            num_classes=0,  # 去掉分类头，只取特征
        )
        
        # 获取 ViT 输出维度
        self.feature_dim = self.vit.num_features  # 通常是 768 for base
        
        # 是否冻结 ViT
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("ViT 参数已冻结")
        
        # Temporal Transformer
        self.temporal_transformer = TemporalTransformer(
            dim=self.feature_dim,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout,
            max_frames=max_frames,
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(self.feature_dim // 2, num_classes),
        )
        
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n模型参数统计:")
        print(f"  总参数量: {total_params / 1e6:.2f}M")
        print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
        print(f"  特征维度: {self.feature_dim}")
    
    def extract_frame_features(self, x):
        """
        提取每帧的特征
        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, T, D)
        """
        B, T, C, H, W = x.shape
        
        # 合并 batch 和 time 维度
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        # ViT 提取特征
        features = self.vit(x)  # (B*T, D)
        
        # 恢复维度
        features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        return features
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch of video sequences
        Returns:
            logits: (B, num_classes)
        """
        # 1. ViT 提取每帧特征
        frame_features = self.extract_frame_features(x)  # (B, T, D)
        
        # 2. Temporal Transformer 建模时序
        temporal_feature = self.temporal_transformer(frame_features)  # (B, D)
        
        # 3. 分类
        logits = self.classifier(temporal_feature)  # (B, num_classes)
        
        return logits
    
    def forward_with_features(self, x):
        """
        返回中间特征（用于可视化或分析）
        """
        frame_features = self.extract_frame_features(x)
        temporal_feature = self.temporal_transformer(frame_features)
        logits = self.classifier(temporal_feature)
        
        return {
            "frame_features": frame_features,
            "temporal_feature": temporal_feature,
            "logits": logits,
        }



def create_model(
    model_type: str = "base",
    pretrained: bool = True,
    freeze_vit: bool = False,
    num_classes: int = 2,
    **kwargs
):
    """
    创建模型的工厂函数
    
    Args:
        model_type: "base", "lite", "large"
        pretrained: 是否使用预训练权重
        freeze_vit: 是否冻结 ViT 参数
        num_classes: 分类数
    """
    configs = {
        "lite": {
            "vit_model": "vit_small_patch16_224",
            "temporal_num_heads": 6,
            "temporal_num_layers": 2,
        },
        "base": {
            "vit_model": "vit_base_patch16_224",
            "temporal_num_heads": 8,
            "temporal_num_layers": 4,
        },
        "large": {
            "vit_model": "vit_large_patch16_224",
            "temporal_num_heads": 12,
            "temporal_num_layers": 6,
        },
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(configs.keys())}")
    
    config = configs[model_type]
    config.update(kwargs)
    
    print(f"\n创建模型: {model_type}")
    print(f"  ViT: {config['vit_model']}")
    print(f"  Temporal layers: {config['temporal_num_layers']}")
    
    model = ViTTemporalModel(
        pretrained=pretrained,
        freeze_vit=freeze_vit,
        num_classes=num_classes,
        **config
    )
    
    return model





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# import timm
# import torchvision.transforms as T


# # ============ 数据增强模块 ============
# class DataAugmentation(nn.Module):
#     """
#     在模型中进行数据增强（训练时启用，测试时关闭）
#     """
#     def __init__(self, image_size=224):
#         super().__init__()
#         self.image_size = image_size
        
#         # 训练时的增强
#         self.train_transform = nn.Sequential(
#             T.RandomHorizontalFlip(p=0.5),
#             T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#             T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
#             T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
#         )
    
#     def forward(self, x):
#         """
#         Args:
#             x: (B, T, C, H, W)
#         """
#         if self.training:
#             B, T, C, H, W = x.shape
#             # 对每帧应用相同的增强（保持时序一致性）
#             x = rearrange(x, 'b t c h w -> (b t) c h w')
#             x = self.train_transform(x)
#             x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
#         return x


# # ============ FreqNet 频域特征提取 ============
# class FreqNet(nn.Module):
#     """
#     频域特征提取模块
#     使用 DCT/FFT 提取频率特征，Deepfake 生成的图像在高频部分往往有异常
    
#     参考: Qian et al. "Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues"
#     """
#     def __init__(self, in_channels=3, out_channels=32):
#         super().__init__()
        
#         # 频域分析
#         self.freq_conv = nn.Sequential(
#             nn.Conv2d(in_channels * 2, 64, kernel_size=3, padding=1),  # *2 for real+imag
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
        
#         # 高频增强分支
#         self.high_freq_conv = nn.Sequential(
#             nn.Conv2d(in_channels * 2, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )
        
#         # 融合
#         self.fusion = nn.Sequential(
#             nn.Conv2d(64 + 32, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
        
#     def forward(self, x):
#         """
#         Args:
#             x: (B, C, H, W) - 单帧图像
#         Returns:
#             freq_features: (B, out_channels, H, W)
#         """
#         B, C, H, W = x.shape
        
#         # 1. 计算 2D FFT
#         # 对每个通道分别计算 FFT
#         x_fft = torch.fft.fft2(x, dim=(-2, -1))
#         x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))  # 低频移到中心
        
#         # 分离实部和虚部
#         x_real = x_fft_shift.real
#         x_imag = x_fft_shift.imag
#         x_freq = torch.cat([x_real, x_imag], dim=1)  # (B, 2C, H, W)
        
#         # 2. 提取频域特征
#         freq_feat = self.freq_conv(x_freq)  # (B, 64, H, W)
        
#         # 3. 高频增强 (mask out low frequency)
#         mask = self._create_high_freq_mask(H, W, x.device)
#         x_high_freq = x_freq * mask
#         high_freq_feat = self.high_freq_conv(x_high_freq)  # (B, 32, H, W)
        
#         # 4. 融合
#         combined = torch.cat([freq_feat, high_freq_feat], dim=1)
#         out = self.fusion(combined)  # (B, out_channels, H, W)
        
#         return out
    
#     def _create_high_freq_mask(self, H, W, device):
#         """创建高频掩码，过滤低频成分"""
#         # 创建距离矩阵
#         y = torch.arange(H, device=device).float() - H // 2
#         x = torch.arange(W, device=device).float() - W // 2
#         y, x = torch.meshgrid(y, x, indexing='ij')
#         dist = torch.sqrt(x**2 + y**2)
        
#         # 高通滤波器：距离中心越近（低频），权重越小
#         radius = min(H, W) // 4
#         mask = 1 - torch.exp(-dist**2 / (2 * radius**2))
#         mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
#         return mask


# class FreqNetLite(nn.Module):
    
#     def __init__(self, in_channels=3, out_channels=32):
#         super().__init__()
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
        
#     def forward(self, x):
#         B, C, H, W = x.shape
        
#         # FFT
#         x_fft = torch.fft.fft2(x, dim=(-2, -1))
#         x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
#         # 幅度谱 (magnitude spectrum)
#         magnitude = torch.abs(x_fft_shift)
#         phase = torch.angle(x_fft_shift)
        
#         x_freq = torch.cat([magnitude, phase], dim=1)
        
#         return self.conv(x_freq)


# # ============ Temporal Transformer ============
# class TemporalTransformer(nn.Module):
#     """
#     Temporal Transformer: 学习帧之间的时序关系
#     """
#     def __init__(
#         self,
#         dim: int = 768,
#         num_heads: int = 8,
#         num_layers: int = 4,
#         mlp_ratio: float = 4.0,
#         dropout: float = 0.1,
#         max_frames: int = 64,
#     ):
#         super().__init__()
        
#         self.dim = dim
        
#         # 可学习的时序位置编码
#         self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_frames, dim))
#         nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
#         # 可学习的 CLS token
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
        
#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim,
#             nhead=num_heads,
#             dim_feedforward=int(dim * mlp_ratio),
#             dropout=dropout,
#             activation='gelu',
#             batch_first=True,
#             norm_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         self.norm = nn.LayerNorm(dim)
        
#     def forward(self, x):
#         """
#         Args:
#             x: (B, T, D)
#         Returns:
#             (B, D)
#         """
#         B, T, D = x.shape
        
#         # 添加时序位置编码
#         x = x + self.temporal_pos_embed[:, :T, :]
        
#         # 添加 CLS token
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat([cls_tokens, x], dim=1)
        
#         # Transformer
#         x = self.transformer(x)
#         x = self.norm(x)
        
#         return x[:, 0]  # CLS token


# # ============ 主模型 ============
# class FreqViTTemporalModel(nn.Module):
#     """
#     FreqNet + ViT + Temporal Transformer
    
#     流程:
#     1. 数据增强 (训练时)
#     2. FreqNet 提取频域特征
#     3. 将频域特征与原图融合
#     4. ViT 提取空间特征
#     5. Temporal Transformer 建模时序
#     6. 分类
#     """
    
#     def __init__(
#         self,
#         # ViT 参数
#         vit_model: str = "vit_base_patch16_224",
#         pretrained: bool = True,
#         freeze_vit: bool = False,
#         # FreqNet 参数
#         use_freqnet: bool = True,
#         freq_channels: int = 32,
#         # Temporal Transformer 参数
#         temporal_num_heads: int = 8,
#         temporal_num_layers: int = 4,
#         temporal_dropout: float = 0.2,
#         max_frames: int = 64,
#         # 分类参数
#         num_classes: int = 2,
#         dropout: float = 0.5,
#         # 数据增强
#         use_augmentation: bool = True,
#         image_size: int = 224,
#     ):
#         super().__init__()
        
#         self.use_freqnet = use_freqnet
#         self.use_augmentation = use_augmentation
        
#         # 1. 数据增强模块
#         if use_augmentation:
#             self.augmentation = DataAugmentation(image_size)
        
#         # 2. FreqNet
#         if use_freqnet:
#             self.freqnet = FreqNet(in_channels=3, out_channels=freq_channels)
            
#             # 融合层：将频域特征融合回 RGB
#             self.freq_fusion = nn.Sequential(
#                 nn.Conv2d(3 + freq_channels, 3, kernel_size=1),
#                 nn.BatchNorm2d(3),
#             )
        
#         # 3. ViT
#         self.vit = timm.create_model(
#             vit_model,
#             pretrained=pretrained,
#             num_classes=0,
#         )
#         self.feature_dim = self.vit.num_features
        
#         if freeze_vit:
#             for param in self.vit.parameters():
#                 param.requires_grad = False
#             print("ViT 参数已冻结")
        
#         # 4. Temporal Transformer
#         self.temporal_transformer = TemporalTransformer(
#             dim=self.feature_dim,
#             num_heads=temporal_num_heads,
#             num_layers=temporal_num_layers,
#             dropout=temporal_dropout,
#             max_frames=max_frames,
#         )
        
#         # 5. 分类头
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(self.feature_dim),
#             nn.Dropout(dropout),
#             nn.Linear(self.feature_dim, self.feature_dim // 2),
#             nn.GELU(),
#             nn.Dropout(dropout / 2),
#             nn.Linear(self.feature_dim // 2, num_classes),
#         )
        
#         self._print_model_info()
    
#     def _print_model_info(self):
#         total_params = sum(p.numel() for p in self.parameters())
#         trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         print(f"\n模型参数统计:")
#         print(f"  总参数量: {total_params / 1e6:.2f}M")
#         print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
#         print(f"  特征维度: {self.feature_dim}")
#         print(f"  FreqNet: {'启用' if self.use_freqnet else '禁用'}")
#         print(f"  数据增强: {'启用' if self.use_augmentation else '禁用'}")
    
#     def extract_frame_features(self, x):
#         """
#         提取每帧的特征
#         Args:
#             x: (B, T, C, H, W)
#         Returns:
#             (B, T, D)
#         """
#         B, T, C, H, W = x.shape
        
#         # 合并 batch 和 time 维度
#         x = rearrange(x, 'b t c h w -> (b t) c h w')
        
#         # FreqNet 处理
#         if self.use_freqnet:
#             freq_feat = self.freqnet(x)  # (B*T, freq_channels, H, W)
#             x = torch.cat([x, freq_feat], dim=1)  # (B*T, 3+freq_channels, H, W)
#             x = self.freq_fusion(x)  # (B*T, 3, H, W)
        
#         # ViT 提取特征
#         features = self.vit(x)  # (B*T, D)
        
#         # 恢复维度
#         features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
#         return features
    
#     def forward(self, x):
#         """
#         Args:
#             x: (B, T, C, H, W)
#         Returns:
#             logits: (B, num_classes)
#         """
#         # 1. 数据增强 (仅训练时)
#         if self.use_augmentation and self.training:
#             x = self.augmentation(x)
        
#         # 2. 提取每帧特征 (包含 FreqNet + ViT)
#         frame_features = self.extract_frame_features(x)
        
#         # 3. Temporal Transformer
#         temporal_feature = self.temporal_transformer(frame_features)
        
#         # 4. 分类
#         logits = self.classifier(temporal_feature)
        
#         return logits
    
#     def forward_with_features(self, x):
#         """返回中间特征（用于可视化）"""
#         if self.use_augmentation and self.training:
#             x = self.augmentation(x)
        
#         frame_features = self.extract_frame_features(x)
#         temporal_feature = self.temporal_transformer(frame_features)
#         logits = self.classifier(temporal_feature)
        
#         return {
#             "frame_features": frame_features,
#             "temporal_feature": temporal_feature,
#             "logits": logits,
#         }


# # ============ 工厂函数 ============
# def create_model(
#     model_type: str = "base",
#     pretrained: bool = True,
#     freeze_vit: bool = False,
#     use_freqnet: bool = True,
#     num_classes: int = 2,
#     max_frames: int = 64,
#     **kwargs
# ):
#     """
#     创建模型
    
#     Args:
#         model_type: "lite", "base", "large"
#         pretrained: 使用预训练权重
#         freeze_vit: 冻结 ViT
#         use_freqnet: 使用频域特征
#         num_classes: 分类数
#         max_frames: 最大帧数
#     """
#     configs = {
#         "lite": {
#             "vit_model": "vit_small_patch16_224",
#             "temporal_num_heads": 6,
#             "temporal_num_layers": 2,
#             "freq_channels": 16,
#         },
#         "base": {
#             "vit_model": "vit_base_patch16_224",
#             "temporal_num_heads": 8,
#             "temporal_num_layers": 4,
#             "freq_channels": 32,
#         },
#         "large": {
#             "vit_model": "vit_large_patch16_224",
#             "temporal_num_heads": 12,
#             "temporal_num_layers": 6,
#             "freq_channels": 64,
#         },
#     }
    
#     if model_type not in configs:
#         raise ValueError(f"Unknown model_type: {model_type}")
    
#     config = configs[model_type]
#     config.update(kwargs)
    
#     print(f"\n创建模型: {model_type}")
#     print(f"  ViT: {config['vit_model']}")
#     print(f"  Temporal layers: {config['temporal_num_layers']}")
#     print(f"  FreqNet: {use_freqnet}")
#     print(f"  Max frames: {max_frames}")
    
#     model = FreqViTTemporalModel(
#         pretrained=pretrained,
#         freeze_vit=freeze_vit,
#         use_freqnet=use_freqnet,
#         num_classes=num_classes,
#         max_frames=max_frames,
#         **config
#     )
    
#     return model
import torch
import torch.nn as nn
from einops import rearrange
import timm
import torch.fft  # ← 新增


class TemporalTransformer(nn.Module):
    
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
        

        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_frames, dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
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
            norm_first=True,  
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
     
        B, T, D = x.shape
        
        x = x + self.temporal_pos_embed[:, :T, :]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        return x[:, 0]


class FrequencyAnalysis(nn.Module):

    def __init__(self, in_channels=3, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim)
        )
    
    def forward(self, x):
        
        freq = torch.fft.fft2(x, norm='ortho')
        freq_magnitude = torch.abs(freq)
        freq_shifted = torch.fft.fftshift(freq_magnitude, dim=(-2, -1))
        freq_log = torch.log1p(freq_shifted)
        return self.encoder(freq_log)


class ViTTemporalModel(nn.Module):
    
    def __init__(
        self,
        vit_model: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_vit: bool = False,
        temporal_num_heads: int = 8,
        temporal_num_layers: int = 4,
        temporal_dropout: float = 0.5,
        max_frames: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_freq: bool = True,
        freq_dim: int = 256,
    ):
        super().__init__()
        
        self.use_freq = use_freq
        self.freq_dim = freq_dim
        

        if self.use_freq:
            self.freq_module = FrequencyAnalysis(in_channels=3, out_dim=freq_dim)
        

        self.vit = timm.create_model(
            vit_model,
            pretrained=pretrained,
            num_classes=0,
        )
        self.feature_dim = self.vit.num_features  # 768
        
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("ViT 参数已冻结")
        

        if self.use_freq:
            self.temporal_dim = self.feature_dim + freq_dim  # 768 + 256 = 1024
        else:
            self.temporal_dim = self.feature_dim  # 768
        
        # Temporal Transformer
        self.temporal_transformer = TemporalTransformer(
            dim=self.temporal_dim,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout,
            max_frames=max_frames,
        )
        

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.temporal_dim, self.temporal_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(self.temporal_dim // 2, num_classes),
        )
        
        self._print_model_info()
    
    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n模型参数统计:")
        print(f"  总参数量: {total_params / 1e6:.2f}M")
        print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
        print(f"  ViT 特征维度: {self.feature_dim}")
        print(f"  使用频域分析: {self.use_freq}")
        print(f"  Temporal 输入维度: {self.temporal_dim}")
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        

        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        

        if self.use_freq:
            freq_features = self.freq_module(x_flat)  # (B*T, 256)
        

        vit_features = self.vit(x_flat)  # (B*T, 768)
        

        if self.use_freq:
            frame_features = torch.cat([freq_features, vit_features], dim=-1)  # (B*T, 1024)
        else:
            frame_features = vit_features
        

        frame_features = rearrange(frame_features, '(b t) d -> b t d', b=B, t=T)  # (B, T, 1024)
        
        # 4. Temporal Transformer
        temporal_feature = self.temporal_transformer(frame_features)  # (B, 1024)
        

        logits = self.classifier(temporal_feature)
        
        return logits
    
    def forward_with_features(self, x):

        B, T, C, H, W = x.shape
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        
        if self.use_freq:
            freq_features = self.freq_module(x_flat)
        else:
            freq_features = None
        
        vit_features = self.vit(x_flat)
        
        if self.use_freq:
            frame_features = torch.cat([freq_features, vit_features], dim=-1)
        else:
            frame_features = vit_features
        
        frame_features = rearrange(frame_features, '(b t) d -> b t d', b=B, t=T)
        temporal_feature = self.temporal_transformer(frame_features)
        logits = self.classifier(temporal_feature)
        
        return {
            "freq_features": freq_features,
            "vit_features": vit_features,
            "frame_features": frame_features,
            "temporal_feature": temporal_feature,
            "logits": logits,
        }
def create_model(
    model_type: str = "base",
    pretrained: bool = True,
    freeze_vit: bool = False,
    num_classes: int = 2,
    temporal_dropout: float = 0.1,
    dropout: float = 0.1,
    use_freq: bool = True,  # ← 新增参数
    **kwargs
):
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
    print(f"  使用频域分析: {use_freq}")  
    
    model = ViTTemporalModel(
        pretrained=pretrained,
        freeze_vit=freeze_vit,
        num_classes=num_classes,
        temporal_dropout=temporal_dropout,
        dropout=dropout,
        use_freq=use_freq,  
        **config
    )
    
    return model
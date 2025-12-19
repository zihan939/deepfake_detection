import torch
import torch.nn as nn
from einops import rearrange
import timm


class TemporalTransformer(nn.Module):
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 32,

        use_pos_embed: bool = True,      
        use_cls_token: bool = True,     
    ):
        super().__init__()
        
        self.dim = dim
        self.use_pos_embed = use_pos_embed
        self.use_cls_token = use_cls_token
        

        if use_pos_embed:
            self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_frames, dim))
            nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        

        if use_cls_token:
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
        

        if self.use_pos_embed:
            x = x + self.temporal_pos_embed[:, :T, :]
        

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # 输出
        if self.use_cls_token:
            return x[:, 0]
        else:
            return x.mean(dim=1)


class ViTTemporalModel(nn.Module):
    
    def __init__(
        self,
        vit_model: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_vit: bool = False,
        # Temporal Transformer 参数
        temporal_num_heads: int = 8,
        temporal_num_layers: int = 4,
        temporal_dropout: float = 0.5,
        max_frames: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
        temporal_aggregation: str = "transformer",  # "transformer", "mean", "last"
        use_temporal_pos_embed: bool = True,
        use_cls_token: bool = True,
    ):
        super().__init__()
        
        self.vit = timm.create_model(
            vit_model,
            pretrained=pretrained,
            num_classes=0, 
        )
        

        self.feature_dim = self.vit.num_features 
        
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("ViT 参数已冻结")

        self.temporal_aggregation = temporal_aggregation
        
        # Temporal Transformer
        if temporal_aggregation == "transformer":
            self.temporal_transformer = TemporalTransformer(
                dim=self.feature_dim,
                num_heads=temporal_num_heads,
                num_layers=temporal_num_layers,
                dropout=temporal_dropout,
                max_frames=max_frames,
                use_pos_embed=use_temporal_pos_embed,
                use_cls_token=use_cls_token,
            )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(self.feature_dim // 2, num_classes),
        )
        
        self._print_model_info()
    
    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n模型参数统计:")
        print(f"  总参数量: {total_params / 1e6:.2f}M")
        print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
        print(f"  特征维度: {self.feature_dim}")
    
    def extract_frame_features(self, x):

        B, T, C, H, W = x.shape
        
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        features = self.vit(x)  # (B*T, D)
        
        features = rearrange(features, '(b t) d -> b t d', b=B, t=T)
        
        return features
    
    def forward(self, x):
        frame_features = self.extract_frame_features(x)  # (B, T, D)
        
        if self.temporal_aggregation == "transformer":
            temporal_feature = self.temporal_transformer(frame_features)
        elif self.temporal_aggregation == "mean":
            temporal_feature = frame_features.mean(dim=1)
        elif self.temporal_aggregation == "last":
            temporal_feature = frame_features[:, -1]
        
        logits = self.classifier(temporal_feature)
        return logits
    
    def forward_with_features(self, x):

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
    # Ablation 选项
    temporal_aggregation: str = "transformer",
    use_temporal_pos_embed: bool = True,
    use_cls_token: bool = True,
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
    
    model = ViTTemporalModel(
        pretrained=pretrained,
        freeze_vit=freeze_vit,
        num_classes=num_classes,
        temporal_aggregation=temporal_aggregation,
        use_temporal_pos_embed=use_temporal_pos_embed,
        use_cls_token=use_cls_token,
        **config
    )
    return model


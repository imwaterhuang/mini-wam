import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

## 使用预训练ResNet-18的视觉骨干作为vision encoder, 参与fine-tuning
#加载 ImageNet-1K 预训练的 ResNet-18。
#删除最后的分类层，只保留视觉 backbone（骨干网络）。
#在 forward 中读取 B, T, C, H, W。
#合并 B 和 T，逐帧编码，再恢复时间维：

class VisualEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)  # 合并
        features = self.feature_extractor(x) #shape: (B*T, 512, 1, 1)
        features = features.flatten(1)  # 展平为 (B*T, 512)
        features = features.view(B, T, -1)  # 展平为 (B, T, 512)
        return features
    
class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp1 = nn.Linear(in_features=4, out_features=64, bias=True)
        self.mlp2 = nn.Linear(in_features=64, out_features=64, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.flatten(1)
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        x = self.relu(x)
        return x
    
###HistoryFusion内部顺序：
###1. 将视觉特征从 [B,2,512] 展平为 [B,1024]。
###2. 沿最后一个维度与位置特征拼接，得到 [B,1088]。
###3. 通过两层 MLP（Multilayer Perceptron，多层感知机）：

class HistoryFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Linear(in_features=1088, out_features=512, bias=True)
        self.mlp2 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.relu = nn.ReLU()

    def forward(self, visual_features, position_features):
        # visual_features: (B, 2, 512)
        # position_features: (B, 64)
        B = visual_features.shape[0]
        visual_features = visual_features.flatten(1)  # 展平为 (B, 1024)
        x = torch.cat([visual_features, position_features], dim=-1)  # 拼接为 (B, 1088)
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        x = self.relu(x)
        return x

#[B,256]
#→ Linear(256,256)
#→ ReLU
#→ Linear(256,32)
#→ reshape
#[B,16,2]  
class ActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.mlp2 = nn.Linear(in_features=256, out_features=32, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        x = x.view(x.size(0), 16, 2)  # reshape为 (B, 16, 2)
        return x
    
class ActionOnlyPolicy(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.visual_encoder = VisualEncoder(pretrained=pretrained)
        self.state_encoder = StateEncoder()
        self.history_fusion = HistoryFusion()
        self.action_head = ActionHead()

    def forward(self, observation_history, position_history):
        # observation_history: (B, 2, C, H, W)
        # position_history: (B, 2, 2)
        visual_features = self.visual_encoder(observation_history)  # (B, 2, 512)
        position_features = self.state_encoder(position_history)  # (B, 64)
        fused_features = self.history_fusion(visual_features, position_features)  # (B, 256)
        action_predictions = self.action_head(fused_features)  # (B, 16, 2)
        return action_predictions


def masked_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Average Smooth L1 action loss over valid time steps only."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.ndim != 3:
        raise ValueError(
            f"prediction and target must have shape [B, T, action_dim], "
            f"got {tuple(prediction.shape)}"
        )
    if valid_mask.shape != prediction.shape[:2]:
        raise ValueError(
            f"valid_mask must have shape {tuple(prediction.shape[:2])}, "
            f"got {tuple(valid_mask.shape)}"
        )
    if valid_mask.dtype != torch.bool:
        raise TypeError(f"valid_mask must have dtype bool, got {valid_mask.dtype}")
    if not valid_mask.any():
        raise ValueError("valid_mask must contain at least one valid action step")

    per_coordinate_loss = F.smooth_l1_loss(prediction, target, reduction="none")
    per_step_loss = per_coordinate_loss.mean(dim=-1)
    mask = valid_mask.to(dtype=per_step_loss.dtype)
    return (per_step_loss * mask).sum() / mask.sum()
    
#episode（回合）末尾可能不足16步，其余位置是 padding（填充）。这些位置不能参与训练。

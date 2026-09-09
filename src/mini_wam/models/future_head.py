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

#ActionPrefixEncoder: 将动作前缀编码为特征向量，供FutureHead使用
#concat with HistoryFusion output: (B, 256+256=512)
#input:true action prefix: (B, 4, 2) output: (B, 256)
#shared per step mlp 2->64 total 256
#with time embedding: (B, 4, 64)
def time_embedding(timesteps, embedding_dim=64):
    """Generate sinusoidal position embeddings."""
    half_dim = embedding_dim // 2

    frequencies = torch.exp(
        torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
        * -(torch.log(
            torch.tensor(
                10000.0,
                device=timesteps.device,
                dtype=timesteps.dtype,
            )
        ) / (half_dim - 1))
    )

    angles = timesteps[:, None] * frequencies[None, :]
    return torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=-1,
    )

class ActionPrefixEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Linear(in_features=2, out_features=64, bias=True)
        self.mlp2 = nn.Linear(in_features=64, out_features=64, bias=True)
        self.relu = nn.ReLU()
        self.time_embedding_dim = 64

    def forward(self, x):
        # x: (B, 4, 2)
        B, T, _ = x.shape
        x = x.reshape(B * T, 2)
        x = self.mlp1(x)
        x = self.relu(x)

        x = x + time_embedding(torch.arange(T, device=x.device).repeat(B),
                               embedding_dim=self.time_embedding_dim,
                               )  # Add time embedding

        x = self.mlp2(x)
        x = self.relu(x)
        x = x.view(B, T * 64) # reshape为 (B, 4*64=256)
        return x

class FrozenTargetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # 1. 创建 ResNet-18
        # 2. 去掉分类层
        # 3. 冻结 self.feature_extractor 的全部参数
        # 4. 切换到 eval 模式
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

    def train(self, mode=True):
        # 无论外部传入 True 还是 False，
        # 都必须让本模块及其子模块保持 eval
        super().train(False)
        return self

    def forward(self, x):
        # x: [B, 4, 3, 96, 96]
        # 合并 B 和 T
        # 在 no_grad 环境中提取特征
        # 恢复成 [B, 4, 512]
        # 沿最后一维进行 L2 normalization
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        with torch.no_grad():
            features = self.feature_extractor(x)  # shape: (B*T, 512, 1, 1)
        features = features.flatten(1)  # 展平为 (B*T, 512)
        features = features.view(B, T, -1)  # 恢复为 (B, T, 512)
        features = F.normalize(features, p=2, dim=-1)  # L2 normalization
        return features

class FutureHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Linear(in_features=512, out_features=1024, bias=True)
        self.mlp2 = nn.Linear(in_features=1024, out_features=512*4, bias=True)

        self.relu = nn.ReLU()
        self.L2norm = nn.functional.normalize

    def forward(self, x):
        # x: (B, 512)
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        x = self.relu(x)
        x = x.view(x.size(0), 4, 512)  # reshape为 (B, 4, 512)
        x = self.L2norm(x, p=2, dim=-1) # L2 normalization along the last dimension
        return x

def masked_cosine_loss(
    prediction: torch.Tensor,  # [B, 4, 512]
    target: torch.Tensor,      # [B, 4, 512]
    valid_mask: torch.Tensor,  # [B, 4]
) -> torch.Tensor:
    """Average cosine distance over valid future time steps only."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.ndim != 3:
        raise ValueError(
            f"prediction and target must have shape [B, T, feature_dim], "
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
        raise ValueError("valid_mask must contain at least one valid future step")

    similarity = F.cosine_similarity(
        prediction,
        target,
        dim=-1,
        eps=1e-8,
    )  # [B, 4]

    per_step_loss = 1.0 - similarity  # [B, 4]

    mask = valid_mask.to(dtype=per_step_loss.dtype)

    return (per_step_loss * mask).sum() / mask.sum()

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

class FutureHeadPolicy(nn.Module):
    def __init__(self, pretrained=True, target_pretrained=True):
        super().__init__()
        self.visual_encoder = VisualEncoder(pretrained=pretrained)
        self.state_encoder = StateEncoder()
        self.history_fusion = HistoryFusion()
        self.action_head = ActionHead()

        self.action_prefix_encoder = ActionPrefixEncoder()
        self.frozen_target_encoder = FrozenTargetEncoder(pretrained=target_pretrained)
        self.future_head = FutureHead()


#Train-only future supervision

    def forward(self, observation_history, position_history, true_action_prefix=None, future_images=None, compute_future=False):
        # observation_history: (B, 2, C, H, W)
        # position_history: (B, 2, 2)
        visual_features = self.visual_encoder(observation_history)  # (B, 2, 512)
        position_features = self.state_encoder(position_history)  # (B, 64)
        fused_features = self.history_fusion(visual_features, position_features)  # (B, 256)

        action_prediction = self.action_head(fused_features)  # (B, 16, 2)

        if not compute_future:
            return action_prediction

        if true_action_prefix is None or future_images is None:
            raise ValueError(
                "compute_future=True 时必须提供 true_action_prefix 和 future_images"
            )

        future_images_features = self.frozen_target_encoder(future_images)  # (B, 4, 512)
        action_prefix_features = self.action_prefix_encoder(true_action_prefix)  # (B, 256)
        future_head_input = torch.cat([fused_features, action_prefix_features], dim=-1)  # (B, 512)
        future_prediction = self.future_head(future_head_input)  # (B, 4, 512)

        return action_prediction, future_prediction, future_images_features

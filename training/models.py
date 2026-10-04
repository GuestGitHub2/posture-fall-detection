"""Small trainable models; PyTorch is only required by training commands."""

from __future__ import annotations

import torch
from torch import nn

COCO_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


class PostureMLP(nn.Module):
    """Tiny MLP over a normalized skeleton, including confidence channels."""

    def __init__(self, num_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(51, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        return self.net(skeleton)


class GraphTemporalBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.spatial = nn.Conv2d(channels_in, channels_out, 1)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(channels_out),
            nn.Dropout(0.15),
        )
        self.residual = nn.Conv2d(channels_in, channels_out, 1)

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        # [N,C,T,V] spatial message passing; matmul is portable to ONNX.
        neighbors = torch.matmul(skeleton, self.adjacency)
        return torch.relu(self.temporal(self.spatial(neighbors)) + self.residual(skeleton))


class SkeletonTemporalNet(nn.Module):
    """Lightweight graph + temporal baseline, not a pretrained ST-GCN++ implementation."""

    def __init__(self, num_classes: int = 2, channels: int = 32) -> None:
        super().__init__()
        adjacency = torch.eye(17)
        for left, right in COCO_EDGES:
            adjacency[left, right] = adjacency[right, left] = 1
        adjacency /= adjacency.sum(dim=0, keepdim=True)
        self.blocks = nn.Sequential(
            GraphTemporalBlock(3, channels, adjacency),
            GraphTemporalBlock(channels, channels, adjacency),
            GraphTemporalBlock(channels, channels * 2, adjacency),
        )
        self.head = nn.Linear(channels * 2, num_classes)

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        # Explicit deployment contract: [N,C,T,V,M], one tracked person per inference.
        value = skeleton.squeeze(-1)
        value = self.blocks(value).mean(dim=(2, 3))
        return self.head(value)


def build_model(metadata: dict) -> nn.Module:
    if metadata["kind"] == "posture":
        return PostureMLP(len(metadata["classes"]), metadata.get("hidden", 64))
    if metadata["kind"] == "fall":
        return SkeletonTemporalNet(len(metadata["classes"]), metadata.get("channels", 32))
    raise ValueError(f"Unknown model kind: {metadata['kind']}")

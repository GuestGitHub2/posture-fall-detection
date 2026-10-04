"""Shared supervised trainer and ONNX export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.dataset import augment
from training.models import build_model


class AugmentedDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        seed: int,
        training: bool,
        temporal: bool = False,
    ) -> None:
        self.data, self.labels = data, labels
        self.rng, self.training, self.temporal = np.random.default_rng(seed), training, temporal

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        points = augment(self.data[index], self.rng) if self.training else self.data[index]
        if self.temporal:
            points = points.transpose(2, 0, 1)[..., None]
        return torch.from_numpy(np.ascontiguousarray(points)), torch.tensor(self.labels[index])


def train(
    data: np.ndarray,
    labels: np.ndarray,
    validation_data: np.ndarray,
    validation_labels: np.ndarray,
    metadata: dict,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    export_path: Path | None = None,
) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    np.random.seed(seed)
    temporal = metadata["kind"] == "fall"
    train_loader = DataLoader(
        AugmentedDataset(data, labels, seed, True, temporal), batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        AugmentedDataset(validation_data, validation_labels, seed, False, temporal),
        batch_size=batch_size,
    )
    model = build_model(metadata)
    counts = np.bincount(labels, minlength=len(metadata["classes"]))
    weights = len(labels) / np.maximum(counts, 1) / len(counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss, history = float("inf"), []
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        loss_total = 0.0
        for values, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(values), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach()) * len(targets)
        model.eval()
        validation_loss, correct = 0.0, 0
        with torch.inference_mode():
            for values, targets in validation_loader:
                logits = model(values)
                validation_loss += float(criterion(logits, targets)) * len(targets)
                correct += int((logits.argmax(1) == targets).sum())
        validation_loss /= len(validation_labels)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_total / len(labels),
            "validation_loss": validation_loss,
            "validation_accuracy": correct / len(validation_labels),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({"metadata": metadata, "state_dict": model.state_dict()}, output)
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if export_path:
        checkpoint = torch.load(output, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        example = torch.zeros((1, 3, metadata["samples"], 17, 1) if temporal else (1, 17, 3))
        export_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            example,
            export_path,
            input_names=["skeleton"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={"skeleton": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )
        export_path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"ONNX exported to {export_path}")
    return {"best_validation_loss": best_loss, "checkpoint": str(output)}

"""Evaluation helpers."""

from __future__ import annotations

from typing import Dict

import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def evaluate(model, data_loader, device) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits

        total_loss += outputs.loss.item()
        all_labels.extend(batch["labels"].cpu().numpy().tolist())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    return {
        "loss": total_loss / max(1, len(data_loader)),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
    }

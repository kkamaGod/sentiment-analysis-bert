"""Inference script for sentiment prediction."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_MAP = {0: "negative", 1: "positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment for one sentence")
    parser.add_argument("--model-dir", type=str, default="artifacts/model")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def predict(text: str, model_dir: str, max_length: int = 128):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_idx = int(probs.argmax().item())

    return {
        "label": LABEL_MAP[pred_idx],
        "confidence": float(probs[pred_idx].item()),
    }


def main() -> None:
    args = parse_args()
    result = predict(args.text, args.model_dir, args.max_length)
    print(result)


if __name__ == "__main__":
    main()

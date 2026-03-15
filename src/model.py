"""Model and tokenizer factory utilities."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class ModelConfig:
    model_name: str = "bert-base-chinese"
    num_labels: int = 2


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def build_model(config: ModelConfig):
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
    )

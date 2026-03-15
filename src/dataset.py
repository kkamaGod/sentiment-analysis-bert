"""Dataset helpers for sentiment analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataConfig:
    """Data and dataloader parameters."""

    csv_path: str = "data/train.csv"
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 128
    train_batch_size: int = 8
    eval_batch_size: int = 16
    test_size: float = 0.2
    random_state: int = 42


class SentimentDataset(Dataset):
    """Torch dataset that tokenizes texts on-the-fly."""

    def __init__(self, texts, labels, tokenizer, max_length: int = 128) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def load_dataframe(config: DataConfig) -> pd.DataFrame:
    """Load and validate source CSV."""
    df = pd.read_csv(config.csv_path)
    required = {config.text_column, config.label_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def create_data_loaders(config: DataConfig, tokenizer) -> Tuple[DataLoader, DataLoader]:
    """Split data into train/validation and return dataloaders."""
    df = load_dataframe(config)

    train_df, val_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df[config.label_column],
    )

    train_dataset = SentimentDataset(
        train_df[config.text_column],
        train_df[config.label_column],
        tokenizer,
        config.max_length,
    )
    val_dataset = SentimentDataset(
        val_df[config.text_column],
        val_df[config.label_column],
        tokenizer,
        config.max_length,
    )

    return (
        DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False),
    )

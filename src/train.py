"""Train a BERT sentiment classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.dataset import DataConfig, create_data_loaders
from src.evaluate import evaluate
from src.model import ModelConfig, build_model, build_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sentiment analysis")
    parser.add_argument("--model-name", type=str, default="bert-base-chinese")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="artifacts/model")
    parser.add_argument("--data-path", type=str, default="data/train.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig(model_name=args.model_name)
    tokenizer = build_tokenizer(args.model_name)
    model = build_model(model_config).to(device)

    data_config = DataConfig(
        csv_path=args.data_path,
        max_length=args.max_length,
        train_batch_size=args.batch_size,
    )
    train_loader, val_loader = create_data_loaders(data_config, tokenizer)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            progress.set_postfix(loss=running_loss / max(1, progress.n))

        metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}: "
            f"val_loss={metrics['loss']:.4f}, val_acc={metrics['accuracy']:.4f}, "
            f"val_f1={metrics['f1']:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

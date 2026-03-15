#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python -m src.train --epochs 1 --batch-size 4
python -m src.predict --text "这个产品真不错，我很满意"
uvicorn app:app --host 0.0.0.0 --port 8000

#!/bin/bash
# Перекачивает OHLCV с MOEX ISS и пересчитывает rv_features.csv
# Запускать из корня проекта: bash refresh_moex.sh

set -e
cd "$(dirname "$0")"

echo "=== Скачиваем котировки MOEX ISS ==="
python3 src/data/download_moex.py \
  --till 2026-05-28 \
  --gld_from 2018-01-01 \
  --slv_from 2018-01-01 \
  --plt_from 2023-01-01 \
  --pld_from 2023-01-01

echo ""
echo "=== Пересчитываем реализованную волатильность ==="
python3 src/models/compute_rv.py

echo ""
echo "=== Пересобираем daily_sentiment.csv ==="
python3 src/data/aggregate_sentiment.py

echo ""
echo "=== Запускаем kNN-модель ==="
python3 src/models/knn_model.py

echo ""
echo "=== Исправляем ключевую ставку ==="
python3 fix_key_rate.py

echo ""
echo "=== ГОТОВО. Статистика: ==="
python3 - << 'PYEOF'
import pandas as pd
rv = pd.read_csv("data/processed/rv_features.csv")
sent = pd.read_csv("data/sentiment/daily_sentiment.csv")
print("rv_features.csv строк:", rv.groupby("secid").size().to_dict())
print("daily_sentiment.csv строк:", len(sent))
print("key_rate уникальных:", sent["macro_key_rate"].nunique())
PYEOF

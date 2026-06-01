"""
knn_model.py — kNN-прогнозирование волатильности на основе
               поиска похожих информационных состояний (HAR + sentiment).

Методология: Halousková & Lyócsa (2025).
Для каждого тестового дня t находим k исторически ближайших дней
по взвешенному расстоянию в пространстве (HAR-признаки, sentiment-вектор).
Прогноз — взвешенное среднее log(RV) соседей (веса ∝ сходству).

Метрики берутся из единого модуля src.models.metrics
(КАНОНИЧЕСКАЯ Patton QLIKE — раньше здесь была другая формула,
сравнение с HAR было некорректным).

Sentiment-признаки СДВИГАЮТСЯ НА 1 ДЕНЬ ВПЕРЁД во избежание data leakage:
для прогноза RV_t используется sentiment, известный к концу дня t-1.

Запуск:
    python3 src/models/knn_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import compute_metrics

# ─── Пути ────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).resolve().parents[2]
RV_FILE       = BASE / "data" / "processed" / "rv_features.csv"
SENT_FILE     = BASE / "data" / "sentiment"  / "daily_sentiment.csv"
OUT_DIR       = BASE / "data" / "processed"  / "ml_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Признаки ─────────────────────────────────────────────────────────────────
HAR_COLS  = ["rv_d", "rv_w", "rv_m"]
SENT_COLS = [
    "sentiment_score", "sentiment_gdelt", "tg_score",
    "attention_google", "attention_volume", "sentiment_combined",
]

# ─── Гиперпараметры (подобраны по сетке на обучающей выборке) ────────────────
# Пересмотрены после фикса data leakage и унификации QLIKE;
# на исправленной задаче оптимумы сместились вверх по k.
BEST_K: dict[str, int] = {
    "GLDRUB_TOM": 75,
    "SLVRUB_TOM": 30,
}
SENT_WEIGHT = 1.0   # вес sentiment-блока относительно HAR-блока


# ─── Основная функция ─────────────────────────────────────────────────────────
def run_knn(
    secid: str,
    rv: pd.DataFrame,
    sent: pd.DataFrame,
    train_frac: float = 0.60,
    k: int = 20,
    sent_weight: float = SENT_WEIGHT,
    sent_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Expanding-window kNN прогноз для одного инструмента.

    sent_cols : подмножество SENT_COLS для ablation. По умолчанию все 6.

    Returns
    -------
    pred_df : DataFrame с колонками [date, actual, predicted]
    metrics : dict с MAE, RMSE, R2_OOS, QLIKE
    """
    if sent_cols is None:
        sent_cols = SENT_COLS

    df = (rv[rv["secid"] == secid]
          .sort_values("date")
          .dropna(subset=["rv_target"] + HAR_COLS)
          .reset_index(drop=True))

    # КРИТИЧНО: shift(1) — sentiment_{t-1} для предсказания RV_t,
    # иначе data leakage (новости и RV одного дня).
    sent_lagged = sent.sort_values("date").copy()
    sent_lagged[sent_cols] = sent_lagged[sent_cols].shift(1)

    df = df.merge(sent_lagged[["date"] + sent_cols], on="date", how="left")
    df[sent_cols] = df[sent_cols].fillna(0.0)

    n       = len(df)
    n_train = int(n * train_frac)

    y = df["rv_target"].values

    # Нормализуем HAR и sentiment блоки независимо (z-score по обучающей выборке)
    H = df[HAR_COLS].values.astype(float)
    S = df[sent_cols].values.astype(float)

    h_mean, h_std = H[:n_train].mean(0), H[:n_train].std(0)
    s_mean, s_std = S[:n_train].mean(0), S[:n_train].std(0)
    h_std[h_std == 0] = 1
    s_std[s_std == 0] = 1

    H = (H - h_mean) / h_std
    S = (S - s_mean) / s_std

    # ── Expanding window ─────────────────────────────────────────────────────
    preds = []
    for t in range(n_train, n):
        # RBF-сходство по HAR и по sentiment
        h_sim = 1.0 / (1.0 + np.sqrt(((H[:t] - H[t]) ** 2).sum(axis=1)))
        s_sim = 1.0 / (1.0 + np.sqrt(((S[:t] - S[t]) ** 2).sum(axis=1)))
        sim   = h_sim + sent_weight * s_sim

        top_k = np.argsort(sim)[-k:]
        w     = sim[top_k]
        w    /= w.sum()
        preds.append(float(w @ y[top_k]))

    y_test  = y[n_train:]
    y_hat   = np.array(preds)
    dates   = df["date"].iloc[n_train:].values

    pred_df = pd.DataFrame({
        "date":      pd.to_datetime(dates),
        "actual":    y_test,
        "predicted": y_hat,
    })

    # Унифицированные метрики (Patton QLIKE с exp() от log-RV)
    m = compute_metrics(y_test, y_hat, is_log=True)
    metrics = {
        "secid":  secid,
        "model":  f"kNN(k={k})",
        "MAE":    round(m["MAE"],    6),
        "RMSE":   round(m["RMSE"],   6),
        "R2_OOS": round(m["R2_OOS"], 6),
        "QLIKE":  round(m["QLIKE"],  6),
    }
    return pred_df, metrics


# ─── Точка входа ─────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  kNN Volatility Model (HAR + sentiment)")
    print("=" * 55)

    rv   = pd.read_csv(RV_FILE,   parse_dates=["date"])
    sent = pd.read_csv(SENT_FILE, parse_dates=["date"])

    all_metrics = []
    for secid in ["GLDRUB_TOM", "SLVRUB_TOM"]:
        k = BEST_K.get(secid, 20)
        print(f"\n[{secid}]  k={k}, sent_weight={SENT_WEIGHT}")

        pred_df, metrics = run_knn(secid, rv, sent, k=k)

        # Сохраняем прогнозы
        out_pred = OUT_DIR / f"{secid}_knn_predictions.csv"
        pred_df.to_csv(out_pred, index=False)
        print(f"  Прогнозы → {out_pred.name}  ({len(pred_df)} строк)")

        # Вывод метрик
        print(f"  MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}"
              f"  R²_OOS={metrics['R2_OOS']:.3f}  QLIKE={metrics['QLIKE']:.3f}")
        all_metrics.append(metrics)

    # Сохраняем сводную таблицу
    summary = pd.DataFrame(all_metrics)
    out_sum = OUT_DIR / "knn_summary.csv"
    summary.to_csv(out_sum, index=False)
    print(f"\nСводка → {out_sum.name}")
    print(summary[["secid","model","MAE","RMSE","R2_OOS","QLIKE"]].to_string(index=False))


if __name__ == "__main__":
    main()

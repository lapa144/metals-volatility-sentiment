"""
aggregate_sentiment.py
======================
Агрегирует все источники sentiment/attention в один дневной CSV
для использования в моделях (HAR-S, XGBoost, LSTM).

Источники:
  1. RSS новости (rss_news.csv) + NLP scoring → текстовый sentiment
  2. GDELT (gdelt_sentiment.csv)              → глобальный sentiment
  3. Google Trends (google_trends.csv)        → attention proxy
  4. ЦБ РФ (cbr_data.csv)                    → макро (USD/RUB, ставка)
  5. FRED (fred_macro.csv)                    → макро (нефть, VIX, DXY)
  6. MOEX объёмы (metals_volume.csv)          → attention по биржевым данным

Выходной файл: data/sentiment/daily_sentiment.csv

Колонки результата:
  date                 — дата
  sentiment_score      — итоговый текстовый sentiment [-1, 1]
  sentiment_n          — количество статей за день
  sentiment_gdelt      — sentiment из GDELT [-1, 1]
  attention_google     — Google Trends [0, 1]
  attention_volume     — нормированный объём торгов MOEX [0, 1]
  macro_usd_rub        — курс USD/RUB
  macro_usd_rub_ret    — лог-доходность USD/RUB
  macro_key_rate       — ключевая ставка ЦБ РФ
  macro_key_rate_chg   — 1 если ставка изменилась
  macro_brent          — нефть Brent ($/баррель)
  macro_brent_ret      — лог-доходность Brent
  macro_vix            — VIX
  macro_dxy            — индекс доллара DXY
  macro_gold_fix       — золото London AM fix ($)
  sentiment_combined   — взвешенное среднее всех sentiment-сигналов

Запуск:
    python src/data/aggregate_sentiment.py
    python src/data/aggregate_sentiment.py --secid GLDRUB_TOM
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
SENT_DIR   = BASE_DIR / "data" / "sentiment"
PROC_DIR   = BASE_DIR / "data" / "processed"
OUTPUT_FILE = SENT_DIR / "daily_sentiment.csv"


# ──────────────────────────────────────────────────────────────
# 1. RSS / NLP sentiment
# ──────────────────────────────────────────────────────────────

def load_rss_sentiment(path: Path = SENT_DIR / "rss_news.csv") -> pd.DataFrame:
    """
    Загружает rss_news.csv, считает простой лексический sentiment
    если NLP scorer недоступен, иначе использует готовые оценки.
    Возвращает дневную агрегацию.
    """
    if not path.exists():
        log.warning("rss_news.csv не найден: %s", path)
        return pd.DataFrame(columns=["date", "sentiment_score", "sentiment_n"])

    df = pd.read_csv(path)

    # Парсим дату публикации
    df["date"] = pd.to_datetime(df["pub_date"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date"])

    # Если есть готовый sentiment_score — используем
    if "sentiment_score" in df.columns:
        df["score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0)
    else:
        # Простой словарный sentiment как fallback
        df["score"] = _lexical_sentiment(df["title"].fillna("") + " " + df["description"].fillna(""))

    # Агрегация по дням
    daily = (
        df.groupby("date")
        .agg(
            sentiment_score=("score", "mean"),
            sentiment_n=("score", "count"),
        )
        .reset_index()
    )
    log.info("RSS sentiment: %d дней, %.3f среднее",
             len(daily), daily["sentiment_score"].mean())
    return daily


def _lexical_sentiment(texts: pd.Series) -> pd.Series:
    """
    Быстрый словарный sentiment по ключевым словам (fallback без NLP).
    Возвращает Series со значениями в [-1, 1].
    """
    positive = [
        "рост", "растёт", "увеличен", "повышен", "рекордн", "подъём",
        "позитив", "оптимизм", "укрепл", "profit", "gain", "rise", "high",
        "спрос растёт", "купить", "инвестиц",
    ]
    negative = [
        "падени", "снижени", "обвал", "кризис", "риск", "потер",
        "санкц", "негатив", "пессимизм", "распродаж", "loss", "fall",
        "decline", "слабост", "давлени",
    ]

    scores = []
    for text in texts:
        t = str(text).lower()
        pos = sum(1 for w in positive if w in t)
        neg = sum(1 for w in negative if w in t)
        total = pos + neg
        if total == 0:
            scores.append(0.0)
        else:
            scores.append((pos - neg) / total)

    return pd.Series(scores, index=texts.index)


# ──────────────────────────────────────────────────────────────
# 2. GDELT sentiment
# ──────────────────────────────────────────────────────────────

def load_gdelt_sentiment(path: Path = SENT_DIR / "gdelt_sentiment.csv") -> pd.DataFrame:
    """
    Загружает gdelt_sentiment.csv.
    Ожидаемые колонки: date, tone (среднее значение GDELT Tone).
    GDELT Tone: положительный = позитив, отрицательный = негатив.
    Нормализуем в [-1, 1].
    """
    if not path.exists():
        log.warning("gdelt_sentiment.csv не найден: %s", path)
        return pd.DataFrame(columns=["date", "sentiment_gdelt"])

    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()

    tone_col = next((c for c in df.columns if "tone" in c.lower()), None)
    if tone_col is None:
        log.warning("Колонка tone не найдена в GDELT CSV")
        return pd.DataFrame(columns=["date", "sentiment_gdelt"])

    # GDELT tone обычно в диапазоне [-10, 10], нормализуем
    daily = df.groupby("date")[tone_col].mean().reset_index()
    daily.columns = ["date", "sentiment_gdelt"]
    daily["sentiment_gdelt"] = (daily["sentiment_gdelt"] / 10.0).clip(-1, 1)

    log.info("GDELT sentiment: %d дней", len(daily))
    return daily


# ──────────────────────────────────────────────────────────────
# 3. Google Trends
# ──────────────────────────────────────────────────────────────

def load_google_trends(path: Path = SENT_DIR / "google_trends.csv") -> pd.DataFrame:
    if not path.exists():
        log.warning("google_trends.csv не найден: %s", path)
        return pd.DataFrame(columns=["date", "attention_google"])

    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()

    if "attention_google" not in df.columns:
        # Берём среднее по всем числовым колонкам
        num_cols = df.select_dtypes(include="number").columns.tolist()
        df["attention_google"] = df[num_cols].mean(axis=1)

    log.info("Google Trends: %d дней", len(df))
    return df[["date", "attention_google"]]


# ──────────────────────────────────────────────────────────────
# 4. MOEX объёмы как attention
# ──────────────────────────────────────────────────────────────

def load_moex_volumes(
    path: Path = PROC_DIR / "metals_volume.csv",
    secid: str = "GLDRUB_TOM",
) -> pd.DataFrame:
    """
    Загружает объёмы MOEX, нормирует 30-дневным rolling-z-score
    → attention_volume (0-1 через sigmoid).
    """
    # Fallback: берём из основного panel если metals_volume.csv нет
    if not path.exists():
        panel_path = PROC_DIR / "metals_1d_panel.csv"
        if not panel_path.exists():
            log.warning("metals_volume.csv и metals_1d_panel.csv не найдены")
            return pd.DataFrame(columns=["date", "attention_volume"])
        df = pd.read_csv(panel_path, parse_dates=["date"])
    else:
        df = pd.read_csv(path, parse_dates=["date"])

    df = df[df["secid"] == secid][["date", "volume"]].copy()
    df["date"] = df["date"].dt.normalize()
    df = df.dropna(subset=["volume"])

    # Z-score за 30 дней → sigmoid → [0,1]
    roll_mean = df["volume"].rolling(30, min_periods=5).mean()
    roll_std  = df["volume"].rolling(30, min_periods=5).std().replace(0, 1)
    z = (df["volume"] - roll_mean) / roll_std
    df["attention_volume"] = 1 / (1 + np.exp(-z))

    log.info("MOEX volumes (%s): %d дней", secid, len(df))
    return df[["date", "attention_volume"]]


# ──────────────────────────────────────────────────────────────
# 5. ЦБ РФ + FRED макро
# ──────────────────────────────────────────────────────────────

def load_cbr(path: Path = SENT_DIR / "cbr_data.csv") -> pd.DataFrame:
    if not path.exists():
        log.warning("cbr_data.csv не найден")
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()
    rename = {
        "usd_rub"       : "macro_usd_rub",
        "usd_rub_ret"   : "macro_usd_rub_ret",
        "eur_rub"       : "macro_eur_rub",
        "key_rate"      : "macro_key_rate",
        "key_rate_change": "macro_key_rate_chg",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    log.info("CBR данные: %d дней", len(df))
    return df


# ──────────────────────────────────────────────────────────────
# 6. Telegram sentiment
# ──────────────────────────────────────────────────────────────

def load_telegram(path: Path = SENT_DIR / "telegram_sentiment.csv") -> pd.DataFrame:
    """
    Загружает дневной Telegram-sentiment из telegram_sentiment.csv.
    Колонки: date, tg_score, tg_n, tg_channels, tg_views_total
    """
    if not path.exists():
        log.warning("telegram_sentiment.csv не найден (запусти parse_telegram.py)")
        return pd.DataFrame(columns=["date", "tg_score", "tg_n"])
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()
    log.info("Telegram sentiment: %d дней, %.4f среднее tg_score",
             len(df), df["tg_score"].mean() if "tg_score" in df.columns else 0)
    return df[["date", "tg_score", "tg_n"]]


def load_fred(path: Path = SENT_DIR / "fred_macro.csv") -> pd.DataFrame:
    if not path.exists():
        log.warning("fred_macro.csv не найден")
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()
    rename = {
        "brent_usd"     : "macro_brent",
        "brent_usd_ret" : "macro_brent_ret",
        "vix"           : "macro_vix",
        "vix_ret"       : "macro_vix_ret",
        "dxy"           : "macro_dxy",
        "dxy_ret"       : "macro_dxy_ret",
        "gold_usd_fix"  : "macro_gold_fix",
        "fed_rate"      : "macro_fed_rate",
        "fed_rate_change": "macro_fed_rate_chg",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    log.info("FRED данные: %d дней, %d колонок", len(df), len(df.columns))
    return df


# ──────────────────────────────────────────────────────────────
# Главная функция агрегации
# ──────────────────────────────────────────────────────────────

def aggregate(
    secid: str = "GLDRUB_TOM",
    date_from: str = "2018-01-01",
    date_to: str | None = None,
    save_path: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Собирает все источники в один дневной DataFrame.
    Недостающие значения заполняются: числа → ffill + 0, sentiment → 0.
    """
    if date_to is None:
        import datetime
        date_to = datetime.date.today().strftime("%Y-%m-%d")

    idx = pd.DataFrame(
        {"date": pd.date_range(start=date_from, end=date_to, freq="D")}
    )

    # Загружаем все источники
    rss     = load_rss_sentiment()
    gdelt   = load_gdelt_sentiment()
    trends  = load_google_trends()
    volumes = load_moex_volumes(secid=secid)
    cbr     = load_cbr()
    fred    = load_fred()
    tg      = load_telegram()

    # Объединяем на дневном индексе
    df = idx.copy()
    for src in [rss, gdelt, trends, volumes, cbr, fred, tg]:
        if src.empty or "date" not in src.columns:
            continue
        df = df.merge(src, on="date", how="left")

    df = df.set_index("date").sort_index()

    # ── Финальные признаки ────────────────────────────────────────────────
    # Текстовый sentiment: RSS и GDELT
    sent_cols = []
    if "sentiment_score" in df.columns:
        df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
        sent_cols.append(("sentiment_score", 0.4))
    if "sentiment_gdelt" in df.columns:
        df["sentiment_gdelt"] = df["sentiment_gdelt"].fillna(0.0)
        sent_cols.append(("sentiment_gdelt", 0.4))
    if "tg_score" in df.columns:
        df["tg_score"] = df["tg_score"].fillna(0.0)
        sent_cols.append(("tg_score", 0.2))

    # Итоговый взвешенный sentiment
    if sent_cols:
        total_w = sum(w for _, w in sent_cols)
        df["sentiment_combined"] = sum(
            df[col] * w for col, w in sent_cols
        ) / total_w
    else:
        df["sentiment_combined"] = 0.0

    # Attention: нормируем Google Trends и volume
    if "attention_google" in df.columns:
        df["attention_google"] = df["attention_google"].ffill().fillna(0.5)
    else:
        df["attention_google"] = 0.5

    if "attention_volume" in df.columns:
        df["attention_volume"] = df["attention_volume"].ffill().fillna(0.5)
    else:
        df["attention_volume"] = 0.5

    # Макро — forward fill (нет данных в выходные)
    macro_cols = [c for c in df.columns if c.startswith("macro_")]
    df[macro_cols] = df[macro_cols].ffill()

    # Заполняем оставшиеся NaN нулями для бинарных и медианой для числовых
    for col in df.columns:
        if col.endswith("_chg") or col == "sentiment_n":
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(df[col].median())

    log.info("Итоговый датасет: %d дней × %d признаков", len(df), len(df.columns))
    log.info("Признаки: %s", list(df.columns))

    # Сохраняем
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)
    log.info("Сохранено → %s", save_path)

    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("СВОДКА: daily_sentiment.csv")
    print("="*60)
    print(f"Период:   {df.index.min().date()} — {df.index.max().date()}")
    print(f"Дней:     {len(df)}")
    print(f"Признаков: {len(df.columns)}")
    print(f"\nПервые строки:")
    key_cols = [c for c in [
        "sentiment_score", "sentiment_gdelt", "sentiment_combined",
        "attention_google", "attention_volume",
        "macro_usd_rub", "macro_key_rate", "macro_brent", "macro_vix",
    ] if c in df.columns]
    print(df[key_cols].head().to_string())
    print(f"\nДоля непустых значений:")
    print((df[key_cols].notna().mean() * 100).round(1).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Агрегатор sentiment/attention")
    parser.add_argument("--secid", default="GLDRUB_TOM",
                        help="Тикер металла для volume attention")
    parser.add_argument("--from", dest="start", default="2018-01-01")
    parser.add_argument("--to",   dest="end",   default=None)
    args = parser.parse_args()

    df = aggregate(args.secid, args.start, args.end)
    print_summary(df)


if __name__ == "__main__":
    main()

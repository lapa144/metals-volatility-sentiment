"""
Парсер GDELT 2.0 — глобальная база событий и тональности новостей.
Использует GDELT GKG (Global Knowledge Graph) для получения sentiment
по темам, связанным с драгоценными металлами.

Документация: https://www.gdeltproject.org/data.html#rawdatafiles
GKG колонки: https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf

Сохраняет данные в data/sentiment/gdelt_sentiment.csv

Запуск:
    # Полная история под золото/серебро (совпадает с MOEX данными)
    python src/data/parse_gdelt.py --from 2018-01-01

    # Только под платину/палладий
    python src/data/parse_gdelt.py --from 2023-01-01

    # Дообновить до сегодня (быстро, если большинство уже скачано)
    python src/data/parse_gdelt.py --from 2026-05-01

Данные сохраняются инкрементально — повторный запуск не перезапишет старое.
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# GDELT GKG 2.0 — ежедневные файлы
GDELT_BASE = "http://data.gdeltproject.org/gdeltv2/{date}000000.gkg.csv.zip"

# Ключевые темы и организации для фильтрации
# GDELT использует категории вида TAX_GOLDSTANDARD, NATURAL_DISASTER и т.д.
THEME_KEYWORDS = [
    "GOLD", "SILVER", "PLATINUM", "PALLADIUM",
    "PRECIOUS_METAL", "COMMODITY", "MINING",
    "RUSSIA", "MOEX", "CENTRAL_BANK",
    "INFLATION", "INTEREST_RATE", "SANCTION",
]

# Колонки нужные нам из GKG (индексы по документации)
# Файл — TSV без заголовка
GKG_COLS = {
    0:  "gkgrecordid",
    1:  "date",
    7:  "themes",
    9:  "locations",
    15: "tone",          # формат: tone,pos,neg,polarity,activity,selfref,wordcount
    17: "sourceurl",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sentiment"
OUTPUT_FILE = OUTPUT_DIR / "gdelt_sentiment.csv"


def fetch_gdelt_day(target_date: date) -> pd.DataFrame | None:
    """Скачивает и парсит один дневной GKG файл."""
    date_str = target_date.strftime("%Y%m%d")
    url = GDELT_BASE.format(date=date_str)

    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            log.warning("Файл не найден для даты %s (404)", date_str)
            return None
        resp.raise_for_status()
    except Exception as e:
        log.warning("Ошибка загрузки %s: %s", date_str, e)
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            fname = z.namelist()[0]
            with z.open(fname) as f:
                # GKG — TSV без заголовка, очень много колонок (27)
                df = pd.read_csv(
                    f,
                    sep="\t",
                    header=None,
                    on_bad_lines="skip",
                    dtype=str,
                    low_memory=False,
                )
    except Exception as e:
        log.warning("Ошибка парсинга %s: %s", date_str, e)
        return None

    # Берём только нужные колонки (если они есть)
    keep = {k: v for k, v in GKG_COLS.items() if k < df.shape[1]}
    df = df[list(keep.keys())].rename(columns=keep)

    # Фильтрация по темам
    if "themes" in df.columns:
        mask = df["themes"].fillna("").str.upper().apply(
            lambda t: any(kw in t for kw in THEME_KEYWORDS)
        )
        df = df[mask]

    if df.empty:
        return df

    # Парсим tone: первое значение — общий тон (позитивный = +, негативный = -)
    if "tone" in df.columns:
        df["tone_score"] = pd.to_numeric(
            df["tone"].str.split(",").str[0], errors="coerce"
        )
        df["tone_positive"] = pd.to_numeric(
            df["tone"].str.split(",").str[1], errors="coerce"
        )
        df["tone_negative"] = pd.to_numeric(
            df["tone"].str.split(",").str[2], errors="coerce"
        )

    df["fetch_date"] = target_date.strftime("%Y-%m-%d")
    log.info("%s: %d релевантных записей", date_str, len(df))
    return df


def aggregate_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует тональность по дате — получаем один дневной индекс."""
    if df.empty or "tone_score" not in df.columns:
        return pd.DataFrame()

    agg = df.groupby("fetch_date").agg(
        gdelt_tone_mean=("tone_score", "mean"),
        gdelt_tone_std=("tone_score", "std"),
        gdelt_positive_mean=("tone_positive", "mean"),
        gdelt_negative_mean=("tone_negative", "mean"),
        gdelt_n_articles=("tone_score", "count"),
    ).reset_index()
    agg.rename(columns={"fetch_date": "date"}, inplace=True)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GDELT sentiment парсер",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры запуска:
  # Полная история под золото/серебро (с 2018)
  python parse_gdelt.py --from 2018-01-01

  # Только под платину/палладий (с 2023)
  python parse_gdelt.py --from 2023-01-01

  # Последний год
  python parse_gdelt.py --from 2025-01-01

  # Дообновить до сегодня
  python parse_gdelt.py --from 2026-05-01

Внимание: каждый день = 1 запрос к GDELT. За 2 года ~ 700 запросов,
займёт 15-30 минут. Результат сохраняется инкрементально.
        """
    )
    parser.add_argument(
        "--from", dest="date_from",
        default="2018-01-01",   # ← совпадает со стартом данных MOEX по золоту
        help="Начальная дата YYYY-MM-DD (default: 2018-01-01)"
    )
    parser.add_argument(
        "--till",
        default=date.today().strftime("%Y-%m-%d"),
        help="Конечная дата YYYY-MM-DD (default: сегодня)"
    )
    args = parser.parse_args()

    till = datetime.strptime(args.till, "%Y-%m-%d").date()
    from_date = datetime.strptime(args.date_from, "%Y-%m-%d").date()

    log.info("Период: %s → %s", from_date, till)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_raw = []
    current = from_date
    while current <= till:
        df_day = fetch_gdelt_day(current)
        if df_day is not None and not df_day.empty:
            all_raw.append(df_day)
        current += timedelta(days=1)

    if not all_raw:
        log.error("Нет данных за выбранный период")
        return

    df_all = pd.concat(all_raw, ignore_index=True)
    df_agg = aggregate_by_day(df_all)

    # Дозаписываем в CSV
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        df_agg = pd.concat([existing, df_agg]).drop_duplicates("date").sort_values("date")

    df_agg.to_csv(OUTPUT_FILE, index=False)
    log.info("Сохранено %d дней → %s", len(df_agg), OUTPUT_FILE)
    print(df_agg.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()

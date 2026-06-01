"""
Парсер дополнительных данных MOEX: объёмы торгов и количество сделок.
Используется как proxy-переменная для "attention" инвесторов.
Сохраняет данные в data/processed/metals_volume.csv

Запуск:
    python src/data/parse_moex_extra.py --till 2026-05-27
"""
from __future__ import annotations

import argparse
import logging
import time
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

BASE_URL = "https://iss.moex.com/iss/engines/currency/markets/selt/boards/CETS/securities"
INTERVAL = 24  # дневной интервал
TIMEOUT = 30
RETRY_DELAYS = (1, 2, 4)

SECIDS = ["GLDRUB_TOM", "SLVRUB_TOM", "PLTRUB_TOM", "PLDRUB_TOM"]

START_DATES = {
    "GLDRUB_TOM": "2018-01-01",
    "SLVRUB_TOM": "2018-01-01",
    "PLTRUB_TOM": "2023-01-01",
    "PLDRUB_TOM": "2023-01-01",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "metals_volume.csv"


def fetch_candles(secid: str, date_from: str, date_till: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{secid}/candles.json"
    params = {
        "from": date_from,
        "till": date_till,
        "interval": INTERVAL,
        "iss.meta": "off",
    }
    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            break
        except Exception as e:
            log.warning("Ошибка %s (попытка %d): %s", secid, attempt + 1, e)
            if attempt < len(RETRY_DELAYS) - 1:
                time.sleep(delay)
            else:
                return pd.DataFrame()

    data = resp.json().get("candles", {})
    columns = data.get("columns", [])
    rows = data.get("data", [])
    if not columns or not rows:
        return pd.DataFrame()

    col_idx = {c: i for i, c in enumerate(columns)}

    records = []
    for row in rows:
        try:
            records.append({
                "date": row[col_idx["begin"]][:10],
                "secid": secid,
                "open":   float(row[col_idx["open"]]),
                "high":   float(row[col_idx["high"]]),
                "low":    float(row[col_idx["low"]]),
                "close":  float(row[col_idx["close"]]),
                "volume": float(row[col_idx["volume"]]) if "volume" in col_idx else None,
                "value":  float(row[col_idx["value"]]) if "value" in col_idx else None,
                # numtrades — количество сделок за день (proxy для attention)
                "numtrades": int(row[col_idx["numtrades"]]) if "numtrades" in col_idx else None,
            })
        except Exception:
            continue

    return pd.DataFrame(records)


def download_all(till: str) -> pd.DataFrame:
    frames = []
    for secid in SECIDS:
        date_from = START_DATES[secid]
        log.info("Скачиваю %s с %s по %s...", secid, date_from, till)

        # Чанкуем по годам
        current = datetime.strptime(date_from, "%Y-%m-%d").date()
        end = datetime.strptime(till, "%Y-%m-%d").date()
        chunks = []

        while current <= end:
            chunk_end = min(current + timedelta(days=365), end)
            df = fetch_candles(secid, current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
            if not df.empty:
                chunks.append(df)
            current = chunk_end + timedelta(days=1)

        if chunks:
            df_sec = pd.concat(chunks, ignore_index=True).drop_duplicates("date")
            log.info("%s: %d строк", secid, len(df_sec))
            frames.append(df_sec)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True).sort_values(["secid", "date"]).reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачивание объёмов торгов металлов с MOEX")
    parser.add_argument("--till", default=date.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = download_all(args.till)

    if df.empty:
        log.error("Данные не получены")
        return

    df.to_csv(OUTPUT_FILE, index=False)
    log.info("Сохранено %d строк → %s", len(df), OUTPUT_FILE)

    # Краткая статистика
    for secid, g in df.groupby("secid"):
        has_numtrades = g["numtrades"].notna().sum()
        print(f"  {secid}: {len(g)} дней, numtrades доступен: {has_numtrades} дней")


if __name__ == "__main__":
    main()

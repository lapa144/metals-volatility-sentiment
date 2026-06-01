#!/usr/bin/env python3
"""
Скачивание дневных свечей по металлам MOEX (GLDRUB_TOM, SLVRUB_TOM, PLTRUB_TOM, PLDRUB_TOM)
с MOEX ISS API. Сохраняет raw CSV и объединённый panel.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# --- Конфигурация ---
BASE_URL = "https://iss.moex.com/iss/engines/currency/markets/selt/boards/CETS/securities"
INTERVAL = 24  # дневные свечи
TIMEOUT = 30
RETRY_DELAYS = (1, 2, 4)  # секунды между попытками

SECIDS = ["GLDRUB_TOM", "SLVRUB_TOM", "PLTRUB_TOM", "PLDRUB_TOM"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    today = date.today().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(
        description="Скачивание дневных свечей металлов MOEX в RUB"
    )
    parser.add_argument(
        "--till",
        default=today,
        help=f"Конечная дата (default: {today})",
    )
    parser.add_argument(
        "--gld_from",
        default="2018-01-01",
        help="Начало периода для Gold GLDRUB_TOM (default: 2018-01-01)",
    )
    parser.add_argument(
        "--slv_from",
        default="2018-01-01",
        help="Начало периода для Silver SLVRUB_TOM (default: 2018-01-01)",
    )
    parser.add_argument(
        "--plt_from",
        default="2023-01-01",
        help="Начало периода для Platinum PLTRUB_TOM (default: 2023-01-01)",
    )
    parser.add_argument(
        "--pld_from",
        default="2023-01-01",
        help="Начало периода для Palladium PLDRUB_TOM (default: 2023-01-01)",
    )
    parser.add_argument(
        "--chunk_days",
        type=int,
        default=365,
        help="Размер чанка в днях (default: 365)",
    )
    parser.add_argument(
        "--outdir",
        default="data",
        help="Корневая папка для данных (default: data)",
    )
    return parser.parse_args()


def _parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def fetch_chunk(secid: str, date_from: str, date_till: str) -> pd.DataFrame | None:
    """Загружает один чанк свечей с retry."""
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
        except (requests.RequestException, requests.Timeout) as e:
            log.warning(
                "HTTP/timeout %s (attempt %d/%d): %s",
                secid,
                attempt + 1,
                len(RETRY_DELAYS),
                e,
            )
            if attempt < len(RETRY_DELAYS) - 1:
                time.sleep(delay)
            else:
                log.error("Пропускаем чанк %s %s–%s после всех попыток", secid, date_from, date_till)
                return None

    try:
        data = resp.json()
    except ValueError as e:
        log.error("Неверный JSON от MOEX: %s", e)
        return None

    candles_block = data.get("candles")
    if not candles_block:
        return pd.DataFrame()

    columns = candles_block.get("columns", [])
    rows = candles_block.get("data", [])
    if not columns or not rows:
        return pd.DataFrame()

    col_idx = {c: i for i, c in enumerate(columns)}
    required = ["begin", "open", "high", "low", "close"]
    for r in required:
        if r not in col_idx:
            log.warning("Колонка %s отсутствует в ответе MOEX", r)
            return pd.DataFrame()

    records = []
    for row in rows:
        try:
            begin_str = row[col_idx["begin"]]
            open_ = float(row[col_idx["open"]])
            high = float(row[col_idx["high"]])
            low = float(row[col_idx["low"]])
            close = float(row[col_idx["close"]])
            volume = float(row[col_idx["volume"]]) if "volume" in col_idx else float("nan")
            value = float(row[col_idx["value"]]) if "value" in col_idx else float("nan")
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        records.append({
            "begin": begin_str,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "value": value,
        })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def download_secid(
    secid: str,
    date_from: date,
    date_till: date,
    chunk_days: int,
    out_raw: Path,
) -> pd.DataFrame:
    """Скачивает все чанки для одного SECID, склеивает и сохраняет."""
    chunks = []
    current = date_from
    total_duplicates = 0

    while current <= date_till:
        chunk_end = min(current + timedelta(days=chunk_days), date_till)
        from_str = current.strftime("%Y-%m-%d")
        till_str = chunk_end.strftime("%Y-%m-%d")

        log.info("SECID=%s chunk_from=%s chunk_till=%s", secid, from_str, till_str)
        df = fetch_chunk(secid, from_str, till_str)

        if df is None:
            current = chunk_end + timedelta(days=1)
            continue

        rows_received = len(df)
        log.info("SECID=%s chunk_from=%s chunk_till=%s rows_received=%d", secid, from_str, till_str, rows_received)

        if not df.empty:
            chunks.append(df)
        current = chunk_end + timedelta(days=1)

    if not chunks:
        empty_df = pd.DataFrame(columns=["begin", "open", "high", "low", "close", "volume", "value"])
        empty_df.to_csv(out_raw, index=False)
        return empty_df

    combined = pd.concat(chunks, ignore_index=True)
    n_before = len(combined)

    # Удаление дубликатов по begin
    combined = combined.drop_duplicates(subset=["begin"], keep="first")
    total_duplicates = n_before - len(combined)

    # Сортировка по дате
    combined["date"] = pd.to_datetime(combined["begin"]).dt.strftime("%Y-%m-%d")
    combined = combined.sort_values("begin").reset_index(drop=True)

    # Сохраняем raw: begin, open, high, low, close, volume, value
    out_cols = ["begin", "open", "high", "low", "close", "volume", "value"]
    combined[out_cols].to_csv(out_raw, index=False)

    # Sanity summary
    n_rows = len(combined)
    min_date = combined["date"].min() if n_rows else "N/A"
    max_date = combined["date"].max() if n_rows else "N/A"
    count_missing_close = combined["close"].isna().sum()
    log.info(
        "SECID=%s n_rows=%d min_date=%s max_date=%s count_missing_close=%d count_duplicates_removed=%d",
        secid,
        n_rows,
        min_date,
        max_date,
        count_missing_close,
        total_duplicates,
    )
    print(
        f"  {secid}: n_rows={n_rows}, min_date={min_date}, max_date={max_date}, "
        f"count_missing_close={count_missing_close}, count_duplicates_removed={total_duplicates}",
        flush=True,
    )

    return combined


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent.parent
    outdir = root / args.outdir
    raw_dir = outdir / "raw"
    processed_dir = outdir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    till = _parse_date(args.till)
    secid_dates = {
        "GLDRUB_TOM": _parse_date(args.gld_from),
        "SLVRUB_TOM": _parse_date(args.slv_from),
        "PLTRUB_TOM": _parse_date(args.plt_from),
        "PLDRUB_TOM": _parse_date(args.pld_from),
    }

    panels = []
    for secid in SECIDS:
        date_from = secid_dates[secid]
        if date_from > till:
            log.warning("SECID=%s: date_from > till, пропускаем", secid)
            continue
        out_raw = raw_dir / f"{secid}_1d.csv"
        df = download_secid(secid, date_from, till, args.chunk_days, out_raw)
        if not df.empty:
            df["secid"] = secid
            panels.append(df[["secid", "date", "open", "high", "low", "close", "volume", "value"]])

    if panels:
        panel = pd.concat(panels, ignore_index=True)
        panel = panel.sort_values(["secid", "date"]).reset_index(drop=True)
        out_panel = processed_dir / "metals_1d_panel.csv"
        panel.to_csv(out_panel, index=False)
        log.info("Panel сохранён: %s (%d строк)", out_panel, len(panel))

    return 0


if __name__ == "__main__":
    sys.exit(main())

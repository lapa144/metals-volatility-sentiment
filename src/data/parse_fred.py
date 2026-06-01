"""
parse_fred.py
=============
Загрузка макроэкономических данных через FRED API (Federal Reserve Bank of St. Louis).
Бесплатно, нужна только регистрация: https://fred.stlouisfed.org/docs/api/api_key.html

Загружаемые серии:
  DCOILBRENTEU  — нефть Brent ($/баррель), дневная
  DTWEXBGS      — индекс доллара DXY (взвешенный), дневной
  VIXCLS        — индекс волатильности VIX (CBOE), дневной
  CPIAUCSL      — CPI США (инфляция), месячный → интерполируем на дни
  FEDFUNDS      — ставка ФРС, месячная → интерполируем
  GOLDAMGBD228NLBM — цена золота London AM fix ($), дневная
  SLVPRUSD      — цена серебра ($), дневная

Сохраняет: data/sentiment/fred_macro.csv

Запуск:
    # API ключ через env-переменную (рекомендуется)
    export FRED_API_KEY=ваш_ключ
    python src/data/parse_fred.py

    # Или напрямую
    python src/data/parse_fred.py --api-key ваш_ключ --from 2018-01-01

Получить ключ бесплатно: https://fred.stlouisfed.org/docs/api/api_key.html
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "sentiment" / "fred_macro.csv"

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ─── Серии FRED ─────────────────────────────────────────────────────────────
FRED_SERIES: dict[str, dict] = {
    # Сырьё
    "brent_usd": {
        "series_id": "DCOILBRENTEU",
        "freq": "daily",
        "desc": "Нефть Brent ($/баррель)",
    },
    "gold_usd_fix": {
        "series_id": "GOLDAMGBD228NLBM",
        "freq": "daily",
        "desc": "Золото London AM fix ($/тр.унция)",
    },
    "silver_usd": {
        "series_id": "SLVPRUSD",
        "freq": "daily",
        "desc": "Серебро ($/тр.унция)",
    },
    # Финансовые индексы
    "dxy": {
        "series_id": "DTWEXBGS",
        "freq": "daily",
        "desc": "Индекс доллара США (DXY-аналог)",
    },
    "vix": {
        "series_id": "VIXCLS",
        "freq": "daily",
        "desc": "VIX — индекс волатильности S&P 500",
    },
    # Макро (месячные, интерполируем)
    "cpi_us": {
        "series_id": "CPIAUCSL",
        "freq": "monthly",
        "desc": "CPI США (инфляция)",
    },
    "fed_rate": {
        "series_id": "FEDFUNDS",
        "freq": "monthly",
        "desc": "Ставка ФРС (%)",
    },
}


def fetch_series(
    series_id: str,
    api_key: str,
    date_from: str,
    date_to: str,
    timeout: int = 30,
    retries: int = 3,
) -> pd.Series:
    """
    Загружает одну серию из FRED API.
    """
    params = {
        "series_id"       : series_id,
        "api_key"         : api_key,
        "file_type"       : "json",
        "observation_start": date_from,
        "observation_end" : date_to,
    }

    for attempt in range(retries):
        try:
            resp = requests.get(FRED_BASE, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            log.warning("FRED %s попытка %d/%d: %s", series_id, attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    else:
        return pd.Series(dtype=float, name=series_id)

    obs = data.get("observations", [])
    if not obs:
        log.warning("FRED %s: нет наблюдений", series_id)
        return pd.Series(dtype=float, name=series_id)

    records = {}
    for o in obs:
        try:
            val = float(o["value"])
            dt  = pd.Timestamp(o["date"])
            records[dt] = val
        except (ValueError, KeyError):
            continue  # "." означает пропущенное значение в FRED

    series = pd.Series(records, name=series_id, dtype=float)
    series.index.name = "date"
    log.info("FRED %s: %d наблюдений", series_id, len(series))
    return series


def build_fred_panel(
    api_key: str,
    date_from: str = "2018-01-01",
    date_to: str | None = None,
    pause: float = 0.5,
) -> pd.DataFrame:
    """
    Загружает все серии и собирает дневной DataFrame.
    Месячные серии интерполируются на дни (forward-fill + linear).
    Добавляет производные признаки (лог-доходности).
    """
    if date_to is None:
        date_to = date.today().strftime("%Y-%m-%d")

    idx = pd.date_range(start=date_from, end=date_to, freq="D", name="date")
    df  = pd.DataFrame(index=idx)

    for col_name, meta in FRED_SERIES.items():
        log.info("Загружаем: %s (%s)", col_name, meta["desc"])
        series = fetch_series(meta["series_id"], api_key, date_from, date_to)

        if series.empty:
            log.warning("  Пропускаем %s", col_name)
            continue

        if meta["freq"] == "monthly":
            # Месячные → дневные через forward-fill
            series = series.reindex(idx).ffill()
        else:
            # Дневные → просто реиндексируем, forward-fill для выходных
            series = series.reindex(idx).ffill()

        df[col_name] = series.values
        time.sleep(pause)

    # ── Производные признаки ─────────────────────────────────────────────────
    for col in ["brent_usd", "gold_usd_fix", "silver_usd", "dxy", "vix"]:
        if col in df.columns:
            df[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))

    # Изменение ставки ФРС (бинарный признак)
    if "fed_rate" in df.columns:
        df["fed_rate_change"] = (df["fed_rate"].diff().fillna(0) != 0).astype(int)

    # YoY инфляция США
    if "cpi_us" in df.columns:
        df["cpi_us_yoy"] = df["cpi_us"].pct_change(periods=252) * 100

    log.info("FRED панель: %d строк × %d колонок", len(df), len(df.columns))
    return df


def save(df: pd.DataFrame, path: Path = OUTPUT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = df.combine_first(existing).sort_index()

    df.to_csv(path)
    log.info("Сохранено → %s (%d строк, %d колонок)", path, len(df), len(df.columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Парсер FRED макроданных")
    parser.add_argument("--api-key", default=os.environ.get("FRED_API_KEY", ""),
                        help="FRED API ключ (или env FRED_API_KEY). Получить: https://fred.stlouisfed.org/docs/api/api_key.html")
    parser.add_argument("--from", dest="start", default="2018-01-01")
    parser.add_argument("--to",   dest="end",   default=None)
    args = parser.parse_args()

    if not args.api_key:
        print(
            "\n[!] FRED API ключ не задан!\n"
            "    Зарегистрируйтесь бесплатно: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "    Затем:\n"
            "      export FRED_API_KEY=ваш_ключ\n"
            "      python src/data/parse_fred.py\n"
            "    Или:\n"
            "      python src/data/parse_fred.py --api-key ваш_ключ\n"
        )
        return

    df = build_fred_panel(args.api_key, args.start, args.end)

    print(f"\nКолонки: {list(df.columns)}")
    print(f"\nОписательная статистика:")
    show_cols = [c for c in ["brent_usd", "gold_usd_fix", "vix", "dxy"] if c in df.columns]
    if show_cols:
        print(df[show_cols].describe().round(2))

    save(df)


if __name__ == "__main__":
    main()

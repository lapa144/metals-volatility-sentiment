"""
parse_cbr.py
============
Парсер данных Банка России:
  1. Курс USD/RUB (официальный, каждый рабочий день)
  2. Курс EUR/RUB
  3. История ключевой ставки ЦБ РФ

Источник: XML API ЦБ РФ — без регистрации и API ключей.
  Курсы: https://www.cbr.ru/scripts/XML_dynamic.asp
  Ставка: https://www.cbr.ru/hd_base/KeyRate/ (HTML-таблица)

Сохраняет: data/sentiment/cbr_data.csv

Запуск:
    python src/data/parse_cbr.py
    python src/data/parse_cbr.py --from 2018-01-01
"""
from __future__ import annotations

import argparse
import logging
import ssl
import urllib3
from datetime import date, datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import requests

# Отключаем SSL-проверку (macOS не всегда имеет нужные сертификаты)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
VERIFY_SSL = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "sentiment" / "cbr_data.csv"

# ЦБ РФ коды валют (VAL_NM_RQ в XML API)
CBR_CURRENCIES = {
    "USD": "R01235",
    "EUR": "R01239",
    "CNY": "R01375",  # Юань — актуально после 2022
}

CBR_DYNAMIC_URL = (
    "https://www.cbr.ru/scripts/XML_dynamic.asp"
    "?date_req1={date_from}&date_req2={date_to}&VAL_NM_RQ={val_code}"
)

CBR_KEYRATE_URL = "https://www.cbr.ru/hd_base/KeyRate/"


def fetch_currency(
    currency: str,
    date_from: str,
    date_to: str,
    timeout: int = 30,
) -> pd.Series:
    """
    Загружает дневные курсы валюты к рублю через XML API ЦБ РФ.
    date_from, date_to — формат YYYY-MM-DD.
    Возвращает Series с индексом date и значениями курса.
    """
    val_code = CBR_CURRENCIES.get(currency.upper())
    if not val_code:
        raise ValueError(f"Неизвестная валюта: {currency}. Доступны: {list(CBR_CURRENCIES)}")

    # ЦБ РФ принимает дату в формате DD/MM/YYYY
    fmt_from = datetime.strptime(date_from, "%Y-%m-%d").strftime("%d/%m/%Y")
    fmt_to   = datetime.strptime(date_to,   "%Y-%m-%d").strftime("%d/%m/%Y")

    url = CBR_DYNAMIC_URL.format(
        date_from=fmt_from, date_to=fmt_to, val_code=val_code
    )
    log.info("Загружаем %s/RUB: %s → %s", currency, date_from, date_to)

    try:
        resp = requests.get(url, timeout=timeout, verify=VERIFY_SSL,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        log.error("Ошибка запроса %s: %s", currency, e)
        return pd.Series(dtype=float, name=f"{currency.lower()}_rub")

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        log.error("Ошибка XML %s: %s", currency, e)
        return pd.Series(dtype=float, name=f"{currency.lower()}_rub")

    records = []
    for record in root.findall("Record"):
        dt_str = record.attrib.get("Date", "")
        val_str = (record.findtext("Value") or "").replace(",", ".")
        nominal_str = (record.findtext("Nominal") or "1").replace(",", ".")
        try:
            dt  = datetime.strptime(dt_str, "%d.%m.%Y").date()
            val = float(val_str) / float(nominal_str)
            records.append((dt, val))
        except (ValueError, ZeroDivisionError):
            continue

    if not records:
        log.warning("Нет данных для %s", currency)
        return pd.Series(dtype=float, name=f"{currency.lower()}_rub")

    series = pd.Series(
        {dt: val for dt, val in records},
        name=f"{currency.lower()}_rub",
        dtype=float,
    )
    series.index = pd.to_datetime(series.index)
    series.index.name = "date"
    log.info("  → %d наблюдений %s", len(series), currency)
    return series


def fetch_key_rate(timeout: int = 30) -> pd.Series:
    """
    Скрейпит историю ключевой ставки ЦБ РФ с cbr.ru.
    Возвращает Series: дата решения → ставка (%).
    """
    log.info("Загружаем ключевую ставку ЦБ РФ...")
    try:
        # Сначала скачиваем HTML через requests (с отключённым SSL),
        # затем парсим pandas
        resp = requests.get(CBR_KEYRATE_URL, timeout=30, verify=VERIFY_SSL,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        from io import StringIO
        tables = pd.read_html(
            StringIO(resp.text),
            thousands=" ",
            decimal=",",
            flavor="lxml",
        )
    except Exception as e:
        log.error("Ошибка загрузки ключевой ставки: %s", e)
        return pd.Series(dtype=float, name="key_rate")

    # Находим нужную таблицу — ищем по наличию колонок с датой и ставкой
    target = None
    for tbl in tables:
        cols_lower = [str(c).lower() for c in tbl.columns]
        if any("дата" in c or "date" in c for c in cols_lower):
            target = tbl
            break

    if target is None and tables:
        target = tables[0]

    if target is None:
        log.warning("Таблица ключевой ставки не найдена")
        return pd.Series(dtype=float, name="key_rate")

    log.info("  Таблица: %d строк, колонки: %s", len(target), list(target.columns))

    # Парсим первые две колонки: дата и ставка
    records = []
    for _, row in target.iterrows():
        vals = list(row)
        if len(vals) < 2:
            continue
        try:
            # Дата может быть в разных форматах
            dt_raw = str(vals[0]).strip()
            for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
                try:
                    dt = datetime.strptime(dt_raw, fmt).date()
                    break
                except ValueError:
                    continue
            else:
                continue

            rate_str = str(vals[1]).replace(",", ".").replace("%", "").strip()
            rate = float(rate_str)
            records.append((dt, rate))
        except (ValueError, IndexError):
            continue

    if not records:
        log.warning("Не удалось распарсить строки ключевой ставки")
        return pd.Series(dtype=float, name="key_rate")

    series = pd.Series(
        {dt: rate for dt, rate in records},
        name="key_rate",
        dtype=float,
    ).sort_index()
    series.index = pd.to_datetime(series.index)
    series.index.name = "date"
    log.info("  → %d наблюдений ключевой ставки (%.1f%% → %.1f%%)",
             len(series), series.iloc[0], series.iloc[-1])
    return series


def build_daily_cbr(
    date_from: str = "2018-01-01",
    date_to: str | None = None,
) -> pd.DataFrame:
    """
    Собирает дневной DataFrame с курсами валют и ключевой ставкой.
    Ключевая ставка forward-fill (действует до следующего решения).
    Добавляет производные признаки:
      - usd_rub_ret    : дневная лог-доходность USD/RUB
      - key_rate_change: 1 если ставка изменилась в этот день, иначе 0
    """
    if date_to is None:
        date_to = date.today().strftime("%Y-%m-%d")

    # Генерируем полный дневной индекс
    idx = pd.date_range(start=date_from, end=date_to, freq="D", name="date")

    # Курсы валют
    usd = fetch_currency("USD", date_from, date_to)
    eur = fetch_currency("EUR", date_from, date_to)
    cny = fetch_currency("CNY", date_from, date_to)

    # Ключевая ставка
    key_rate = fetch_key_rate()
    # Обрезаем по нашему диапазону
    key_rate = key_rate[
        (key_rate.index >= pd.Timestamp(date_from)) &
        (key_rate.index <= pd.Timestamp(date_to))
    ]

    # Собираем на дневном индексе
    df = pd.DataFrame(index=idx)
    df = df.join(usd,      how="left")
    df = df.join(eur,      how="left")
    df = df.join(cny,      how="left")

    # Курсы — forward fill (в выходные официального курса нет)
    for col in ["usd_rub", "eur_rub", "cny_rub"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Ключевая ставка — точечные события → forward fill
    if not key_rate.empty and hasattr(key_rate.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(key_rate.index):
        # Обрезаем по диапазону только если индекс DatetimeIndex
        try:
            key_rate = key_rate.loc[
                (key_rate.index >= pd.Timestamp(date_from)) &
                (key_rate.index <= pd.Timestamp(date_to))
            ]
        except Exception:
            pass
    df = df.join(key_rate, how="left")
    if "key_rate" in df.columns:
        df["key_rate_change"] = df["key_rate"].notna().astype(int)
        df["key_rate"] = df["key_rate"].ffill()
    else:
        df["key_rate"] = float("nan")
        df["key_rate_change"] = 0

    # Лог-доходности курсов
    if "usd_rub" in df.columns:
        import numpy as np
        df["usd_rub_ret"] = np.log(df["usd_rub"] / df["usd_rub"].shift(1))
        df["eur_rub_ret"] = np.log(df["eur_rub"] / df["eur_rub"].shift(1))

    df.index.name = "date"
    log.info("CBR данные: %d дней, %d колонок", len(df), len(df.columns))
    return df


def save(df: pd.DataFrame, path: Path = OUTPUT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = df.combine_first(existing).sort_index()

    df.to_csv(path)
    log.info("Сохранено → %s (%d строк)", path, len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Парсер данных ЦБ РФ")
    parser.add_argument("--from", dest="start", default="2018-01-01",
                        help="Начальная дата YYYY-MM-DD")
    parser.add_argument("--to", dest="end", default=None,
                        help="Конечная дата YYYY-MM-DD")
    args = parser.parse_args()

    df = build_daily_cbr(args.start, args.end)

    print(f"\nПервые строки:\n{df.head()}")
    print(f"\nОписательная статистика:")
    print(df[["usd_rub", "eur_rub", "key_rate"]].describe().round(2))

    save(df)


if __name__ == "__main__":
    main()

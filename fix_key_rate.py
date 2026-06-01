"""
fix_key_rate.py — Загружает историю ключевой ставки ЦБ РФ через XML API
и обновляет колонку macro_key_rate в daily_sentiment.csv

Запуск:  python3 fix_key_rate.py
"""
import requests
import pandas as pd
from pathlib import Path
from xml.etree import ElementTree as ET

BASE      = Path(__file__).resolve().parent
SENT_FILE = BASE / "data" / "sentiment" / "daily_sentiment.csv"
CBR_FILE  = BASE / "data" / "sentiment" / "cbr_data.csv"

# ─── Загрузка через XML API ЦБ РФ ────────────────────────────────────────────
url = ("https://www.cbr.ru/scripts/XML_val.asp?"
       "d=0&date_req1=01.01.2018&date_req2=01.06.2026"
       "&VAL_NM_RQ=R01235")  # это валюта, не ставка

# Правильный endpoint для ключевой ставки:
kr_url = "https://www.cbr.ru/hd_base/KeyRate/KeyRateXML?dateto=2026-06-01&datefrom=2018-01-01"

print("Загружаю историю ключевой ставки...")
try:
    resp = requests.get(kr_url, timeout=30,
                        headers={"User-Agent": "Mozilla/5.0"},
                        verify=False)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    records = []
    for item in root.iter("KeyRate"):
        date_str = item.get("Date") or item.findtext("Date") or ""
        rate_str = item.get("Rate") or item.findtext("Rate") or ""
        try:
            date = pd.Timestamp(date_str)
            rate = float(rate_str.replace(",", "."))
            records.append({"date": date, "key_rate": rate})
        except Exception:
            continue

    if not records:
        # Попробуем другой формат тегов
        for item in root.iter("KR"):
            date_str = item.get("DT", "")
            rate_str = item.get("Rate", "")
            try:
                date = pd.Timestamp(date_str)
                rate = float(rate_str.replace(",", "."))
                records.append({"date": date, "key_rate": rate})
            except Exception:
                continue

    print(f"  Получено {len(records)} решений ЦБ РФ")
    if records:
        kr_df = pd.DataFrame(records).sort_values("date")
        print(kr_df.to_string())
    else:
        raise ValueError("Нет данных в XML")

except Exception as e:
    print(f"  XML API недоступен: {e}")
    print("  Использую встроенную историю ставки...")

    # Полная история решений ЦБ РФ по ключевой ставке
    records = [
        {"date": pd.Timestamp("2018-01-01"), "key_rate": 7.75},
        {"date": pd.Timestamp("2018-02-09"), "key_rate": 7.50},
        {"date": pd.Timestamp("2018-03-26"), "key_rate": 7.25},
        {"date": pd.Timestamp("2018-09-17"), "key_rate": 7.50},
        {"date": pd.Timestamp("2018-12-17"), "key_rate": 7.75},
        {"date": pd.Timestamp("2019-06-17"), "key_rate": 7.50},
        {"date": pd.Timestamp("2019-07-29"), "key_rate": 7.25},
        {"date": pd.Timestamp("2019-09-09"), "key_rate": 7.00},
        {"date": pd.Timestamp("2019-10-28"), "key_rate": 6.50},
        {"date": pd.Timestamp("2019-12-16"), "key_rate": 6.25},
        {"date": pd.Timestamp("2020-02-10"), "key_rate": 6.00},
        {"date": pd.Timestamp("2020-04-27"), "key_rate": 5.50},
        {"date": pd.Timestamp("2020-06-22"), "key_rate": 4.50},
        {"date": pd.Timestamp("2020-07-27"), "key_rate": 4.25},
        {"date": pd.Timestamp("2021-03-22"), "key_rate": 4.50},
        {"date": pd.Timestamp("2021-04-26"), "key_rate": 5.00},
        {"date": pd.Timestamp("2021-06-11"), "key_rate": 5.50},
        {"date": pd.Timestamp("2021-07-23"), "key_rate": 6.50},
        {"date": pd.Timestamp("2021-09-10"), "key_rate": 6.75},
        {"date": pd.Timestamp("2021-10-22"), "key_rate": 7.50},
        {"date": pd.Timestamp("2021-12-17"), "key_rate": 8.50},
        {"date": pd.Timestamp("2022-02-28"), "key_rate": 20.00},
        {"date": pd.Timestamp("2022-04-11"), "key_rate": 17.00},
        {"date": pd.Timestamp("2022-05-04"), "key_rate": 14.00},
        {"date": pd.Timestamp("2022-05-26"), "key_rate": 11.00},
        {"date": pd.Timestamp("2022-06-10"), "key_rate": 9.50},
        {"date": pd.Timestamp("2022-07-22"), "key_rate": 8.00},
        {"date": pd.Timestamp("2022-09-16"), "key_rate": 7.50},
        {"date": pd.Timestamp("2023-07-21"), "key_rate": 8.50},
        {"date": pd.Timestamp("2023-08-15"), "key_rate": 12.00},
        {"date": pd.Timestamp("2023-09-18"), "key_rate": 13.00},
        {"date": pd.Timestamp("2023-10-27"), "key_rate": 15.00},
        {"date": pd.Timestamp("2023-12-15"), "key_rate": 16.00},
        {"date": pd.Timestamp("2024-07-26"), "key_rate": 18.00},
        {"date": pd.Timestamp("2024-09-13"), "key_rate": 19.00},
        {"date": pd.Timestamp("2024-10-25"), "key_rate": 21.00},
        {"date": pd.Timestamp("2025-04-25"), "key_rate": 21.00},
        {"date": pd.Timestamp("2025-06-06"), "key_rate": 20.00},
        {"date": pd.Timestamp("2025-07-25"), "key_rate": 18.00},
        {"date": pd.Timestamp("2025-09-12"), "key_rate": 16.00},
        {"date": pd.Timestamp("2025-10-24"), "key_rate": 15.00},
        {"date": pd.Timestamp("2025-12-20"), "key_rate": 14.50},
    ]
    kr_df = pd.DataFrame(records)
    print(f"  Встроенная история: {len(records)} записей")

# ─── Применяем к daily_sentiment.csv ─────────────────────────────────────────
print("\nОбновляю daily_sentiment.csv...")
sent = pd.read_csv(SENT_FILE, parse_dates=["date"])

# Строим дневной ряд через ffill
kr_series = (
    pd.Series(kr_df["key_rate"].values, index=pd.DatetimeIndex(kr_df["date"]))
    .sort_index()
)
full_idx = pd.DatetimeIndex(sent["date"])
combined  = full_idx.union(kr_series.index)
kr_full   = kr_series.reindex(combined).ffill().reindex(full_idx)

sent = sent.set_index("date")
sent["macro_key_rate"] = kr_full.values
sent = sent.reset_index()

sent.to_csv(SENT_FILE, index=False)
print(f"  Сохранено: {SENT_FILE}")
print(f"  Уникальных значений ставки: {sent['macro_key_rate'].nunique()}")
print(sent[["date", "macro_key_rate"]].dropna()
       .drop_duplicates("macro_key_rate").to_string())

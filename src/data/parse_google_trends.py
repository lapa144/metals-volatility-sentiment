"""
parse_google_trends.py
======================
Парсер Google Trends — прокси "внимания" инвесторов к металлам.

Логика:
  - Для периодов > 90 дней Google возвращает недельные данные → интерполируем на дни
  - Нормализуем каждый запрос в диапазон [0, 1]
  - Итоговый attention_google = средневзвешенное по всем запросам

Запросы:
  Металлы:  "золото цена", "купить золото", "серебро цена", "палладий"
  Макро:    "ключевая ставка", "инфляция россия", "курс доллара"
  Глобал:   "gold price", "silver price"

Сохраняет: data/sentiment/google_trends.csv

Запуск:
    python src/data/parse_google_trends.py
    python src/data/parse_google_trends.py --from 2020-01-01
    python src/data/parse_google_trends.py --from 2018-01-01 --geo RU

Зависимости:
    pip install pytrends pandas
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime
from pathlib import Path

import urllib3
import pandas as pd

# ── Патч совместимости pytrends с urllib3 >= 2.0 ────────────────────────────
# urllib3 2.0 переименовал method_whitelist → allowed_methods,
# pytrends ещё не обновился → патчим вручную
try:
    _original_retry_init = urllib3.util.retry.Retry.__init__

    def _patched_retry_init(self, *args, **kwargs):
        if "method_whitelist" in kwargs:
            kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
        _original_retry_init(self, *args, **kwargs)

    urllib3.util.retry.Retry.__init__ = _patched_retry_init
except Exception:
    pass

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "sentiment"
OUTPUT_FILE = OUTPUT_DIR / "google_trends.csv"

# ─── Группы запросов (по 5 — лимит pytrends) ───────────────────────────────
# Каждая группа запрашивается отдельно и нормируется относительно друг друга
QUERY_GROUPS: dict[str, list[str]] = {
    "metals_ru": [
        "золото цена",
        "купить золото",
        "серебро цена",
        "платина цена",
        "палладий",
    ],
    "macro_ru": [
        "ключевая ставка",
        "инфляция россия",
        "курс доллара",
        "центральный банк",
        "санкции",
    ],
    "metals_en": [
        "gold price",
        "silver price",
        "gold buy",
        "precious metals",
        "palladium price",
    ],
}

# Веса групп для итогового индекса
GROUP_WEIGHTS = {
    "metals_ru": 0.5,
    "macro_ru":  0.3,
    "metals_en": 0.2,
}


def fetch_group(
    queries: list[str],
    geo: str,
    start_date: str,
    end_date: str,
    retries: int = 3,
    sleep: float = 10.0,
) -> pd.DataFrame | None:
    """
    Запрашивает одну группу из ≤5 запросов через pytrends.
    Возвращает DataFrame с дневными (или недельными) индексами.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        raise ImportError("pip install pytrends")

    timeframe = f"{start_date} {end_date}"

    for attempt in range(retries):
        try:
            pt = TrendReq(hl="ru-RU", tz=180, timeout=(10, 30), retries=2,
                          requests_args={"verify": False})
            pt.build_payload(queries, cat=0, timeframe=timeframe, geo=geo, gprop="")
            df = pt.interest_over_time()
            if df.empty:
                log.warning("Пустой ответ для %s", queries)
                return None
            df = df.drop(columns=["isPartial"], errors="ignore")
            log.info("Получено %d строк для %s", len(df), queries[:2])
            return df
        except Exception as e:
            log.warning("Попытка %d/%d, ошибка: %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(sleep * (attempt + 1))

    return None


def interpolate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Если данные недельные — интерполируем на дневные.
    Определяем по минимальному шагу между датами.
    """
    if df.empty:
        return df

    freq = pd.infer_freq(df.index)
    if freq and freq.startswith("W"):
        # Недельные данные → ресемплируем на дни с линейной интерполяцией
        df = df.resample("D").interpolate(method="linear")
        log.info("Интерполировано с недельных на дневные: %d строк", len(df))
    elif freq and freq.startswith("D"):
        pass  # уже дневные
    else:
        # Неизвестная частота — пробуем ресемплировать
        try:
            df = df.resample("D").interpolate(method="linear")
        except Exception:
            pass

    return df


def normalize_group(df: pd.DataFrame) -> pd.Series:
    """
    Нормализует группу запросов в [0,1] и возвращает среднее по группе.
    Нормализация: делим на max(100) — гугл итак даёт 0-100.
    """
    normed = df / 100.0
    return normed.mean(axis=1).rename("score")


def fetch_all_groups(
    geo: str = "RU",
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    pause: float = 15.0,
) -> pd.DataFrame:
    """
    Запрашивает все группы запросов и собирает взвешенный индекс.
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    group_series: dict[str, pd.Series] = {}

    for group_name, queries in QUERY_GROUPS.items():
        log.info("Запрашиваем группу '%s'...", group_name)
        df = fetch_group(queries, geo, start_date, end_date)

        if df is None or df.empty:
            log.warning("Группа '%s' пропущена.", group_name)
            continue

        df = interpolate_to_daily(df)
        group_series[group_name] = normalize_group(df)

        # Пауза между запросами — Google блокирует при частых запросах
        log.info("Пауза %.0f сек...", pause)
        time.sleep(pause)

    if not group_series:
        log.error("Не удалось получить данные ни из одной группы.")
        return pd.DataFrame()

    # Собираем взвешенный индекс
    frames = []
    total_weight = 0.0
    for gname, series in group_series.items():
        w = GROUP_WEIGHTS.get(gname, 1.0 / len(group_series))
        frames.append(series.rename(gname) * w)
        total_weight += w

    combined = pd.concat(frames, axis=1).sum(axis=1) / total_weight
    combined.name = "attention_google"
    combined.index.name = "date"

    # Итоговый DataFrame с детализацией по группам
    detail = pd.concat(
        [s.rename(k) for k, s in group_series.items()], axis=1
    )
    detail["attention_google"] = combined
    detail.index = pd.to_datetime(detail.index).normalize()
    detail.index.name = "date"

    return detail


def save(df: pd.DataFrame, path: Path = OUTPUT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = df.combine_first(existing)
        df = df.sort_index()

    df.to_csv(path)
    log.info("Сохранено %d строк → %s", len(df), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Парсер Google Trends для металлов")
    parser.add_argument("--from", dest="start", default="2018-01-01",
                        help="Начальная дата (YYYY-MM-DD)")
    parser.add_argument("--to", dest="end", default=None,
                        help="Конечная дата (YYYY-MM-DD), по умолчанию сегодня")
    parser.add_argument("--geo", default="RU",
                        help="Регион для Google Trends (RU, US, '' = мировой)")
    parser.add_argument("--pause", type=float, default=15.0,
                        help="Пауза между группами запросов (сек)")
    args = parser.parse_args()

    log.info("Запуск Google Trends: %s → %s, geo=%s",
             args.start, args.end or "сегодня", args.geo)

    df = fetch_all_groups(
        geo=args.geo,
        start_date=args.start,
        end_date=args.end,
        pause=args.pause,
    )

    if df.empty:
        log.error("Данные не получены.")
        return

    save(df)
    print(f"\nПервые строки:\n{df.head()}")
    print(f"\nКолонки: {list(df.columns)}")
    print(f"Диапазон: {df.index.min().date()} — {df.index.max().date()}")


if __name__ == "__main__":
    main()

"""
compute_rv.py
=============
Расчёт реализованной волатильности из дневных OHLC данных MOEX.

Используемые оценщики (range-based, эффективнее close-to-close):
  - Parkinson (1980)      — использует High/Low
  - Garman-Klass (1980)   — использует OHLC
  - Rogers-Satchell (1991)— не требует overnight-gap коррекции
  - Close-to-Close        — классический (log-return²), бенчмарк

После расчёта RV формируются HAR-лаги:
  RV_d  = RV_{t}          (дневная)
  RV_w  = mean(RV_{t-4}..RV_{t})   (недельная, 5 торговых дней)
  RV_m  = mean(RV_{t-21}..RV_{t})  (месячная, 22 торговых дня)

Выход: data/processed/rv_features.csv
"""

import os
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Пути
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PANEL = os.path.join(BASE_DIR, "data", "processed", "metals_1d_panel.csv")
OUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "rv_features.csv")


# ──────────────────────────────────────────────────────────────
# Оценщики волатильности
# ──────────────────────────────────────────────────────────────

def rv_close_to_close(df: pd.DataFrame) -> pd.Series:
    """
    Квадрат дневного лог-доходности (Close-to-Close).
    RV_t = (ln C_t - ln C_{t-1})²
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    return log_ret ** 2


def rv_parkinson(df: pd.DataFrame) -> pd.Series:
    """
    Parkinson (1980) оценщик.
    RV_t = 1 / (4 ln 2) * (ln H_t - ln L_t)²
    Эффективнее C2C в ~5 раз при геометрическом броуновском движении.
    """
    return (np.log(df["high"] / df["low"]) ** 2) / (4.0 * np.log(2))


def rv_garman_klass(df: pd.DataFrame) -> pd.Series:
    """
    Garman-Klass (1980) оценщик.
    RV_t = 0.5*(ln H/L)² - (2ln2 - 1)*(ln C/O)²
    Эффективнее C2C в ~7.4 раза.
    """
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    return 0.5 * hl - (2.0 * np.log(2) - 1.0) * co


def rv_rogers_satchell(df: pd.DataFrame) -> pd.Series:
    """
    Rogers-Satchell (1991) оценщик.
    RV_t = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
    Не зависит от overnight drift — подходит для биржевых данных.
    """
    u = np.log(df["high"] / df["open"])
    d = np.log(df["low"]  / df["open"])
    c = np.log(df["close"] / df["open"])
    return u * (u - c) + d * (d - c)


def annualize_rv(rv_series: pd.Series, trading_days: int = 252) -> pd.Series:
    """Перевод дневной дисперсии в годовую стандартную девиацию."""
    return np.sqrt(np.maximum(rv_series, 0) * trading_days)


# ──────────────────────────────────────────────────────────────
# HAR-лаги
# ──────────────────────────────────────────────────────────────

def make_har_features(rv: pd.Series, prefix: str = "rv") -> pd.DataFrame:
    """
    Строит DataFrame с HAR-лагами:
      {prefix}_d  — дневная RV (значение предыдущего дня, т.е. сдвиг на 1)
      {prefix}_w  — среднее за 5 предыдущих дней
      {prefix}_m  — среднее за 22 предыдущих дня
      {prefix}_target — целевая переменная (RV на следующий день)
    Все признаки сдвинуты так, чтобы предсказывать RV_{t+1}.
    """
    rv = rv.copy()
    df = pd.DataFrame(index=rv.index)

    # Лаги: shift(1) — значения за вчера
    df[f"{prefix}_d"] = rv.shift(1)
    df[f"{prefix}_w"] = rv.shift(1).rolling(window=5,  min_periods=3).mean()
    df[f"{prefix}_m"] = rv.shift(1).rolling(window=22, min_periods=10).mean()
    df[f"{prefix}_target"] = rv  # y = RV_t (предсказываем текущий день)

    return df


# ──────────────────────────────────────────────────────────────
# Основная функция
# ──────────────────────────────────────────────────────────────

def compute_rv_features(
    panel_path: str = RAW_PANEL,
    out_path: str   = OUT_PATH,
    estimator: str  = "garman_klass",
    log_rv: bool    = True,
) -> pd.DataFrame:
    """
    Загружает панель OHLC, считает RV выбранным оценщиком,
    строит HAR-лаги, сохраняет в CSV.

    Parameters
    ----------
    panel_path : путь к metals_1d_panel.csv
    out_path   : путь для сохранения rv_features.csv
    estimator  : "garman_klass" | "parkinson" | "rogers_satchell" | "close_to_close"
    log_rv     : если True — работаем с log(RV), иначе с уровнями (HAR-log vs HAR-level)

    Returns
    -------
    pd.DataFrame с колонками: secid, date, rv_raw, rv_ann, rv_d, rv_w, rv_m, rv_target
    """
    estimators = {
        "garman_klass"    : rv_garman_klass,
        "parkinson"       : rv_parkinson,
        "rogers_satchell" : rv_rogers_satchell,
        "close_to_close"  : rv_close_to_close,
    }
    if estimator not in estimators:
        raise ValueError(f"Неизвестный оценщик: {estimator}. Доступны: {list(estimators)}")

    panel = pd.read_csv(panel_path, parse_dates=["date"])
    panel = panel.sort_values(["secid", "date"]).reset_index(drop=True)

    results = []

    for secid, grp in panel.groupby("secid"):
        grp = grp.set_index("date").sort_index()

        # Убираем строки с нулевыми ценами (нет торгов)
        grp = grp[(grp["close"] > 0) & (grp["open"] > 0) &
                  (grp["high"] > 0)  & (grp["low"] > 0)]

        # Сырая RV (дисперсия, не аннуализированная)
        rv_raw = estimators[estimator](grp).rename("rv_raw")
        rv_raw = rv_raw.clip(lower=0)  # Garman-Klass может дать мини-отрицательные

        # Аннуализированная волатильность (std, %)
        rv_ann = annualize_rv(rv_raw).rename("rv_ann")

        # Log-трансформация (для HAR-log модели)
        if log_rv:
            rv_model = np.log(rv_raw.replace(0, np.nan)).rename("rv_log")
        else:
            rv_model = rv_raw.rename("rv_level")

        # HAR-лаги
        har = make_har_features(rv_model, prefix="rv")

        # Сборка
        out = pd.concat([rv_raw, rv_ann, har], axis=1)
        out.index.name = "date"
        out = out.reset_index()
        out.insert(0, "secid", secid)
        out["estimator"] = estimator
        out["log_rv"] = log_rv

        results.append(out)

    features = pd.concat(results, ignore_index=True)
    features = features.dropna(subset=["rv_d", "rv_w", "rv_m", "rv_target"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    features.to_csv(out_path, index=False)
    print(f"[compute_rv] Сохранено {len(features)} строк → {out_path}")
    print(features.groupby("secid")[["rv_raw", "rv_ann"]].describe().round(6))

    return features


# ──────────────────────────────────────────────────────────────
# Утилиты для последующих моделей
# ──────────────────────────────────────────────────────────────

def load_rv_features(
    path: str = OUT_PATH,
    secid: str | None = None,
) -> pd.DataFrame:
    """Загружает rv_features.csv, опционально фильтрует по тикеру."""
    df = pd.read_csv(path, parse_dates=["date"])
    if secid:
        df = df[df["secid"] == secid].reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = compute_rv_features()
    print("\nПервые 5 строк (GLDRUB_TOM):")
    print(df[df["secid"] == "GLDRUB_TOM"].head())

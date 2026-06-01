"""
metrics.py
==========
Единый модуль метрик качества прогнозов волатильности.
Используется всеми моделями (HAR, XGBoost, kNN) для корректного сравнения.

ВАЖНО: до унификации в har_model.py и knn_model.py были РАЗНЫЕ формулы QLIKE,
что делало попарное сравнение некорректным. Теперь все модели используют
каноническую формулу Patton (2011) через единый импорт.

Метрики:
    mae(y_true, y_pred)        — Mean Absolute Error
    rmse(y_true, y_pred)       — Root Mean Squared Error
    qlike(rv, sigma2)          — Patton (2011), РАБОТАЕТ В УРОВНЯХ дисперсии (не log)
    r2_oos(y_true, y_pred,
           y_bench=None)       — Goyal–Welch (2008), относительно среднего обучающей выборки
    compute_metrics(...)       — собирает все метрики разом; делает exp(), если is_log=True
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Базовые метрики
# ──────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def qlike(rv: np.ndarray, sigma2: np.ndarray, eps: float = 1e-10) -> float:
    """
    QLIKE (Patton 2011, eq. 14) — робастная к шуму прокси волатильности
    функция потерь, СИММЕТРИЧНАЯ относительно перестановки прогнозов:

        QLIKE = E[ RV/σ² − ln(RV/σ²) − 1 ]

    Принимает значения В УРОВНЯХ дисперсии (а не log).
    Минимум = 0 при σ² = RV.
    """
    rv = np.maximum(np.asarray(rv, dtype=float), eps)
    sigma2 = np.maximum(np.asarray(sigma2, dtype=float), eps)
    ratio = rv / sigma2
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def r2_oos(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_bench: Optional[np.ndarray] = None,
) -> float:
    """
    Out-of-sample R² (Goyal & Welch 2008):
        R²_OOS = 1 − MSE_model / MSE_benchmark

    По умолчанию бенчмарк = историческое среднее тестовой выборки
    (адаптированная формула). Положительный R²_OOS означает превосходство
    над наивным прогнозом средним.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_bench is None:
        y_bench = np.full_like(y_true, np.mean(y_true))
    mse_m = float(np.mean((y_true - y_pred) ** 2))
    mse_b = float(np.mean((y_true - y_bench) ** 2))
    return float(1.0 - mse_m / (mse_b + 1e-30))


# ──────────────────────────────────────────────────────────────
# Сводная функция
# ──────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_bench: Optional[np.ndarray] = None,
    is_log: bool = True,
) -> dict:
    """
    Возвращает словарь {MAE, RMSE, R2_OOS, QLIKE}.

    Если is_log=True (прогноз и факт в логарифмах),
    для QLIKE производится exp()-преобразование к уровням дисперсии.
    MAE/RMSE/R2_OOS считаются на тех же шкалах, что и переданы.
    """
    metrics = {
        "MAE":    mae(y_true, y_pred),
        "RMSE":   rmse(y_true, y_pred),
        "R2_OOS": r2_oos(y_true, y_pred, y_bench),
    }
    if is_log:
        rv_level    = np.exp(y_true)
        sigma_level = np.exp(y_pred)
    else:
        rv_level    = y_true
        sigma_level = y_pred
    metrics["QLIKE"] = qlike(rv_level, sigma_level)
    return metrics


# ──────────────────────────────────────────────────────────────
# Diebold–Mariano (1995) test — для статистической значимости
# различий между двумя моделями. Используется в src/models/dm_test.py.
# ──────────────────────────────────────────────────────────────

def _newey_west_variance(x: np.ndarray, lags: int) -> float:
    """
    HAC оценка долгосрочной дисперсии (Newey–West 1987).
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    gamma0 = float(np.dot(x, x) / n)
    s = gamma0
    for h in range(1, lags + 1):
        cov = float(np.dot(x[h:], x[:-h]) / n)
        w = 1.0 - h / (lags + 1.0)  # Bartlett kernel
        s += 2.0 * w * cov
    return s


def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    loss: str = "se",
    h: int = 1,
) -> dict:
    """
    Тест Диболда–Мариано (1995) на равенство ожидаемых потерь двух прогнозов.

    H0: E[L(e1)] = E[L(e2)]  ⇔  E[d_t] = 0,
    где d_t = L(e1_t) − L(e2_t).

    Parameters
    ----------
    e1, e2 : массивы ошибок прогноза (y_true − y_pred) двух моделей
    loss   : "se" (квадратичная), "ae" (абсолютная) или "qlike"
             ВАЖНО: для loss="qlike" e1, e2 должны быть КОРТЕЖАМИ (rv, sigma2)
                    в уровнях дисперсии, а не ошибками.
    h      : горизонт прогноза (для HAC: lags = h − 1; при h=1 → 0 лагов
             → используем 1 лаг по умолчанию для надёжности)

    Returns
    -------
    dict с ключами:
        DM        — статистика Диболда–Мариано
        p_value   — двусторонняя p-value по N(0,1)
        mean_d    — средняя разность потерь
        loss      — название функции потерь
        n         — число наблюдений
        verdict   — "model1 better" / "model2 better" / "no significant difference"
    """
    if loss == "qlike":
        rv1, sig2_1 = e1
        rv2, sig2_2 = e2
        rv1 = np.maximum(np.asarray(rv1, float), 1e-10)
        rv2 = np.maximum(np.asarray(rv2, float), 1e-10)
        sig2_1 = np.maximum(np.asarray(sig2_1, float), 1e-10)
        sig2_2 = np.maximum(np.asarray(sig2_2, float), 1e-10)
        L1 = rv1 / sig2_1 - np.log(rv1 / sig2_1) - 1.0
        L2 = rv2 / sig2_2 - np.log(rv2 / sig2_2) - 1.0
    else:
        e1 = np.asarray(e1, dtype=float)
        e2 = np.asarray(e2, dtype=float)
        if loss == "se":
            L1, L2 = e1 ** 2, e2 ** 2
        elif loss == "ae":
            L1, L2 = np.abs(e1), np.abs(e2)
        else:
            raise ValueError(f"Unknown loss: {loss}")

    d = L1 - L2
    n = len(d)
    mean_d = float(np.mean(d))

    lags = max(h - 1, 1)
    var_d = _newey_west_variance(d, lags=lags)

    if var_d <= 0:
        dm_stat = float("nan")
        p_val = float("nan")
    else:
        dm_stat = mean_d / np.sqrt(var_d / n)
        # Двусторонний p-value по нормальному распределению
        # (Harvey–Leybourne–Newbold поправка пропущена — мала при n>50)
        p_val = 2.0 * (1.0 - _normal_cdf(abs(dm_stat)))

    if np.isnan(dm_stat):
        verdict = "test failed"
    elif p_val > 0.10:
        verdict = "no significant difference"
    elif mean_d < 0:
        verdict = "model1 better"
    else:
        verdict = "model2 better"

    return {
        "DM":      round(float(dm_stat), 4),
        "p_value": round(float(p_val), 4),
        "mean_d":  round(float(mean_d), 6),
        "loss":    loss,
        "n":       int(n),
        "verdict": verdict,
    }


def _normal_cdf(x: float) -> float:
    """Стандартная нормальная CDF через erf (без scipy)."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

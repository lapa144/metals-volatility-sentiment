"""
har_model.py
============
HAR-RV и HAR-RV-Sentiment модели (Corsi 2009 + расширения).

Спецификации:
  HAR-level : RV_{t}   = β₀ + β₁·RV_d + β₂·RV_w + β₃·RV_m + ε
  HAR-log   : ln RV_{t} = β₀ + β₁·ln RV_d + β₂·ln RV_w + β₃·ln RV_m + ε
  HAR-S     : HAR-log + β₄·S_{t-1} + β₅·S̄^{(w)}_{t-1}
              (S — sentiment индекс, ОТЛАГИРОВАН на 1 день во избежание leakage)

Оценка:
  - OLS (statsmodels, если доступен; иначе numpy.linalg.lstsq fallback)
  - Expanding-window OOS прогноз (по умолчанию: обучение 60%, тест 40%)

Метрики берутся из единого модуля src.models.metrics (унифицированная
формула QLIKE по Patton 2011 для корректного сравнения с другими моделями).

Использование:
  from src.models.har_model import HARModel
  m = HARModel(secid="GLDRUB_TOM", spec="log_sentiment",
               sentiment_col="sentiment_combined")
  m.fit_rolling()
  print(m.metrics())
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Literal, Optional
from dataclasses import dataclass, field

from .metrics import compute_metrics

warnings.filterwarnings("ignore", category=FutureWarning)

# statsmodels доступен не во всех окружениях (sandbox без pip).
# При его отсутствии используем чистый numpy.linalg.lstsq для OLS.
try:
    import statsmodels.api as sm  # type: ignore
    _HAS_STATSMODELS = True
except ImportError:
    sm = None
    _HAS_STATSMODELS = False

# ──────────────────────────────────────────────────────────────
# Пути
# ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RV_FEATURES   = os.path.join(BASE_DIR, "data", "processed", "rv_features.csv")
SENTIMENT_CSV = os.path.join(BASE_DIR, "data", "sentiment", "daily_sentiment.csv")
RESULTS_DIR   = os.path.join(BASE_DIR, "data", "processed", "har_results")


# ──────────────────────────────────────────────────────────────
# OLS: реализация с fallback на numpy.linalg.lstsq
# ──────────────────────────────────────────────────────────────

class _OLSResult:
    """Минимальный результат OLS, совместимый по интерфейсу с statsmodels."""

    def __init__(self, beta: np.ndarray, x_names: list[str]):
        self.beta = beta
        self.params = pd.Series(beta, index=x_names)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X.values @ self.beta


def _fit_ols(X: pd.DataFrame, y: pd.Series, use_hac: bool = False):
    """
    OLS-фит. Если доступен statsmodels — использует его (с опциональным HAC),
    иначе чистый numpy.linalg.lstsq.
    """
    if _HAS_STATSMODELS:
        if use_hac:
            return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        return sm.OLS(y, X).fit()
    # numpy fallback
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    return _OLSResult(beta, list(X.columns))


# ──────────────────────────────────────────────────────────────
# Загрузка данных
# ──────────────────────────────────────────────────────────────

# Колонки sentiment, которые могут использоваться в HAR-S.
# По умолчанию используется sentiment_combined (взвешенный RSS+GDELT+Telegram),
# как описано в paper/chapters/04_architecture.tex, eq. (4.1).
DEFAULT_SENTIMENT_COL = "sentiment_combined"


def load_data(
    secid: str,
    rv_path: str  = RV_FEATURES,
    sent_path: str = SENTIMENT_CSV,
    sentiment_col: str = DEFAULT_SENTIMENT_COL,
) -> pd.DataFrame:
    """
    Загружает rv_features.csv, опционально джойнит sentiment.
    Возвращает DataFrame с датой в индексе.

    ВАЖНО: sentiment СДВИГАЕТСЯ НА 1 ДЕНЬ ВПЕРЁД во избежание leakage —
    для прогноза RV_t используется sentiment, известный к концу дня t-1.
    """
    rv = pd.read_csv(rv_path, parse_dates=["date"])
    rv = rv[rv["secid"] == secid].set_index("date").sort_index()

    if os.path.exists(sent_path):
        sent = pd.read_csv(sent_path, parse_dates=["date"]).set_index("date")

        if sentiment_col not in sent.columns:
            warnings.warn(
                f"Колонка '{sentiment_col}' не найдена в {sent_path}. "
                f"Доступные: {list(sent.columns)[:5]}... "
                f"Использую нулевой sentiment.",
                RuntimeWarning,
            )
            rv["sentiment_score"] = 0.0
            rv["sentiment_w"]     = 0.0
        else:
            # КРИТИЧНО: shift(1) — sentiment_{t-1} для предсказания RV_t
            s = sent[[sentiment_col]].shift(1).rename(
                columns={sentiment_col: "sentiment_score"}
            )
            rv = rv.join(s, how="left")
            rv["sentiment_score"] = rv["sentiment_score"].fillna(0.0)
            # Недельное скользящее sentiment_w (тоже из уже отлагированного ряда)
            rv["sentiment_w"] = rv["sentiment_score"].rolling(5, min_periods=1).mean()
    else:
        rv["sentiment_score"] = 0.0
        rv["sentiment_w"]     = 0.0

    return rv.dropna(subset=["rv_d", "rv_w", "rv_m", "rv_target"])


# ──────────────────────────────────────────────────────────────
# HAR модель
# ──────────────────────────────────────────────────────────────

SPECS = Literal["level", "log", "log_sentiment"]


@dataclass
class HARModel:
    """
    HAR-RV модель с rolling-window out-of-sample оценкой.

    Parameters
    ----------
    secid      : тикер (GLDRUB_TOM, SLVRUB_TOM, ...)
    spec       : "level" | "log" | "log_sentiment"
    train_frac : доля обучающей выборки (0.6 → 60%)
    min_train  : минимальный размер окна обучения
    rolling    : True — expanding window; False — static split
    """
    secid         : str   = "GLDRUB_TOM"
    spec          : SPECS = "log"
    train_frac    : float = 0.6
    min_train     : int   = 120
    rolling       : bool  = True
    sentiment_col : str   = DEFAULT_SENTIMENT_COL  # источник sentiment для HAR-S

    _data      : pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    _preds     : pd.Series    = field(default_factory=pd.Series,    repr=False)
    _actuals   : pd.Series    = field(default_factory=pd.Series,    repr=False)
    _coefs     : pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    @staticmethod
    def _add_constant(df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет колонку 'const' = 1 (statsmodels-совместимо, но без statsmodels)."""
        out = df.copy()
        out.insert(0, "const", 1.0)
        return out

    def _get_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Формирует матрицу признаков X (с константой) и целевую y."""
        base_features = ["rv_d", "rv_w", "rv_m"]

        if self.spec == "log_sentiment":
            features = base_features + ["sentiment_score", "sentiment_w"]
        else:
            features = base_features

        X = self._add_constant(df[features])
        y = df["rv_target"]
        return X, y

    def fit_rolling(
        self,
        rv_path: str   = RV_FEATURES,
        sent_path: str = SENTIMENT_CSV,
    ) -> "HARModel":
        """
        Expanding-window (или static-split) OOS прогноз.
        На каждом шаге обучает OLS на истории, предсказывает следующий день.
        """
        data = load_data(self.secid, rv_path, sent_path,
                         sentiment_col=self.sentiment_col)
        self._data = data.copy()

        n = len(data)
        train_end = max(self.min_train, int(n * self.train_frac))

        preds   = []
        actuals = []
        dates   = []
        coef_list = []

        for t in range(train_end, n):
            if self.rolling:
                train = data.iloc[:t]
            else:
                train = data.iloc[:train_end]

            test_row = data.iloc[[t]]

            X_train, y_train = self._get_xy(train)
            X_test,  y_test  = self._get_xy(test_row)

            try:
                ols = _fit_ols(X_train, y_train, use_hac=False)
                pred = float(np.asarray(ols.predict(X_test))[0])
            except Exception:
                ols = None
                pred = float(y_train.mean())

            preds.append(pred)
            actuals.append(float(y_test.iloc[0]))
            dates.append(data.index[t])

            if t == train_end and ols is not None:
                coef_list.append(ols.params.rename(data.index[t]))

        self._preds   = pd.Series(preds,   index=dates, name="predicted")
        self._actuals = pd.Series(actuals, index=dates, name="actual")

        if coef_list:
            self._coefs = pd.DataFrame(coef_list)

        print(f"[HAR] {self.secid} | spec={self.spec} | sent={self.sentiment_col} "
              f"| test n={len(preds)}")
        return self

    def fit_full(
        self,
        rv_path: str   = RV_FEATURES,
        sent_path: str = SENTIMENT_CSV,
    ):
        """
        Оценка OLS на ВСЕЙ выборке (для анализа коэффициентов).
        Если доступен statsmodels — возвращает результат с .summary() и HAC.
        Иначе возвращает _OLSResult (только .params и .predict()).
        """
        data = load_data(self.secid, rv_path, sent_path,
                         sentiment_col=self.sentiment_col)
        X, y = self._get_xy(data)
        return _fit_ols(X, y, use_hac=True)

    def metrics(self) -> dict:
        """Возвращает словарь метрик OOS."""
        if self._preds.empty:
            raise RuntimeError("Сначала вызови fit_rolling()")
        y_true = self._actuals.values
        y_pred = self._preds.values
        is_log = self.spec in ("log", "log_sentiment")
        return compute_metrics(y_true, y_pred, is_log=is_log)

    def predictions(self) -> pd.DataFrame:
        """DataFrame с фактическими и предсказанными значениями."""
        return pd.DataFrame({
            "actual":    self._actuals,
            "predicted": self._preds,
        })

    def save_results(self, out_dir: str = RESULTS_DIR) -> str:
        """Сохраняет прогнозы и метрики в CSV."""
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{self.secid}_{self.spec}_predictions.csv"
        path  = os.path.join(out_dir, fname)
        self.predictions().to_csv(path)

        mfname = f"{self.secid}_{self.spec}_metrics.csv"
        mpath  = os.path.join(out_dir, mfname)
        pd.Series(self.metrics(), name="value").to_csv(mpath)

        print(f"[HAR] Сохранено: {path}, {mpath}")
        return path


# ──────────────────────────────────────────────────────────────
# Удобная функция для запуска нескольких тикеров сразу
# ──────────────────────────────────────────────────────────────

def run_all_har(
    secids: list[str] = ("GLDRUB_TOM", "SLVRUB_TOM"),
    specs: list[SPECS] = ("level", "log", "log_sentiment"),
    sentiment_col: str = DEFAULT_SENTIMENT_COL,
    rv_path: str   = RV_FEATURES,
    sent_path: str = SENTIMENT_CSV,
    save: bool = True,
) -> pd.DataFrame:
    """
    Запускает HAR для всех комбинаций secid × spec.
    Для log_sentiment используется указанный sentiment_col
    (по умолчанию — взвешенный sentiment_combined).
    Возвращает сводную таблицу метрик.
    """
    rows = []
    for secid in secids:
        for spec in specs:
            try:
                m = HARModel(secid=secid, spec=spec,
                             sentiment_col=sentiment_col)
                m.fit_rolling(rv_path, sent_path)
                met = m.metrics()
                met.update({"secid": secid, "spec": spec,
                            "sentiment_col": sentiment_col if spec == "log_sentiment" else "-"})
                rows.append(met)
                if save:
                    m.save_results()
            except Exception as e:
                print(f"[HAR] ОШИБКА {secid} {spec}: {e}")

    summary = pd.DataFrame(rows).set_index(["secid", "spec"])
    print("\n=== Сводная таблица метрик HAR ===")
    print(summary.round(6))

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        summary.to_csv(os.path.join(RESULTS_DIR, "har_summary.csv"))

    return summary


def run_har_sentiment_ablation(
    secid: str = "GLDRUB_TOM",
    sentiment_cols: list[str] = (
        "sentiment_combined",  # RSS + GDELT + Telegram (взвешенный)
        "sentiment_score",     # только RSS
        "sentiment_gdelt",     # только GDELT
        "tg_score",            # только Telegram
        "attention_google",    # Google Trends attention
    ),
    rv_path: str   = RV_FEATURES,
    sent_path: str = SENTIMENT_CSV,
    save: bool = True,
) -> pd.DataFrame:
    """
    Ablation study HAR-log-S: сравнение разных sentiment-источников.
    Базовая модель HAR-log (без sentiment) включается для сравнения.
    """
    rows = []

    # baseline: HAR-log без sentiment
    m0 = HARModel(secid=secid, spec="log").fit_rolling(rv_path, sent_path)
    base = m0.metrics()
    base.update({"secid": secid, "sentiment_source": "none (HAR-log baseline)"})
    rows.append(base)

    # каждое HAR-log-S с разным sentiment-источником
    for s_col in sentiment_cols:
        try:
            m = HARModel(secid=secid, spec="log_sentiment",
                         sentiment_col=s_col).fit_rolling(rv_path, sent_path)
            met = m.metrics()
            met.update({"secid": secid, "sentiment_source": s_col})
            rows.append(met)
        except Exception as e:
            print(f"[HAR-ablation] ОШИБКА {secid} {s_col}: {e}")

    df = pd.DataFrame(rows)
    print(f"\n=== HAR-S Ablation: {secid} ===")
    print(df.round(6).to_string(index=False))

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(os.path.join(RESULTS_DIR, f"{secid}_har_ablation.csv"),
                  index=False)
    return df


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Быстрый тест на золоте
    m = HARModel(secid="GLDRUB_TOM", spec="log")
    m.fit_rolling()
    print("\nМетрики (HAR-log, GLDRUB_TOM):")
    for k, v in m.metrics().items():
        print(f"  {k}: {v:.6f}")

    print("\nОценка на полной выборке:")
    res = m.fit_full()
    print(res.summary())

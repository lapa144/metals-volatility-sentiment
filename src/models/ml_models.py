"""
ml_models.py
============
XGBoost и LSTM модели прогнозирования волатильности.

Оба класса реализуют единый интерфейс:
  .fit(X_train, y_train)
  .predict(X_test) → np.ndarray
  .evaluate(X_test, y_test) → dict метрик

Используются HAR-признаки из compute_rv.py + sentiment из daily_sentiment.csv.

Запуск:
  python src/models/ml_models.py
  → сохраняет результаты в data/processed/ml_results/
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Literal

from .metrics import compute_metrics  # унифицированный модуль метрик

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Пути
# ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RV_FEATURES   = os.path.join(BASE_DIR, "data", "processed", "rv_features.csv")
SENTIMENT_CSV = os.path.join(BASE_DIR, "data", "sentiment", "daily_sentiment.csv")
RESULTS_DIR   = os.path.join(BASE_DIR, "data", "processed", "ml_results")


# ──────────────────────────────────────────────────────────────
# Загрузка и подготовка признаков
# ──────────────────────────────────────────────────────────────

FEATURE_COLS = ["rv_d", "rv_w", "rv_m"]


def load_features(
    secid: str,
    rv_path: str   = RV_FEATURES,
    sent_path: str = SENTIMENT_CSV,
    add_lags: int  = 5,
    sentiment_cols: tuple[str, ...] = (
        "sentiment_combined",  # основной взвешенный индекс
        "sentiment_gdelt",
        "tg_score",
        "attention_google",
    ),
) -> pd.DataFrame:
    """
    Загружает признаки для ML моделей.

    Добавляет:
      • дополнительные лаги RV (rv_lag_2 .. rv_lag_N),
      • sentiment-вектор (если доступен в daily_sentiment.csv),
        ОТЛАГИРОВАННЫЙ на 1 день во избежание data leakage.
    """
    rv = pd.read_csv(rv_path, parse_dates=["date"])
    rv = rv[rv["secid"] == secid].set_index("date").sort_index()

    # Дополнительные лаги rv_raw
    for lag in range(2, add_lags + 1):
        rv[f"rv_lag_{lag}"] = rv["rv_d"].shift(lag - 1)

    # Sentiment с лагом на 1 день
    sent_used: list[str] = []
    if os.path.exists(sent_path):
        sent = pd.read_csv(sent_path, parse_dates=["date"]).set_index("date").sort_index()

        present = [c for c in sentiment_cols if c in sent.columns]
        if present:
            # КРИТИЧНО: shift(1) — sentiment_{t-1} для прогноза RV_t
            sent_lag = sent[present].shift(1)
            rv = rv.join(sent_lag, how="left")
            for c in present:
                rv[c] = rv[c].fillna(0.0)
            sent_used = list(present)

            # Скользящие средние по основному sentiment-индексу
            if "sentiment_combined" in present:
                rv["sentiment_w"] = rv["sentiment_combined"].rolling(5,  min_periods=1).mean()
                rv["sentiment_m"] = rv["sentiment_combined"].rolling(22, min_periods=5).mean()
                sent_used += ["sentiment_w", "sentiment_m"]
    if not sent_used:
        rv["sentiment_combined"] = 0.0
        rv["sentiment_w"]        = 0.0
        rv["sentiment_m"]        = 0.0
        sent_used = ["sentiment_combined", "sentiment_w", "sentiment_m"]

    extra_cols = [f"rv_lag_{i}" for i in range(2, add_lags + 1)]
    all_feat   = FEATURE_COLS + extra_cols + sent_used

    rv = rv.dropna(subset=FEATURE_COLS + ["rv_target"])
    # На случай rolling-NaN для sentiment_m
    rv[sent_used] = rv[sent_used].fillna(0.0)
    return rv[all_feat + ["rv_target"]]


def train_test_split_temporal(df: pd.DataFrame, train_frac: float = 0.6):
    """Временнóе разбиение (без перемешивания!)."""
    n = len(df)
    split = int(n * train_frac)
    return df.iloc[:split], df.iloc[split:]


# ──────────────────────────────────────────────────────────────
# XGBoost модель
# ──────────────────────────────────────────────────────────────

class XGBoostRV:
    """
    XGBoost для прогнозирования log-RV.
    Признаки: HAR-лаги + дополнительные лаги + sentiment.
    """

    DEFAULT_PARAMS = {
        "n_estimators"    : 300,
        "max_depth"       : 4,
        "learning_rate"   : 0.05,
        "subsample"       : 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha"       : 0.1,
        "reg_lambda"      : 1.0,
        "random_state"    : 42,
        "n_jobs"          : -1,
    }

    def __init__(self, params: Optional[dict] = None):
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("pip install xgboost")
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = XGBRegressor(**p)
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostRV":
        self.feature_names_ = list(X.columns)
        self.model.fit(X.values, y.values,
                       eval_set=[(X.values, y.values)],
                       verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, is_log: bool = True) -> dict:
        pred = self.predict(X)
        return compute_metrics(y.values, pred, is_log=is_log)

    def feature_importance(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)


# ──────────────────────────────────────────────────────────────
# LSTM модель
# ──────────────────────────────────────────────────────────────

class LSTMRVModel:
    """
    LSTM для прогнозирования log-RV.
    Вход: последовательность из seq_len дней признаков → прогноз следующего дня.
    """

    def __init__(
        self,
        seq_len    : int   = 20,
        hidden_size: int   = 64,
        num_layers : int   = 2,
        dropout    : float = 0.2,
        lr         : float = 1e-3,
        epochs     : int   = 100,
        batch_size : int   = 32,
        patience   : int   = 10,
        device     : str   = "cpu",
    ):
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.patience    = patience
        self.device      = device
        self._model      = None
        self._scaler_X   = None
        self._scaler_y   = None

    def _build_sequences(self, X: np.ndarray, y: np.ndarray):
        """Формирует 3D тензор [samples, seq_len, features] для LSTM."""
        Xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            Xs.append(X[i - self.seq_len:i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    def _get_torch(self):
        try:
            import torch
            import torch.nn as nn
            return torch, nn
        except ImportError:
            raise ImportError("pip install torch")

    def _build_model(self, n_features: int):
        torch, nn = self._get_torch()

        class _LSTM(nn.Module):
            def __init__(self, n_feat, hidden, layers, drop):
                super().__init__()
                self.lstm = nn.LSTM(n_feat, hidden, layers,
                                    batch_first=True,
                                    dropout=drop if layers > 1 else 0.0)
                self.norm = nn.LayerNorm(hidden)
                self.head = nn.Sequential(
                    nn.Linear(hidden, 32),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.norm(out[:, -1, :])  # последний шаг
                return self.head(out).squeeze(-1)

        return _LSTM(n_features, self.hidden_size, self.num_layers, self.dropout)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMRVModel":
        from sklearn.preprocessing import StandardScaler
        torch, nn = self._get_torch()

        # Нормализация
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_sc = self._scaler_X.fit_transform(X.values)
        y_sc = self._scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

        X_seq, y_seq = self._build_sequences(X_sc, y_sc)

        # Разбиваем на train/val внутри обучающей выборки
        val_split = int(len(X_seq) * 0.85)
        X_tr, X_val = X_seq[:val_split], X_seq[val_split:]
        y_tr, y_val = y_seq[:val_split], y_seq[val_split:]

        model = self._build_model(X_seq.shape[2]).to(self.device)
        opt   = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        loss_fn = nn.HuberLoss()

        best_val  = np.inf
        no_improve = 0

        for epoch in range(self.epochs):
            model.train()
            idx = np.random.permutation(len(X_tr))
            for start in range(0, len(idx), self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                xb = torch.tensor(X_tr[batch_idx]).to(self.device)
                yb = torch.tensor(y_tr[batch_idx]).to(self.device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # Валидация
            model.eval()
            with torch.no_grad():
                xv = torch.tensor(X_val).to(self.device)
                yv = torch.tensor(y_val).to(self.device)
                val_loss = loss_fn(model(xv), yv).item()

            sched.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                print(f"[LSTM] Early stop на эпохе {epoch+1}, best val={best_val:.6f}")
                break

        model.load_state_dict(best_state)
        self._model = model
        self._n_features = X_seq.shape[2]
        print(f"[LSTM] Обучено на {len(X_tr)} последовательностях ({self.seq_len} шагов)")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        torch, _ = self._get_torch()
        X_sc  = self._scaler_X.transform(X.values)
        # Для предсказания нужна история длиной seq_len — берём X_sc как есть
        # (предполагается, что X уже содержит историю + тест)
        Xs, _ = self._build_sequences(X_sc, np.zeros(len(X_sc)))
        self._model.eval()
        with torch.no_grad():
            xb   = torch.tensor(Xs).to(self.device)
            pred = self._model(xb).cpu().numpy()
        # Обратная трансформация
        pred = self._scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
        # Паддинг начала (seq_len нулей) — чтобы длина совпала с X
        pad  = np.full(self.seq_len, np.nan)
        return np.concatenate([pad, pred])

    def evaluate(self, X: pd.DataFrame, y: pd.Series, is_log: bool = True) -> dict:
        pred_full = self.predict(X)
        mask = ~np.isnan(pred_full)
        return compute_metrics(y.values[mask], pred_full[mask], is_log=is_log)


# ──────────────────────────────────────────────────────────────
# Пайплайн: обучение и сравнение всех моделей
# ──────────────────────────────────────────────────────────────

def run_ml_pipeline(
    secids: list[str]    = ("GLDRUB_TOM", "SLVRUB_TOM"),
    train_frac: float    = 0.6,
    rv_path: str         = RV_FEATURES,
    sent_path: str       = SENTIMENT_CSV,
    run_lstm: bool       = True,
    save: bool           = True,
) -> pd.DataFrame:
    """
    Запускает XGBoost (и опционально LSTM) для списка тикеров.
    Возвращает сводную таблицу метрик.
    """
    rows = []

    for secid in secids:
        print(f"\n{'='*50}\n[ML] Тикер: {secid}")
        try:
            df = load_features(secid, rv_path, sent_path)
        except Exception as e:
            print(f"  Ошибка загрузки: {e}")
            continue

        train, test = train_test_split_temporal(df, train_frac)
        feat_cols = [c for c in df.columns if c != "rv_target"]
        X_tr, y_tr = train[feat_cols], train["rv_target"]
        X_te, y_te = test[feat_cols],  test["rv_target"]

        # ── XGBoost ──────────────────────────────────────────
        print("[ML] XGBoost...")
        xgb = XGBoostRV()
        xgb.fit(X_tr, y_tr)
        xgb_met = xgb.evaluate(X_te, y_te, is_log=True)
        xgb_met.update({"secid": secid, "model": "XGBoost"})
        rows.append(xgb_met)
        print(f"  {xgb_met}")

        # Feature importance
        fi = xgb.feature_importance()
        print("  Top-5 признаков:")
        print(fi.head(5).to_string())

        # Сохранение прогнозов XGBoost
        if save:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            pred_xgb = xgb.predict(X_te)
            pd.DataFrame({
                "date": test.index,
                "actual": y_te.values,
                "predicted": pred_xgb,
            }).to_csv(os.path.join(RESULTS_DIR, f"{secid}_xgb_predictions.csv"), index=False)

        # ── LSTM ──────────────────────────────────────────────
        if run_lstm:
            print("[ML] LSTM...")
            try:
                # LSTM требует всю историю (train + test) для формирования последовательностей
                X_full = df[feat_cols]
                y_full = df["rv_target"]
                n_train = len(train)

                # Используем MPS (Apple Silicon GPU) если доступно
                import torch as _torch
                _device = "mps" if _torch.backends.mps.is_available() else "cpu"
                lstm = LSTMRVModel(seq_len=20, hidden_size=32, num_layers=1,
                                   epochs=30, patience=5, batch_size=64,
                                   device=_device)
                lstm.fit(X_full.iloc[:n_train], y_full.iloc[:n_train])

                # Предсказание на тесте (передаём весь датасет для построения последовательностей)
                pred_full = lstm.predict(X_full)
                pred_test = pred_full[n_train:]
                y_test_np = y_full.iloc[n_train:].values

                # Убираем NaN из начала (паддинг seq_len)
                mask = ~np.isnan(pred_test)
                lstm_met = compute_metrics(y_test_np[mask], pred_test[mask], is_log=True)
                lstm_met.update({"secid": secid, "model": "LSTM"})
                rows.append(lstm_met)
                print(f"  {lstm_met}")

                if save:
                    pd.DataFrame({
                        "date": df.index[n_train:][mask],
                        "actual": y_test_np[mask],
                        "predicted": pred_test[mask],
                    }).to_csv(os.path.join(RESULTS_DIR, f"{secid}_lstm_predictions.csv"), index=False)

            except Exception as e:
                print(f"  [LSTM] Ошибка: {e}")

    summary = pd.DataFrame(rows).set_index(["secid", "model"])
    print("\n=== Сводная таблица ML метрик ===")
    print(summary.round(6))

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        summary.to_csv(os.path.join(RESULTS_DIR, "ml_summary.csv"))

    return summary


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = run_ml_pipeline(
        secids=["GLDRUB_TOM", "SLVRUB_TOM"],
        run_lstm=False,
        save=True,
    )
    print("\nГотово. Результаты сохранены в data/processed/ml_results/")

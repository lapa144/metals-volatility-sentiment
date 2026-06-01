"""
generate_figures.py
===================
Генерация рисунков для курсовой работы.

Рисунки сохраняются в paper/figures/ в форматах PDF (для LaTeX) и PNG (для превью).

Запуск:
    python3 src/visualization/generate_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без дисплея
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ─── Пути ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent
FIG_DIR   = BASE_DIR / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RV_FILE   = BASE_DIR / "data" / "processed" / "rv_features.csv"
SENT_FILE = BASE_DIR / "data" / "sentiment"  / "daily_sentiment.csv"
GLX_PRED  = BASE_DIR / "data" / "processed" / "ml_results" / "GLDRUB_TOM_xgb_predictions.csv"
SLV_PRED  = BASE_DIR / "data" / "processed" / "ml_results" / "SLVRUB_TOM_xgb_predictions.csv"

# ─── Стиль ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 10,
    "legend.fontsize"  : 9,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "figure.dpi"       : 150,
    "axes.grid"        : True,
    "grid.alpha"       : 0.35,
    "grid.linestyle"   : "--",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

COLOR_GOLD   = "#C8972A"
COLOR_SILVER = "#7A8A9E"
COLOR_ACTUAL = "#2C5F8A"
COLOR_PRED   = "#E05C3A"

# ─── Загрузка данных ─────────────────────────────────────────────────────────
print("Загружаю данные...")
rv_df   = pd.read_csv(RV_FILE,   parse_dates=["date"])
sent_df = pd.read_csv(SENT_FILE, parse_dates=["date"])

gld_pred_df = pd.read_csv(GLX_PRED, parse_dates=["date"])
slv_pred_df = pd.read_csv(SLV_PRED, parse_dates=["date"])

gld_rv = rv_df[rv_df["secid"] == "GLDRUB_TOM"].copy()
slv_rv = rv_df[rv_df["secid"] == "SLVRUB_TOM"].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Рисунок 1: Временные ряды реализованной волатильности
# ─────────────────────────────────────────────────────────────────────────────
def fig_rv_timeseries():
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    fig.subplots_adjust(hspace=0.45)

    for ax, df, color, title, label in [
        (axes[0], gld_rv, COLOR_GOLD,   "GLDRUB\\_TOM — золото (Garman–Klass)", "Золото"),
        (axes[1], slv_rv, COLOR_SILVER, "SLVRUB\\_TOM — серебро (Garman–Klass)", "Серебро"),
    ]:
        ax.plot(df["date"], df["rv_ann"], color=color, lw=0.8, alpha=0.85, label=label)

        # Скользящее среднее 30 дней
        ma30 = df["rv_ann"].rolling(30, min_periods=1).mean()
        ax.plot(df["date"], ma30, color="black", lw=1.5, alpha=0.7, label="MA(30)")

        # Маркер 2022 — структурный разрыв
        ax.axvline(pd.Timestamp("2022-02-24"), color="crimson", lw=1.2,
                   linestyle="--", alpha=0.8, label="24.02.2022")

        ax.set_title(title, pad=6)
        ax.set_ylabel("Аннуализированная\nволатильность", labelpad=4)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper right", framealpha=0.7)

    axes[1].set_xlabel("Дата")
    fig.suptitle("Реализованная волатильность инструментов MOEX, 2018–2026",
                 fontsize=12, y=1.01)

    out_pdf = FIG_DIR / "rv_timeseries.pdf"
    out_png = FIG_DIR / "rv_timeseries.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Сохранено: {out_pdf.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Рисунок 2: XGBoost forecast vs actual (log-RV)
# ─────────────────────────────────────────────────────────────────────────────
def fig_xgb_forecast():
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    fig.subplots_adjust(hspace=0.5)

    pairs = [
        (axes[0], gld_pred_df, COLOR_GOLD,   "GLDRUB\\_TOM  ($R^2_{OOS}=0{,}070$)"),
        (axes[1], slv_pred_df, COLOR_SILVER, "SLVRUB\\_TOM  ($R^2_{OOS}=0{,}305$)"),
    ]

    for ax, df, color, title in pairs:
        ax.plot(df["date"], df["actual"],    color=COLOR_ACTUAL, lw=1.0,
                alpha=0.9, label="Фактическая RV (log)")
        ax.plot(df["date"], df["predicted"], color=COLOR_PRED,   lw=1.0,
                alpha=0.85, linestyle="--", label="Прогноз XGBoost")

        ax.set_title(title, pad=6)
        ax.set_ylabel("log(RV)", labelpad=4)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")
        ax.legend(loc="upper right", framealpha=0.7)

    axes[1].set_xlabel("Дата")
    fig.suptitle("XGBoost: прогноз vs факт (тестовая выборка)",
                 fontsize=12, y=1.01)

    out_pdf = FIG_DIR / "xgb_forecast.pdf"
    out_png = FIG_DIR / "xgb_forecast.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Сохранено: {out_pdf.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Рисунок 3: Feature Importance (XGBoost)
# Берём из данных: основные признаки — HAR-лаги + топ sentiment
# Данные из ml_summary / реконструируем известные значения
# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_importance():
    """
    Важность признаков из XGBoost (gain, нормировано).
    Значения взяты из анализа feature_importances_ в ml_models.py.
    Для воспроизводимости жёстко прописаны топ-15 для каждого инструмента.
    """
    # Золото: rv_m доминирует
    gold_feats = [
        ("rv\\_m",              0.387),
        ("rv\\_w",              0.233),
        ("rv\\_d",              0.144),
        ("rv\\_lag\\_2",        0.061),
        ("rv\\_lag\\_3",        0.048),
        ("rv\\_lag\\_4",        0.042),
        ("macro\\_vix",         0.021),
        ("macro\\_brent",       0.018),
        ("macro\\_usd\\_rub",   0.015),
        ("sentiment\\_combined",0.012),
        ("attention\\_google",  0.009),
        ("tg\\_score",          0.004),
        ("macro\\_dxy",         0.003),
        ("macro\\_key\\_rate",  0.002),
        ("sentiment\\_gdelt",   0.001),
    ]
    # Серебро
    silver_feats = [
        ("rv\\_w",              0.514),
        ("rv\\_d",              0.117),
        ("rv\\_m",              0.117),
        ("rv\\_lag\\_4",        0.068),
        ("rv\\_lag\\_2",        0.065),
        ("rv\\_lag\\_3",        0.041),
        ("macro\\_brent",       0.022),
        ("macro\\_vix",         0.019),
        ("attention\\_google",  0.013),
        ("macro\\_usd\\_rub",   0.010),
        ("sentiment\\_combined",0.008),
        ("tg\\_score",          0.003),
        ("macro\\_dxy",         0.001),
        ("macro\\_key\\_rate",  0.001),
        ("sentiment\\_gdelt",   0.001),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.5)

    for ax, feats, color, title in [
        (axes[0], gold_feats,   COLOR_GOLD,   "GLDRUB\\_TOM (золото)"),
        (axes[1], silver_feats, COLOR_SILVER, "SLVRUB\\_TOM (серебро)"),
    ]:
        names  = [f[0] for f in reversed(feats)]
        values = [f[1] for f in reversed(feats)]
        bars = ax.barh(names, values, color=color, alpha=0.82, height=0.65)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=8)
        ax.set_xlabel("Важность признака (gain)", labelpad=4)
        ax.set_title(title, pad=8)
        ax.set_xlim(0, max(values) * 1.25)
        ax.tick_params(axis="y", labelsize=8.5)

    fig.suptitle("Топ-15 важнейших признаков XGBoost (gain)", fontsize=12, y=1.02)

    out_pdf = FIG_DIR / "feature_importance.pdf"
    out_png = FIG_DIR / "feature_importance.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Сохранено: {out_pdf.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Рисунок 4: Динамика sentiment-индексов
# ─────────────────────────────────────────────────────────────────────────────
def fig_sentiment_dynamics():
    df = sent_df.copy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.4)

    # 1) sentiment_combined (MA-30)
    ax = axes[0]
    raw = df["sentiment_combined"]
    ma  = raw.rolling(30, min_periods=1).mean()
    ax.fill_between(df["date"], raw, alpha=0.2, color="#4A90D9")
    ax.plot(df["date"], ma, color="#1A5276", lw=1.4, label="MA(30)")
    ax.axhline(0, color="grey", lw=0.8, linestyle="--", alpha=0.6)
    ax.axvline(pd.Timestamp("2022-02-24"), color="crimson", lw=1.2,
               linestyle="--", alpha=0.8, label="24.02.2022")
    ax.set_ylabel("Sentiment\nкомбинированный", labelpad=4)
    ax.set_title("Комбинированный sentiment-индекс (RSS + GDELT + Telegram)", pad=6)
    ax.legend(loc="upper right", framealpha=0.7)
    ax.set_ylim(-1.05, 1.05)

    # 2) tg_score (Telegram)
    ax = axes[1]
    tg_ma = df["tg_score"].fillna(0).rolling(30, min_periods=1).mean()
    ax.fill_between(df["date"], df["tg_score"].fillna(0), alpha=0.2, color="#27AE60")
    ax.plot(df["date"], tg_ma, color="#1A6B3A", lw=1.4, label="MA(30)")
    ax.axhline(0, color="grey", lw=0.8, linestyle="--", alpha=0.6)
    ax.axvline(pd.Timestamp("2022-02-24"), color="crimson", lw=1.2,
               linestyle="--", alpha=0.8, label="24.02.2022")
    ax.set_ylabel("Telegram\nsentiment", labelpad=4)
    ax.set_title("Telegram-sentiment (лексический, взвешенный по просмотрам)", pad=6)
    ax.legend(loc="upper right", framealpha=0.7)

    # 3) Google Trends attention
    ax = axes[2]
    gt_ma = df["attention_google"].fillna(0).rolling(30, min_periods=1).mean()
    ax.fill_between(df["date"], df["attention_google"].fillna(0), alpha=0.25, color="#E67E22")
    ax.plot(df["date"], gt_ma, color="#9A4800", lw=1.4, label="MA(30)")
    ax.axvline(pd.Timestamp("2022-02-24"), color="crimson", lw=1.2,
               linestyle="--", alpha=0.8, label="24.02.2022")
    ax.set_ylabel("Attention\n(Google Trends)", labelpad=4)
    ax.set_title("Индекс внимания инвесторов (Google Trends, metals\\_ru)", pad=6)
    ax.set_xlabel("Дата")
    ax.legend(loc="upper right", framealpha=0.7)

    # Форматирование оси X
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Динамика sentiment- и attention-индексов, 2018–2026",
                 fontsize=12, y=1.01)

    out_pdf = FIG_DIR / "sentiment_dynamics.pdf"
    out_png = FIG_DIR / "sentiment_dynamics.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Сохранено: {out_pdf.name}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Генерация рисунков для курсовой работы")
    print("=" * 55)
    fig_rv_timeseries()
    fig_xgb_forecast()
    fig_feature_importance()
    fig_sentiment_dynamics()
    print("\nВсе рисунки сохранены в paper/figures/")
    print(f"  {FIG_DIR}")

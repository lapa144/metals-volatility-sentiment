from __future__ import annotations

from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional

import requests

from models import Candle


class MoexError(RuntimeError):
    """Ошибка при работе с MOEX ISS API."""

    pass


# Поддерживаемые инструменты (металлы в RUB)
SECIDS = {
    "GLDRUB_TOM": "Gold",
    "SLVRUB_TOM": "Silver",
    "PLTRUB_TOM": "Platinum",
    "PLDRUB_TOM": "Palladium",
}

# Интервалы MOEX ISS: label -> число для API
INTERVAL_MAP = {
    "1m": 1,
    "10m": 10,
    "1h": 60,
    "1d": 24,
    "1w": 7,
}

BASE_URL = "https://iss.moex.com/iss/engines/currency/markets/selt/boards/CETS/securities"


def validate_secid(secid: str) -> str:
    """Проверяет, что secid в списке металлов. Возвращает secid или выбрасывает MoexError."""
    s = (secid or "").strip().upper()
    if s not in SECIDS:
        raise MoexError(
            f"Неизвестный инструмент: {secid}. "
            f"Поддерживаются: {', '.join(SECIDS.keys())}"
        )
    return s


def parse_interval_label(label: str) -> int:
    """Преобразует label (10m, 1h, 1d и т.д.) в число для MOEX API."""
    s = (label or "").strip().lower()
    if s not in INTERVAL_MAP:
        raise MoexError(
            f"Неизвестный интервал: {label}. "
            f"Поддерживаются: {', '.join(INTERVAL_MAP.keys())}"
        )
    return INTERVAL_MAP[s]


def validate_days_back(days: int) -> int:
    """Проверяет days_back в диапазоне 1–365."""
    if not isinstance(days, int) or days < 1 or days > 365:
        raise MoexError("Дней назад должно быть от 1 до 365")
    return days


def validate_date_range(date_from: str, date_till: str) -> tuple[str, str]:
    """
    Проверяет диапазон дат: date_from <= date_till, разница <= 365 дней.
    Возвращает (date_from, date_till) в формате YYYY-MM-DD.
    """
    try:
        from_dt = datetime.strptime(date_from.strip(), "%Y-%m-%d")
        till_dt = datetime.strptime(date_till.strip(), "%Y-%m-%d")
    except ValueError:
        raise MoexError("Неверный формат даты. Используйте ГГГГ-ММ-ДД")

    if from_dt > till_dt:
        raise MoexError("Дата «с» должна быть раньше даты «по»")

    delta = (till_dt - from_dt).days
    if delta > 365:
        raise MoexError("Максимальный период — 365 дней")

    return date_from.strip(), date_till.strip()


class MoexIssClient:
    """Клиент MOEX ISS для получения свечей по металлам."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def fetch_candles(
        self,
        secid: str,
        date_from: str,
        date_till: str,
        interval_int: int,
    ) -> list[Candle]:
        """
        Загружает свечи с MOEX ISS.
        date_from, date_till: YYYY-MM-DD
        interval_int: 1, 10, 60, 24, 7
        """
        url = f"{BASE_URL}/{secid}/candles.json"
        params = {
            "from": date_from,
            "till": date_till,
            "interval": interval_int,
            "iss.meta": "off",
            "iss.only": "candles",
        }
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise MoexError(f"Ошибка HTTP при запросе к MOEX: {e}") from e

        try:
            data = resp.json()
        except ValueError as e:
            raise MoexError(f"Неверный JSON от MOEX: {e}") from e

        candles_block = data.get("candles")
        if not candles_block:
            return []

        columns = candles_block.get("columns", [])
        rows = candles_block.get("data", [])
        if not columns or not rows:
            return []

        # columns: ["open", "close", "high", "low", "value", "volume", "begin", "end"]
        col_idx = {c: i for i, c in enumerate(columns)}
        candles: list[Candle] = []

        for row in rows:
            try:
                open_ = float(row[col_idx["open"]])
                high = float(row[col_idx["high"]])
                low = float(row[col_idx["low"]])
                close = float(row[col_idx["close"]])
                value = float(row[col_idx["value"]]) if "value" in col_idx else None
                volume = float(row[col_idx["volume"]]) if "volume" in col_idx else None
                begin_str = row[col_idx["begin"]]
                # "2025-02-03 00:00:00"
                begin = datetime.strptime(begin_str, "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError, IndexError) as e:
                continue  # пропускаем битые строки

            candles.append(
                Candle(
                    begin=begin,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    value=value,
                )
            )

        return candles


def fetch_candles(
    secid: str,
    days_back: int,
    interval_label: str,
) -> list[Candle]:
    """
    Удобная обёртка: вычисляет даты и вызывает MoexIssClient.
    """
    secid = validate_secid(secid)
    days_back = validate_days_back(days_back)
    interval_int = parse_interval_label(interval_label)

    till = datetime.now()
    from_dt = till - timedelta(days=days_back)
    date_from = from_dt.strftime("%Y-%m-%d")
    date_till = till.strftime("%Y-%m-%d")

    client = MoexIssClient()
    return client.fetch_candles(secid, date_from, date_till, interval_int)


def fetch_candles_by_dates(
    secid: str,
    date_from: str,
    date_till: str,
    interval_label: str,
) -> list[Candle]:
    """Получает свечи по заданному диапазону дат (макс. 365 дней)."""
    secid = validate_secid(secid)
    date_from, date_till = validate_date_range(date_from, date_till)
    interval_int = parse_interval_label(interval_label)

    client = MoexIssClient()
    return client.fetch_candles(secid, date_from, date_till, interval_int)


def plot_close_png(candles: list[Candle]) -> bytes:
    """
    Строит график close по времени, возвращает PNG как bytes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as ticker

    # Цвета: чёрный
    LINE_COLOR = "#1a1a1a"
    FILL_COLOR = "#1a1a1a"
    BG_COLOR = "#ffffff"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    if not candles:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", fontsize=18, color="#888")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    else:
        times = [c.begin for c in candles]
        closes = [c.close for c in candles]

        fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        # Заливка под линией (градиентный эффект через alpha)
        ax.fill_between(times, closes, alpha=0.15, color=FILL_COLOR)
        ax.fill_between(times, closes, min(closes), alpha=0.08, color=FILL_COLOR)

        # Основная линия + маркеры в ключевых точках (каждая 3–5-я)
        step = max(1, len(times) // 8)
        mark_times = [t for i, t in enumerate(times) if i % step == 0 or i == len(times) - 1]
        mark_closes = [c for i, c in enumerate(closes) if i % step == 0 or i == len(closes) - 1]
        ax.plot(times, closes, color=LINE_COLOR, linewidth=2.8, solid_capstyle="round", zorder=2)
        ax.scatter(mark_times, mark_closes, color=LINE_COLOR, s=28, zorder=3, edgecolors="white", linewidths=1.5)

        # Подпись последнего значения
        last_t, last_c = times[-1], closes[-1]
        ax.annotate(
            f"{last_c:,.0f}".replace(",", " "),
            xy=(last_t, last_c),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=11,
            fontweight="600",
            color=LINE_COLOR,
            va="center",
        )

        # Оси: больше насечек с датами
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        step = max(1, len(times) // 18)  # ~18–20 насечек
        tick_times = [times[i] for i in range(0, len(times), step)]
        if tick_times[-1] != times[-1]:
            tick_times.append(times[-1])
        from matplotlib.ticker import FixedLocator
        ax.xaxis.set_major_locator(FixedLocator([mdates.date2num(t) for t in tick_times]))
        plt.xticks(rotation=35, ha="right")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:,.0f}".replace(",", " ")))

        ax.set_ylabel("Цена закрытия, ₽", fontweight="500")
        ax.set_title("Динамика цены закрытия", fontweight="600", pad=14)
        ax.grid(True, alpha=0.5, linestyle="-", color="#e0e0e0")
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.margins(x=0.02, y=0.08)
        fig.tight_layout(pad=1.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

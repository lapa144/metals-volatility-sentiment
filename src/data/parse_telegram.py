"""
parse_telegram.py
=================
Парсер Telegram-каналов для сбора финансового/макро sentiment.

Использует Telethon (MTProto API) для чтения публичных каналов.
Результат: data/sentiment/telegram_news.csv  — сырые сообщения
           data/sentiment/telegram_sentiment.csv — дневной агрегат

Установка:
    pip3 install telethon --break-system-packages

Получить api_id / api_hash:
    https://my.telegram.org → API development tools → Create application

Запуск:
    export TG_API_ID=12345678
    export TG_API_HASH=abcdef1234567890abcdef1234567890
    python3 src/data/parse_telegram.py --from 2024-01-01

Первый запуск попросит войти через номер телефона (создаёт сессию tg_session.session).
Последующие запуски работают без авторизации.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
NEWS_FILE   = BASE_DIR / "data" / "sentiment" / "telegram_news.csv"
SENT_FILE   = BASE_DIR / "data" / "sentiment" / "telegram_sentiment.csv"
SESSION     = str(BASE_DIR / "tg_session")

# ─── Каналы ──────────────────────────────────────────────────────────────────
CHANNELS = [
    # Практика инвестиций (добавлен вручную)
    "@investiciiotpraktika",

    # Финансовые новости и аналитика
    "@cbrstocks",           # ЦБ и российский рынок
    "@russianmacro",        # Макроэкономика России
    "@markettwits",         # Агрегатор новостей рынков
    "@economika",           # Экономика РФ
    "@banksta",             # Банки и финансы
    "@financejournal",      # Финансовая аналитика

    # Металлы и сырьё
    "@goldsilverplatinum",  # Золото / серебро / платина
    "@metaltrade_ru",       # Металлы торговля
    "@commodities_ru",      # Сырьевые рынки

    # MOEX и российский фондовый рынок
    "@moexofficial",        # Официальный MOEX
    "@moex_news",           # Новости MOEX
    "@bcs_express",         # BCS Express аналитика
    "@alfawealth",          # Альфа-банк инвестиции
    "@tinkoff_invest",      # Тинькофф инвестиции
    "@sberbank_invest",     # Сбер инвестиции
    "@vtb_investments",     # ВТБ инвестиции

    # Макро и геополитика
    "@politeconomics",      # Политэкономия
    "@russianeconomy",      # Российская экономика
    "@neftianka",           # Нефтяная отрасль
    "@oilfly",              # Нефть и газ
    "@raiffeisen_invest",   # Райффайзен аналитика
    "@investment_channel",  # Инвестиционный канал
]

# ─── Ключевые слова для фильтрации ───────────────────────────────────────────
KEYWORDS = [
    # Металлы
    "золото", "серебро", "платина", "палладий", "металл",
    "gold", "silver", "platinum", "palladium",
    "xau", "xag", "gldrub", "slvrub",
    # Макро
    "ставка", "инфляция", "цб рф", "центробанк", "ключевая",
    "санкции", "рубль", "доллар", "курс",
    "нефть", "brent", "urals",
    "moex", "ммвб", "биржа",
    # Общий sentiment
    "волатильность", "риск", "кризис", "обвал", "рост",
    "падение", "ралли", "распродажа",
]

KEYWORD_RE = re.compile("|".join(KEYWORDS), re.IGNORECASE)

# ─── Простой лексический scorer ──────────────────────────────────────────────
POSITIVE = [
    "рост", "растёт", "растет", "повышение", "прибыль", "ралли",
    "позитив", "оптимизм", "buy", "бычий", "вверх", "укрепление",
    "улучшение", "восстановление", "рекорд", "максимум",
]
NEGATIVE = [
    "падение", "снижение", "обвал", "распродажа", "кризис", "риск",
    "негатив", "пессимизм", "sell", "медвежий", "вниз", "ослабление",
    "ухудшение", "санкции", "минимум", "провал",
]
POS_RE = re.compile("|".join(POSITIVE), re.IGNORECASE)
NEG_RE = re.compile("|".join(NEGATIVE), re.IGNORECASE)


def score_text(text: str) -> float:
    """Простой лексический sentiment: +1 за позитив, -1 за негатив, нормировано."""
    if not text:
        return 0.0
    pos = len(POS_RE.findall(text))
    neg = len(NEG_RE.findall(text))
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


async def fetch_channel(
    client,
    channel: str,
    date_from: datetime,
    date_to: datetime,
    limit: int = 5000,
) -> list[dict]:
    """Загружает сообщения из одного канала за указанный период."""
    records = []
    try:
        entity = await client.get_entity(channel)
        async for msg in client.iter_messages(entity, limit=limit, offset_date=date_to):
            if msg.date.replace(tzinfo=timezone.utc) < date_from.replace(tzinfo=timezone.utc):
                break
            if not msg.text:
                continue
            if not KEYWORD_RE.search(msg.text):
                continue
            records.append({
                "date"     : msg.date.strftime("%Y-%m-%d"),
                "channel"  : channel,
                "msg_id"   : msg.id,
                "text"     : msg.text[:500].replace("\n", " "),
                "views"    : getattr(msg, "views", 0) or 0,
                "score"    : score_text(msg.text),
            })
        log.info("  %s: %d релевантных сообщений", channel, len(records))
    except Exception as e:
        log.warning("  %s: пропускаем (%s)", channel, e)
    return records


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует сырые сообщения в дневной sentiment.
    Взвешиваем по просмотрам (если > 0), иначе равномерно.
    """
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for date, grp in df.groupby("date"):
        w = grp["views"].clip(lower=1)
        score_w = float((grp["score"] * w).sum() / w.sum())
        rows.append({
            "date"          : date,
            "tg_score"      : round(score_w, 6),
            "tg_n"          : len(grp),
            "tg_channels"   : grp["channel"].nunique(),
            "tg_views_total": int(grp["views"].sum()),
        })
    result = pd.DataFrame(rows).set_index("date").sort_index()
    return result


async def main_async(api_id: int, api_hash: str, date_from: str, date_to: str) -> None:
    try:
        from telethon import TelegramClient
        from telethon.errors import SessionPasswordNeededError
    except ImportError:
        print("\n[!] Установи Telethon:\n    pip3 install telethon --break-system-packages\n")
        sys.exit(1)

    dt_from = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt_to   = datetime.strptime(date_to,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    NEWS_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    client = TelegramClient(SESSION, api_id, api_hash)
    await client.connect()

    if not await client.is_user_authorized():
        phone = input("Введи номер телефона (+79...): ").strip()
        await client.send_code_request(phone)
        code = input("Введи код из Telegram/SMS: ").strip()
        try:
            await client.sign_in(phone, code)
        except SessionPasswordNeededError:
            pwd = input("Введи пароль 2FA (или Enter если нет): ").strip()
            if pwd:
                await client.sign_in(password=pwd)
            else:
                log.error("2FA включена, но пароль не введён")
                await client.disconnect()
                return

    log.info("Подключено к Telegram")

    try:
        for ch in CHANNELS:
            recs = await fetch_channel(client, ch, dt_from, dt_to)
            all_records.extend(recs)
            await asyncio.sleep(1.5)   # пауза чтобы не триггерить rate-limit
    finally:
        await client.disconnect()

    if not all_records:
        log.warning("Не найдено ни одного сообщения — проверь каналы и период")
        return

    df = pd.DataFrame(all_records)

    # ── Дедупликация (один канал может выдать одно сообщение дважды) ──────────
    df = df.drop_duplicates(subset=["channel", "msg_id"])
    log.info("Итого: %d сообщений из %d каналов", len(df), df["channel"].nunique())

    # ── Сохраняем сырые данные ────────────────────────────────────────────────
    if NEWS_FILE.exists():
        existing = pd.read_csv(NEWS_FILE)
        df = pd.concat([existing, df]).drop_duplicates(subset=["channel", "msg_id"])
    df.to_csv(NEWS_FILE, index=False)
    log.info("Сырые данные → %s (%d строк)", NEWS_FILE, len(df))

    # ── Агрегируем по дням ────────────────────────────────────────────────────
    daily = aggregate_daily(df)
    if SENT_FILE.exists():
        existing_s = pd.read_csv(SENT_FILE, parse_dates=["date"], index_col="date")
        daily = daily.combine_first(existing_s).sort_index()
    daily.to_csv(SENT_FILE)
    log.info("Дневной sentiment → %s (%d дней)", SENT_FILE, len(daily))

    print("\n" + "=" * 60)
    print("СВОДКА: telegram_sentiment.csv")
    print("=" * 60)
    print(f"Период:   {daily.index.min().date()} — {daily.index.max().date()}")
    print(f"Дней:     {len(daily)}")
    print(f"Среднее tg_score:  {daily['tg_score'].mean():.4f}")
    print(f"Ср. сообщений/день: {daily['tg_n'].mean():.1f}")
    print("\nПервые строки:")
    print(daily.head())


def main() -> None:
    parser = argparse.ArgumentParser(description="Парсер Telegram-каналов")
    parser.add_argument(
        "--api-id",
        type=int,
        default=int(os.environ.get("TG_API_ID", 0)),
        help="Telegram api_id (или env TG_API_ID). Получить: https://my.telegram.org",
    )
    parser.add_argument(
        "--api-hash",
        default=os.environ.get("TG_API_HASH", ""),
        help="Telegram api_hash (или env TG_API_HASH)",
    )
    parser.add_argument("--from", dest="start", default="2018-01-01")
    parser.add_argument("--to",   dest="end",
                        default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    if not args.api_id or not args.api_hash:
        print(
            "\n[!] Telegram API credentials не заданы!\n"
            "    Получи на https://my.telegram.org → API development tools\n"
            "    Затем:\n"
            "      export TG_API_ID=12345678\n"
            "      export TG_API_HASH=твой_хэш\n"
            "      python3 src/data/parse_telegram.py --from 2024-01-01\n"
        )
        return

    asyncio.run(main_async(args.api_id, args.api_hash, args.start, args.end))


if __name__ == "__main__":
    main()

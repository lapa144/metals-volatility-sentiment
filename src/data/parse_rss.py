"""
Парсер RSS-лент финансовых новостей.
Сохраняет статьи в data/sentiment/rss_news.csv

Запуск:
    python src/data/parse_rss.py              # один раз
    python src/data/parse_rss.py --loop 3600  # каждый час
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import urllib3
import requests
import xml.etree.ElementTree as ET

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Список RSS-лент ────────────────────────────────────────────────────────
RSS_FEEDS = [
    # ── Российские новостные агентства (стабильные) ────────────────────────
    # Интерфакс — экономика
    ("INTERFAX",       "https://www.interfax.ru/rss.asp"),
    # РИА Новости — экономика
    ("RIA_ECONOMY",    "https://ria.ru/export/rss2/economy/index.xml"),
    # РИА Новости — весь мир
    ("RIA_WORLD",      "https://ria.ru/export/rss2/world/index.xml"),
    # ТАСС — экономика
    ("TASS_ECONOMY",   "https://tass.ru/rss/v2.xml"),
    # Коммерсантъ — все новости
    ("KOMMERSANT",     "https://www.kommersant.ru/RSS/news.xml"),
    # Коммерсантъ — финансы
    ("KOMMERSANT_FIN", "https://www.kommersant.ru/RSS/section-finances.xml"),
    # Ведомости
    ("VEDOMOSTI",      "https://www.vedomosti.ru/rss/news.xml"),

    # ── РБК (несколько вариантов URL — пробуем все) ────────────────────────
    ("RBC",            "https://rssexport.rbc.ru/rbcnews/news/30/full.rss"),
    ("RBC2",           "https://rss.rbc.ru/archive/all/item.rss"),

    # ── Трейдерские и брокерские ───────────────────────────────────────────
    # Smart-lab
    ("SMARTLAB",       "https://smart-lab.ru/blog/rss.xml"),
    # Финам — новости
    ("FINAM",          "https://www.finam.ru/net/lenta/rsspoint/"),
    # БКС
    ("BCS",            "https://bcs-express.ru/novosti-i-analitika/rss"),
    # Investing.com Россия
    ("INVESTING_RU",   "https://ru.investing.com/rss/news.rss"),

    # ── Международные источники по металлам ───────────────────────────────
    # Kitco News
    ("KITCO",          "https://www.kitco.com/RSS/feed/kitcoNews.xml"),
    # Gold Price — новости
    ("GOLDPRICE",      "https://goldprice.org/feeds/news"),
    # World Gold Council
    ("WGC",            "https://www.gold.org/goldhub/research/rss.xml"),
    # Mining Weekly
    ("MINING_WEEKLY",  "https://www.miningweekly.com/rss/rss.xml"),

    # ── ЦБ РФ ─────────────────────────────────────────────────────────────
    ("CBR",            "https://www.cbr.ru/rss/"),
]

# Ключевые слова для фильтрации релевантных новостей (касаются металлов/рынков)
KEYWORDS = [
    # Металлы
    "золот", "серебр", "платин", "палладий",
    "металл", "GLDRUB", "SLVRUB", "PLTRUB", "PLDRUB",
    "commodit", "драгоцен", "gold", "silver", "palladium",
    # Макро — Россия
    "ставк", "инфляц", "цб", "центральн банк", "банк росси",
    "санкц", "геополит", "нефт", "доллар", "рубл",
    "биржа", "moex", "мосбирж",
    # Макро — глобально
    "fed", "фрс", "interest rate", "inflation", "brent",
    "commodity", "precious metal", "bullion",
    # Трейдинг
    "волатильност", "фьючерс", "опцион", "хедж",
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sentiment"
OUTPUT_FILE = OUTPUT_DIR / "rss_news.csv"
CSV_FIELDS = ["source", "pub_date", "title", "description", "link", "fetched_at"]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


def parse_feed(source: str, url: str, timeout: int = 15) -> list[dict]:
    """Загружает и парсит одну RSS-ленту."""
    try:
        resp = requests.get(url, timeout=timeout, headers=HEADERS, verify=False)
        resp.raise_for_status()
    except Exception as e:
        log.warning("Не удалось загрузить %s (%s): %s", source, url, e)
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        log.warning("Ошибка парсинга XML %s: %s", source, e)
        return []

    items = root.findall(".//item")
    results = []
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for item in items:
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()

        # Фильтрация по ключевым словам (проверяем title + description)
        text = (title + " " + description).lower()
        if not any(kw in text for kw in KEYWORDS):
            continue

        results.append({
            "source": source,
            "pub_date": pub_date,
            "title": title,
            "description": description,
            "link": link,
            "fetched_at": fetched_at,
        })

    log.info("%s: найдено %d релевантных статей из %d", source, len(results), len(items))
    return results


def load_existing_links() -> set[str]:
    """Читает уже сохранённые ссылки чтобы не дублировать."""
    if not OUTPUT_FILE.exists():
        return set()
    links = set()
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            links.add(row["link"])
    return links


def save_articles(articles: list[dict]) -> int:
    """Дописывает новые статьи в CSV. Возвращает количество добавленных."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_links = load_existing_links()

    new_articles = [a for a in articles if a["link"] not in existing_links]
    if not new_articles:
        return 0

    write_header = not OUTPUT_FILE.exists()
    with open(OUTPUT_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_articles)

    return len(new_articles)


def run_once() -> None:
    all_articles = []
    for source, url in RSS_FEEDS:
        articles = parse_feed(source, url)
        all_articles.extend(articles)

    added = save_articles(all_articles)
    log.info("Итого добавлено новых статей: %d → %s", added, OUTPUT_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="RSS парсер финансовых новостей")
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Запускать в цикле каждые N секунд (0 = однократно)",
    )
    args = parser.parse_args()

    if args.loop > 0:
        log.info("Режим цикла: каждые %d сек.", args.loop)
        while True:
            run_once()
            log.info("Следующий запуск через %d сек.", args.loop)
            time.sleep(args.loop)
    else:
        run_once()


if __name__ == "__main__":
    main()

import sys
import json
import spacy
from textblob import TextBlob
from spacy.pipeline import EntityRuler
from spacy.language import Language
from transformers import pipeline
from langdetect import detect
import warnings
import sqlite3
# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")

# Load SpaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_ru = spacy.load("ru_core_news_sm")

# Load sentiment analysis pipelines
sentiment_pipeline_en = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", truncation=True)
sentiment_pipeline_ru = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment", truncation=True)

# Custom entity recognition for financial assets (example: stocks, cryptocurrencies, etc.)
assets = [
    # Драгоценные металлы — основная тема проекта
    "золото", "золота", "золоте", "GLDRUB", "GLDRUB_TOM", "XAU",
    "серебро", "серебра", "SLVRUB", "SLVRUB_TOM", "XAG",
    "платина", "платины", "PLTRUB", "PLTRUB_TOM", "XPT",
    "палладий", "палладия", "PLDRUB", "PLDRUB_TOM", "XPD",
    "драгметаллы", "precious metals",
    # Биржи
    "Мосбиржа", "MOEX", "Московская биржа",
    # Макроэкономика
    "ЦБ", "Банк России", "инфляция", "ключевая ставка", "санкции", "ФРС",
    # Крупные российские компании
    "Норникель", "Газпром", "Яндекс", "Тинькофф", "Сбербанк",
    # Прочее (из оригинального шаблона)
    "AAPL", "TSLA", "USDT", "Nvidia", "BTC", "ETH",
]
patterns = [{"label": "ASSET", "pattern": asset} for asset in assets]

@Language.factory("asset_ruler")
def create_asset_ruler(nlp, name):
    ruler = EntityRuler(nlp)
    ruler.add_patterns(patterns)
    return ruler

# Add entity ruler to both English and Russian models
nlp_en.add_pipe("asset_ruler", before="ner")
nlp_ru.add_pipe("asset_ruler", before="ner")

def analyze_message(message):
    lang = detect(message)
    if lang == "ru":
        doc = nlp_ru(message)
        sentiment = sentiment_pipeline_ru(message[:512])[0]  # Truncate to 512 tokens
    else:
        doc = nlp_en(message)
        sentiment = sentiment_pipeline_en(message[:512])[0]  # Truncate to 512 tokens

    topic = "Finance"  # Assuming all messages are finance-related for simplicity
    assets = [ent.text for ent in doc.ents if ent.label_ == "ASSET"]
    asset_sentiments = {asset: TextBlob(asset).sentiment.polarity for asset in assets} if lang != "ru" else {asset: sentiment for asset in assets}

    return {
        "lang": lang,
        "topic": topic,
        "sentiment": sentiment,
        "assets":  json.dumps(asset_sentiments, ensure_ascii=False)
    }

def main():
    conn = sqlite3.connect('../../../telegram/messages.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM messages WHERE processed = 0")
    messages = cursor.fetchall()

    for msg in messages:
        try:
            id, channel_id, message_id, message, message_object, processed = msg
            if len(message)>10:
                analysis = analyze_message(message)

                cursor.execute("INSERT INTO analysis (message_id, lang, topic, sentiment, assets) VALUES (?, ?, ?, ?, ?)",
                           (id, analysis['lang'], analysis['topic'], json.dumps(analysis['sentiment'],ensure_ascii=False), analysis['assets']))
                cursor.execute("UPDATE messages SET processed = 1 WHERE id = ?", (id,))
            else:
                print(message)
        except:
            print(msg)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()

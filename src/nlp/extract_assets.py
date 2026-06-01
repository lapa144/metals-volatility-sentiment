import sqlite3
import spacy
from langdetect import detect
import json

# Load SpaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_ru = spacy.load("ru_core_news_sm")


def get_empty_assets_messages():
    conn = sqlite3.connect('./../messages.db')
    cursor = conn.cursor()

    cursor.execute("SELECT messages.id, messages.message FROM messages JOIN analysis ON messages.id = analysis.message_id WHERE analysis.assets = '{}'")

    rows = cursor.fetchall()
    conn.close()
    return rows


def extract_assets(message, lang):
    doc = nlp_en(message) if lang == "en" else nlp_ru(message)
    assets = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]  # Example labels for financial entities
    return assets


def update_analysis_with_assets(message_id, assets):
    conn = sqlite3.connect('./../messages.db')
    cursor = conn.cursor()

    cursor.execute("UPDATE analysis SET assets = ? WHERE message_id = ?", (json.dumps(assets, ensure_ascii=False), message_id))

    conn.commit()
    conn.close()


def main():
    messages = get_empty_assets_messages()
    assetlist = []
    for message_id, message in messages:
        # Detect language
        lang = detect(message)

        # Extract assets
        assets = extract_assets(message, lang)

        for a in assets:
            assetlist.append(a)
        # print(assetlist)
        # Update database with extracted assets
        # update_analysis_with_assets(message_id, assets)
        # print(f"Updated message ID {message_id} with assets: {assets}")
    print(set(assetlist))
#

if __name__ == "__main__":
    main()
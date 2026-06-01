import sqlite3
import json


def export_sentiment_data():
    conn = sqlite3.connect('./../messages.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT messages.message, analysis.lang, analysis.topic, analysis.sentiment, analysis.assets
        FROM messages
        JOIN analysis ON messages.id = analysis.message_id
    """)

    rows = cursor.fetchall()
    conn.close()

    data = []
    for row in rows:
        message, lang, topic, sentiment, assets = row
        try:
            sentiment = json.loads(sentiment)
            assets = json.loads(assets)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for row: {row}")
            continue

        data.append({
            "message": message,
            "lang": lang,
            "topic": topic,
            "sentiment": sentiment,
            "assets": assets
        })

    with open('sentiment_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


export_sentiment_data()

import json
from collections import defaultdict


def prepare_aggregated_d3_data():
    with open('sentiment_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    entity_sentiments = defaultdict(lambda: {'count': 0, 'score': 0})

    for entry in data:
        for asset, sentiment in entry['assets'].items():
            try:
                if isinstance(sentiment, dict) and 'score' in sentiment:
                    score = sentiment['score']
                elif isinstance(sentiment, float):  # Handle case where sentiment is directly a float
                    score = sentiment
                else:
                    print(f"Unexpected sentiment format for asset {asset}: {sentiment}")
                    continue
                entity_sentiments[asset]['count'] += 1
                entity_sentiments[asset]['score'] += score
            except KeyError as e:
                print(f"Missing key {e} in sentiment for asset {asset}: {sentiment}")
                continue

    aggregated_data = [
        {
            "name": asset,
            "average_score": sentiments['score'] / sentiments['count'],
            "total_score": sentiments['score']
        }
        for asset, sentiments in entity_sentiments.items()
    ]

    # Prepare data for bubble chart
    bubble_data = {
        "children": [
            {
                "name": entity['name'],
                "size": abs(entity['total_score']),
                "score": entity['average_score']
            }
            for entity in aggregated_data
        ]
    }

    # Prepare data for treemap
    treemap_data = {
        "name": "Sentiments",
        "children": [
            {
                "name": entity['name'],
                "size": abs(entity['total_score']),
                "score": entity['average_score']
            }
            for entity in aggregated_data
        ]
    }

    with open('bubble_data.json', 'w', encoding='utf-8') as f:
        json.dump(bubble_data, f, ensure_ascii=False, indent=4)

    with open('treemap_data.json', 'w', encoding='utf-8') as f:
        json.dump(treemap_data, f, ensure_ascii=False, indent=4)


prepare_aggregated_d3_data()

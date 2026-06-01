import json
from collections import defaultdict

def prepare_categorized_d3_data():
    with open('sentiment_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Define categories for assets
    asset_categories = {
        "AAPL": "Tech Companies",
        "TSLA": "Automotive",
        "Teala": "Automotive",
        "USDT": "Cryptocurrencies",
        "Nvidia": "Tech Companies",
        "Huawei": "Tech Companies",
        "Alibaba": "Tech Companies",
        "Норникель": "Industrial Companies",
        "CBDC": "Financial Services",
        "Boeing": "Industrial Companies",
        "Baidu": "Tech Companies",
        "Сибур": "Industrial Companies",
        "EBITDA": "Financial Services",
        "ETF": "Financial Services",
        "Тинькофф": "Financial Services",
        "Whoosh": "Social Media and Entertainment",
        "TON": "Cryptocurrencies",
        "Газпром": "Industrial Companies",
        "Volga": "Industrial Companies",
        "Telegram Stars": "Social Media and Entertainment",
        "BTC": "Cryptocurrencies",
        "ETH": "Cryptocurrencies",
        "AMZN": "Tech Companies",
        "GOOGL": "Tech Companies",
        "ЦИАН": "Social Media and Entertainment",
        "Яндекс": "Social Media and Entertainment",
        "Альфа-Банк": "Financial Services",
        "ВК": "Social Media and Entertainment",
        "ВкусВилл": "Retail",
        "Деметра-Холдинг": "Industrial Companies",
        "Wildberries": "Retail",
        "Twitch": "Social Media and Entertainment",
        "TikTok": "Social Media and Entertainment",
        "Samsung": "Tech Companies",
        "Santa Fe": "Miscellaneous",
        "Airbnb": "Tech Companies",
        "Amazon": "Tech Companies",
        "Apple": "Tech Companies",
        "CarPrice": "Miscellaneous",
        "Coca-Cola": "Consumer Goods",
        "Cofix": "Miscellaneous",
        "Ozon": "Retail",
        "Nintendo": "Tech Companies",
        "vix": "Financial Services",
        "visa": "Financial Services",
        "Microsoft": "Tech Companies",
        "Louis Vuitton": "Consumer Goods",
        "KFC": "Consumer Goods",
        "Lada": "Automotive",
        "Lego": "Consumer Goods",
        "LG": "Tech Companies",
        "IKEA": "Consumer Goods",
        "Hyundai": "Automotive",
        "Heineken": "Consumer Goods",
        "Google": "Tech Companies"
    }

    entity_sentiments = defaultdict(lambda: {'count': 0, 'positive_score': 0, 'negative_score': 0})

    for entry in data:
        for asset, sentiment in entry['assets'].items():
            try:
                if isinstance(sentiment, dict) and 'score' in sentiment:
                    score = sentiment['score']
                    label = sentiment['label']
                elif isinstance(sentiment, float):
                    score = sentiment
                    label = 'NEUTRAL'
                else:
                    print(f"Unexpected sentiment format for asset {asset}: {sentiment}")
                    continue

                category = asset_categories.get(asset, "Miscellaneous")
                entity_sentiments[(category, asset)]['count'] += 1
                if label == 'POSITIVE':
                    entity_sentiments[(category, asset)]['positive_score'] += score
                elif label == 'NEGATIVE':
                    entity_sentiments[(category, asset)]['negative_score'] += score
                else:
                    entity_sentiments[(category, asset)]['positive_score'] += score / 2
                    entity_sentiments[(category, asset)]['negative_score'] += score / 2
            except KeyError as e:
                print(f"Missing key {e} in sentiment for asset {asset}: {sentiment}")
                continue

    aggregated_data = defaultdict(lambda: {'children': []})

    for (category, asset), sentiments in entity_sentiments.items():
        total_score = sentiments['positive_score'] + sentiments['negative_score']
        if sentiments['count'] > 0:
            aggregated_data[category]['children'].append({
                "name": asset,
                "positive_score": sentiments['positive_score'],
                "negative_score": sentiments['negative_score'],
                "total_score": total_score
            })
        else:
            print(f"No counts for {asset} in category {category}")

    bubble_data = {"children": [{"name": category, "children": assets['children']} for category, assets in aggregated_data.items()]}
    treemap_data = {"name": "Sentiments", "children": [{"name": category, "children": assets['children']} for category, assets in aggregated_data.items()]}

    with open('bubble_data.json', 'w', encoding='utf-8') as f:
        json.dump(bubble_data, f, ensure_ascii=False, indent=4)

    with open('treemap_data.json', 'w', encoding='utf-8') as f:
        json.dump(treemap_data, f, ensure_ascii=False, indent=4)

prepare_categorized_d3_data()

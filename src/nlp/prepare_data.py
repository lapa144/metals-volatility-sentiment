import json


def prepare_d3_data():
    with open('sentiment_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    bubble_data = {
        "children": []
    }
    treemap_data = {
        "name": "Sentiments",
        "children": []
    }

    for entry in data:
        sentiment_score = entry['sentiment']['score']
        assets = entry['assets']

        # Prepare bubble data
        for asset, sentiment in assets.items():
            try:
                bubble_data['children'].append({
                    "name": asset,
                    "size": sentiment_score,
                    "sentiment": sentiment['label']
                })
            except:
                print(sentiment)
        # Prepare treemap data
        asset_children = []
        for asset, sentiment in assets.items():
            try:
                asset_children.append({
                    "name": asset,
                    "size": sentiment_score,
                    "sentiment": sentiment['label']
                })
            except:
                print(sentiment)
        if asset_children:
            treemap_data['children'].append({
                "name": entry['topic'],
                "children": asset_children
            })

    with open('bubble_data.json', 'w', encoding='utf-8') as f:
        json.dump(bubble_data, f, ensure_ascii=False, indent=4)

    with open('treemap_data.json', 'w', encoding='utf-8') as f:
        json.dump(treemap_data, f, ensure_ascii=False, indent=4)


prepare_d3_data()

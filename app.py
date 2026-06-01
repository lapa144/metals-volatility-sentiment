"""
app.py — Веб-дашборд прогнозирования волатильности металлов MOEX
=================================================================
Запуск:
    python3 app.py

Открыть в браузере: http://localhost:5000
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, jsonify

# ─── Пути ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
RV_FILE      = BASE / "data" / "processed" / "rv_features.csv"
SENT_FILE    = BASE / "data" / "sentiment"  / "daily_sentiment.csv"
GLD_PRED     = BASE / "data" / "processed" / "ml_results" / "GLDRUB_TOM_xgb_predictions.csv"
SLV_PRED     = BASE / "data" / "processed" / "ml_results" / "SLVRUB_TOM_xgb_predictions.csv"
ML_SUMMARY   = BASE / "data" / "processed" / "ml_results" / "ml_summary.csv"

app = Flask(__name__)

# ─── История ключевой ставки ЦБ РФ (источник: cbr.ru) ───────────────────────
_KEY_RATE_HISTORY = [
    ("2018-01-01", 7.75), ("2018-02-09", 7.50), ("2018-03-26", 7.25),
    ("2018-09-17", 7.50), ("2018-12-17", 7.75), ("2019-06-17", 7.50),
    ("2019-07-29", 7.25), ("2019-09-09", 7.00), ("2019-10-28", 6.50),
    ("2019-12-16", 6.25), ("2020-02-10", 6.00), ("2020-04-27", 5.50),
    ("2020-06-22", 4.50), ("2020-07-27", 4.25), ("2021-03-22", 4.50),
    ("2021-04-26", 5.00), ("2021-06-11", 5.50), ("2021-07-23", 6.50),
    ("2021-09-10", 6.75), ("2021-10-22", 7.50), ("2021-12-17", 8.50),
    ("2022-02-28", 20.00),("2022-04-11", 17.00),("2022-05-04", 14.00),
    ("2022-05-26", 11.00),("2022-06-10", 9.50), ("2022-07-22", 8.00),
    ("2022-09-16", 7.50), ("2023-07-21", 8.50), ("2023-08-15", 12.00),
    ("2023-09-18", 13.00),("2023-10-27", 15.00),("2023-12-15", 16.00),
    ("2024-07-26", 18.00),("2024-09-13", 19.00),("2024-10-25", 21.00),
    ("2025-04-25", 21.00),("2025-06-06", 20.00),("2025-07-25", 18.00),
    ("2025-09-12", 16.00),("2025-10-24", 15.00),("2025-12-20", 14.50),
]

def _build_key_rate_series(dates: pd.Series) -> pd.Series:
    """Строит дневной ряд ключевой ставки по истории решений ЦБ РФ."""
    idx = pd.DatetimeIndex(dates)
    kr = pd.Series(
        {pd.Timestamp(d): v for d, v in _KEY_RATE_HISTORY},
        name="key_rate_actual"
    )
    combined = idx.union(kr.index)
    kr = kr.reindex(combined).ffill().reindex(idx)
    return kr


# ─── Загрузка данных ─────────────────────────────────────────────────────────
def load_data():
    rv   = pd.read_csv(RV_FILE,   parse_dates=["date"])
    sent = pd.read_csv(SENT_FILE, parse_dates=["date"])
    gld  = pd.read_csv(GLD_PRED,  parse_dates=["date"])
    slv  = pd.read_csv(SLV_PRED,  parse_dates=["date"])
    summary = pd.read_csv(ML_SUMMARY)
    return rv, sent, gld, slv, summary


# ─── HTML-шаблон ─────────────────────────────────────────────────────────────
BASE_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MOEX Metals Volatility Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body { background: #0f1117; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
    .navbar { background: #1a1d27 !important; border-bottom: 1px solid #2d3148; }
    .navbar-brand { color: #f0b429 !important; font-weight: 700; letter-spacing: .5px; }
    .nav-link { color: #aab0c6 !important; }
    .nav-link.active, .nav-link:hover { color: #f0b429 !important; }
    .card {
      background: #1a1d27;
      border: 1px solid #2d3148;
      border-radius: 12px;
    }
    .card-header {
      background: #20243a;
      border-bottom: 1px solid #2d3148;
      border-radius: 12px 12px 0 0 !important;
      font-weight: 600;
      color: #c8d0e8;
    }
    .metric-card {
      background: #1a1d27;
      border: 1px solid #2d3148;
      border-radius: 10px;
      padding: 18px 22px;
    }
    .metric-val { font-size: 2rem; font-weight: 700; }
    .metric-lbl { font-size: .82rem; color: #7a84a0; text-transform: uppercase; letter-spacing: .5px; }
    .gold  { color: #f0b429; }
    .silver{ color: #8da8c8; }
    .green { color: #34d399; }
    .red   { color: #f87171; }
    .badge-pos { background: #1a3a2a; color: #34d399; border-radius: 6px; padding: 3px 10px; font-size:.8rem; }
    .badge-neg { background: #3a1a1a; color: #f87171; border-radius: 6px; padding: 3px 10px; font-size:.8rem; }
    footer { color: #444; font-size: .8rem; margin-top: 40px; padding-bottom: 20px; }
  </style>
</head>
<body>

<nav class="navbar navbar-expand-lg navbar-dark sticky-top">
  <div class="container-fluid px-4">
    <a class="navbar-brand" href="/">⚡ MOEX Metals Vol</a>
    <div class="collapse navbar-collapse">
      <ul class="navbar-nav me-auto">
        <li class="nav-item"><a class="nav-link {% if page=='home' %}active{% endif %}" href="/">Обзор</a></li>
        <li class="nav-item"><a class="nav-link {% if page=='vol' %}active{% endif %}" href="/volatility">Волатильность</a></li>
        <li class="nav-item"><a class="nav-link {% if page=='sent' %}active{% endif %}" href="/sentiment">Sentiment</a></li>
      </ul>
      <span class="text-muted" style="font-size:.8rem">Данные: MOEX ISS · GDELT · Telegram · FRED</span>
    </div>
  </div>
</nav>

<div class="container-fluid px-4 mt-4">
  {% block content %}{% endblock %}
</div>

<footer class="container-fluid px-4 text-center">
  HSE Course Project 2026 · Лапшин Владислав · Прогнозирование волатильности металлов MOEX
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

INDEX_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<!-- Метрики -->
<div class="row g-3 mb-4">
  {% for m in metrics %}
  <div class="col-6 col-md-3">
    <div class="metric-card h-100">
      <div class="metric-lbl mb-1">{{ m.label }}</div>
      <div class="metric-val {{ m.cls }}">{{ m.value }}</div>
      {% if m.badge %}
        <span class="{{ 'badge-pos' if m.positive else 'badge-neg' }}">{{ m.badge }}</span>
      {% endif %}
    </div>
  </div>
  {% endfor %}
</div>

<!-- Два графика рядом -->
<div class="row g-3 mb-4">
  <div class="col-12 col-lg-6">
    <div class="card">
      <div class="card-header">🥇 Золото GLDRUB_TOM — Реализованная волатильность</div>
      <div class="card-body p-2"><div id="chartGld" style="height:320px"></div></div>
    </div>
  </div>
  <div class="col-12 col-lg-6">
    <div class="card">
      <div class="card-header">🥈 Серебро SLVRUB_TOM — Реализованная волатильность</div>
      <div class="card-body p-2"><div id="chartSlv" style="height:320px"></div></div>
    </div>
  </div>
</div>

<!-- Sentiment overview -->
<div class="row g-3">
  <div class="col-12">
    <div class="card">
      <div class="card-header">📡 Комбинированный sentiment-индекс (RSS + GDELT + Telegram)</div>
      <div class="card-body p-2"><div id="chartSent" style="height:260px"></div></div>
    </div>
  </div>
</div>

<script>
const layout = (title) => ({
  paper_bgcolor:'transparent', plot_bgcolor:'transparent',
  font:{color:'#aab0c6', size:11},
  margin:{l:50,r:20,t:10,b:40},
  xaxis:{gridcolor:'#2d3148', showgrid:true},
  yaxis:{gridcolor:'#2d3148', showgrid:true},
  legend:{bgcolor:'transparent'},
  hovermode:'x unified',
  showlegend:true
});
const cfg = {responsive:true, displayModeBar:false};

// Gold
const gldData = {{ gld_rv | tojson }};
Plotly.newPlot('chartGld', [
  {x:gldData.date, y:gldData.rv_ann, name:'RV (ann.)', type:'scatter',
   mode:'lines', line:{color:'#f0b429', width:1.2}, fill:'tozeroy', fillcolor:'rgba(240,180,41,0.08)'},
  {x:gldData.date, y:gldData.ma30,   name:'MA(30)',   type:'scatter',
   mode:'lines', line:{color:'#fff', width:1.8}, opacity:.7}
], layout(), cfg);

// Silver
const slvData = {{ slv_rv | tojson }};
Plotly.newPlot('chartSlv', [
  {x:slvData.date, y:slvData.rv_ann, name:'RV (ann.)', type:'scatter',
   mode:'lines', line:{color:'#8da8c8', width:1.2}, fill:'tozeroy', fillcolor:'rgba(141,168,200,0.08)'},
  {x:slvData.date, y:slvData.ma30,   name:'MA(30)',   type:'scatter',
   mode:'lines', line:{color:'#fff', width:1.8}, opacity:.7}
], layout(), cfg);

// Sentiment
const sentData = {{ sent_overview | tojson }};
Plotly.newPlot('chartSent', [
  {x:sentData.date, y:sentData.raw, name:'Sentiment', type:'scatter',
   mode:'lines', line:{color:'#4a90d9', width:.8}, fill:'tozeroy', fillcolor:'rgba(74,144,217,0.1)'},
  {x:sentData.date, y:sentData.ma,  name:'MA(30)',    type:'scatter',
   mode:'lines', line:{color:'#60a5fa', width:2}}
], {...layout(), shapes:[
  {type:'line', x0:'2022-02-24', x1:'2022-02-24', y0:0, y1:1, yref:'paper',
   line:{color:'#f87171', dash:'dash', width:1.5}}
]}, cfg);
</script>
""")

VOL_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="row g-3 mb-4">
  <div class="col-12">
    <div class="card">
      <div class="card-header">🥇 Золото GLDRUB_TOM — XGBoost: прогноз vs факт (log-RV, тест 40%)</div>
      <div class="card-body p-2"><div id="chartGldFc" style="height:360px"></div></div>
    </div>
  </div>
</div>
<div class="row g-3 mb-4">
  <div class="col-12">
    <div class="card">
      <div class="card-header">🥈 Серебро SLVRUB_TOM — XGBoost: прогноз vs факт (log-RV, тест 40%)</div>
      <div class="card-body p-2"><div id="chartSlvFc" style="height:360px"></div></div>
    </div>
  </div>
</div>
<div class="row g-3">
  {% for m in metrics %}
  <div class="col-6 col-md-3">
    <div class="metric-card">
      <div class="metric-lbl">{{ m.label }}</div>
      <div class="metric-val {{ m.cls }}">{{ m.value }}</div>
    </div>
  </div>
  {% endfor %}
</div>

<script>
const layout = () => ({
  paper_bgcolor:'transparent', plot_bgcolor:'transparent',
  font:{color:'#aab0c6', size:11},
  margin:{l:50,r:20,t:10,b:40},
  xaxis:{gridcolor:'#2d3148'},
  yaxis:{gridcolor:'#2d3148', title:'log(RV)'},
  hovermode:'x unified', showlegend:true,
  legend:{bgcolor:'transparent'}
});
const cfg = {responsive:true, displayModeBar:false};

const gldFc = {{ gld_fc | tojson }};
Plotly.newPlot('chartGldFc', [
  {x:gldFc.date, y:gldFc.actual,    name:'Факт',    type:'scatter',
   mode:'lines', line:{color:'#f0b429', width:1.5}},
  {x:gldFc.date, y:gldFc.predicted, name:'XGBoost', type:'scatter',
   mode:'lines', line:{color:'#f87171', width:1.5, dash:'dot'}}
], layout(), cfg);

const slvFc = {{ slv_fc | tojson }};
Plotly.newPlot('chartSlvFc', [
  {x:slvFc.date, y:slvFc.actual,    name:'Факт',    type:'scatter',
   mode:'lines', line:{color:'#8da8c8', width:1.5}},
  {x:slvFc.date, y:slvFc.predicted, name:'XGBoost', type:'scatter',
   mode:'lines', line:{color:'#f87171', width:1.5, dash:'dot'}}
], layout(), cfg);
</script>
""")

SENT_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="row g-3 mb-4">
  <div class="col-12">
    <div class="card">
      <div class="card-header">📰 Комбинированный sentiment (RSS + GDELT + Telegram)</div>
      <div class="card-body p-2"><div id="chartComb" style="height:280px"></div></div>
    </div>
  </div>
</div>
<div class="row g-3 mb-4">
  <div class="col-12 col-lg-6">
    <div class="card">
      <div class="card-header">✈️ Telegram sentiment (взвешенный по просмотрам)</div>
      <div class="card-body p-2"><div id="chartTg" style="height:260px"></div></div>
    </div>
  </div>
  <div class="col-12 col-lg-6">
    <div class="card">
      <div class="card-header">🔍 Google Trends — внимание инвесторов (metals_ru)</div>
      <div class="card-body p-2"><div id="chartGt" style="height:260px"></div></div>
    </div>
  </div>
</div>
<div class="row g-3">
  <div class="col-12">
    <div class="card">
      <div class="card-header">🌍 Макро: USD/RUB · Ключевая ставка ЦБ РФ · Brent · VIX</div>
      <div class="card-body p-2"><div id="chartMacro" style="height:280px"></div></div>
    </div>
  </div>
</div>

<script>
const vline = (x) => ({
  type:'line', x0:x, x1:x, y0:0, y1:1, yref:'paper',
  line:{color:'#f87171', dash:'dash', width:1.5}
});
const layout = (extra={}) => ({
  paper_bgcolor:'transparent', plot_bgcolor:'transparent',
  font:{color:'#aab0c6', size:11},
  margin:{l:55,r:20,t:10,b:40},
  xaxis:{gridcolor:'#2d3148'},
  yaxis:{gridcolor:'#2d3148'},
  hovermode:'x unified', showlegend:true,
  legend:{bgcolor:'transparent'},
  shapes:[vline('2022-02-24')],
  ...extra
});
const cfg = {responsive:true, displayModeBar:false};

const s = {{ sent | tojson }};

// Combined sentiment
Plotly.newPlot('chartComb', [
  {x:s.date, y:s.sentiment_combined, name:'Sentiment', type:'scatter',
   mode:'lines', line:{color:'#4a90d9', width:.9}, fill:'tozeroy', fillcolor:'rgba(74,144,217,0.1)'},
  {x:s.date, y:s.sent_ma, name:'MA(30)', type:'scatter',
   mode:'lines', line:{color:'#60a5fa', width:2}}
], layout(), cfg);

// Telegram
Plotly.newPlot('chartTg', [
  {x:s.date, y:s.tg_score, name:'tg_score', type:'scatter',
   mode:'lines', line:{color:'#34d399', width:.9}, fill:'tozeroy', fillcolor:'rgba(52,211,153,0.08)'},
  {x:s.date, y:s.tg_ma, name:'MA(30)', type:'scatter',
   mode:'lines', line:{color:'#6ee7b7', width:2}}
], layout(), cfg);

// Google Trends
Plotly.newPlot('chartGt', [
  {x:s.date, y:s.attention_google, name:'Google Trends', type:'scatter',
   mode:'lines', line:{color:'#fb923c', width:.9}, fill:'tozeroy', fillcolor:'rgba(251,146,60,0.08)'},
  {x:s.date, y:s.gt_ma, name:'MA(30)', type:'scatter',
   mode:'lines', line:{color:'#fdba74', width:2}}
], layout(), cfg);

// Macro: dual axis
const m = {{ macro | tojson }};
Plotly.newPlot('chartMacro', [
  {x:m.date, y:m.macro_usd_rub,  name:'USD/RUB',    yaxis:'y',  type:'scatter', mode:'lines', line:{color:'#f0b429',width:1.5}},
  {x:m.date, y:m.macro_brent,    name:'Brent $',    yaxis:'y2', type:'scatter', mode:'lines', line:{color:'#60a5fa',width:1.5}},
  {x:m.date, y:m.macro_vix,      name:'VIX',        yaxis:'y2', type:'scatter', mode:'lines', line:{color:'#f87171',width:1.5,dash:'dot'}},
  {x:m.date, y:m.macro_key_rate, name:'Ставка ЦБ%', yaxis:'y2', type:'scatter', mode:'lines', line:{color:'#a78bfa',width:1.5,dash:'dash'}}
], layout({
  yaxis:{gridcolor:'#2d3148', title:'USD/RUB'},
  yaxis2:{overlaying:'y', side:'right', title:'USD / %', gridcolor:'transparent'}
}), cfg);
</script>
""")


# ─── Роуты ───────────────────────────────────────────────────────────────────

def _nan_to_none(lst):
    return [None if (v is not None and isinstance(v, float) and np.isnan(v)) else v for v in lst]

def _series(s: pd.Series):
    return _nan_to_none(s.tolist())

def _dates(s: pd.Series):
    return s.dt.strftime("%Y-%m-%d").tolist()


@app.route("/")
def index():
    rv, sent, gld_fc, slv_fc, summary = load_data()

    gld_rv = rv[rv["secid"] == "GLDRUB_TOM"].copy()
    slv_rv = rv[rv["secid"] == "SLVRUB_TOM"].copy()

    gld_rv_data = {
        "date":   _dates(gld_rv["date"]),
        "rv_ann": _series(gld_rv["rv_ann"]),
        "ma30":   _series(gld_rv["rv_ann"].rolling(30, min_periods=1).mean()),
    }
    slv_rv_data = {
        "date":   _dates(slv_rv["date"]),
        "rv_ann": _series(slv_rv["rv_ann"]),
        "ma30":   _series(slv_rv["rv_ann"].rolling(30, min_periods=1).mean()),
    }
    sent_ov = {
        "date": _dates(sent["date"]),
        "raw":  _series(sent["sentiment_combined"]),
        "ma":   _series(sent["sentiment_combined"].rolling(30, min_periods=1).mean()),
    }

    # Метрики
    gld_m = summary[summary["secid"] == "GLDRUB_TOM"].iloc[0]
    slv_m = summary[summary["secid"] == "SLVRUB_TOM"].iloc[0]
    cur_gld_rv = float(gld_rv["rv_ann"].iloc[-1])
    cur_slv_rv = float(slv_rv["rv_ann"].iloc[-1])

    metrics = [
        {"label": "RV Золото (посл.)",  "value": f"{cur_gld_rv:.3f}",       "cls": "gold",   "badge": None, "positive": True},
        {"label": "RV Серебро (посл.)", "value": f"{cur_slv_rv:.3f}",       "cls": "silver", "badge": None, "positive": True},
        {"label": "R² OOS Золото",      "value": f"{float(gld_m['R2_OOS']):.3f}",
         "cls": "green" if float(gld_m["R2_OOS"]) > 0 else "red",
         "badge": "XGBoost", "positive": float(gld_m["R2_OOS"]) > 0},
        {"label": "R² OOS Серебро",     "value": f"{float(slv_m['R2_OOS']):.3f}",
         "cls": "green" if float(slv_m["R2_OOS"]) > 0 else "red",
         "badge": "XGBoost", "positive": float(slv_m["R2_OOS"]) > 0},
    ]

    return render_template_string(
        INDEX_HTML,
        page="home",
        gld_rv=gld_rv_data,
        slv_rv=slv_rv_data,
        sent_overview=sent_ov,
        metrics=metrics,
    )


@app.route("/volatility")
def volatility():
    rv, sent, gld_fc, slv_fc, summary = load_data()

    gld_fc_data = {"date": _dates(gld_fc["date"]),
                   "actual": _series(gld_fc["actual"]),
                   "predicted": _series(gld_fc["predicted"])}
    slv_fc_data = {"date": _dates(slv_fc["date"]),
                   "actual": _series(slv_fc["actual"]),
                   "predicted": _series(slv_fc["predicted"])}

    gld_m = summary[summary["secid"] == "GLDRUB_TOM"].iloc[0]
    slv_m = summary[summary["secid"] == "SLVRUB_TOM"].iloc[0]

    metrics = [
        {"label": "MAE Золото",   "value": f"{float(gld_m['MAE']):.3f}",  "cls": "gold"},
        {"label": "RMSE Золото",  "value": f"{float(gld_m['RMSE']):.3f}", "cls": "gold"},
        {"label": "MAE Серебро",  "value": f"{float(slv_m['MAE']):.3f}",  "cls": "silver"},
        {"label": "RMSE Серебро", "value": f"{float(slv_m['RMSE']):.3f}", "cls": "silver"},
        {"label": "R² OOS Золото",  "value": f"{float(gld_m['R2_OOS']):.3f}",
         "cls": "green" if float(gld_m["R2_OOS"]) > 0 else "red"},
        {"label": "QLIKE Золото", "value": f"{float(gld_m['QLIKE']):.3f}", "cls": "gold"},
        {"label": "R² OOS Серебро", "value": f"{float(slv_m['R2_OOS']):.3f}",
         "cls": "green" if float(slv_m["R2_OOS"]) > 0 else "red"},
        {"label": "QLIKE Серебро","value": f"{float(slv_m['QLIKE']):.3f}", "cls": "silver"},
    ]

    return render_template_string(
        VOL_HTML, page="vol",
        gld_fc=gld_fc_data, slv_fc=slv_fc_data,
        metrics=metrics,
    )


@app.route("/sentiment")
def sentiment():
    _, sent, _, _, _ = load_data()

    sent_data = {
        "date":               _dates(sent["date"]),
        "sentiment_combined": _series(sent["sentiment_combined"]),
        "sent_ma":            _series(sent["sentiment_combined"].rolling(30, min_periods=1).mean()),
        "tg_score":           _series(sent["tg_score"].fillna(0)),
        "tg_ma":              _series(sent["tg_score"].fillna(0).rolling(30, min_periods=1).mean()),
        "attention_google":   _series(sent["attention_google"].fillna(0)),
        "gt_ma":              _series(sent["attention_google"].fillna(0).rolling(30, min_periods=1).mean()),
    }
    # Ключевая ставка: берём из CSV (если fix_key_rate.py уже запущен),
    # иначе строим из встроенной истории решений ЦБ РФ.
    kr_csv = sent["macro_key_rate"].ffill()
    key_rate_series = (
        kr_csv if kr_csv.dropna().nunique() > 1
        else _build_key_rate_series(sent["date"])
    )

    macro_data = {
        "date":           _dates(sent["date"]),
        "macro_usd_rub":  _series(sent["macro_usd_rub"].ffill()),
        "macro_brent":    _series(sent["macro_brent"].ffill()),
        "macro_vix":      _series(sent["macro_vix"].ffill()),
        "macro_key_rate": _series(key_rate_series),
    }

    return render_template_string(
        SENT_HTML, page="sent",
        sent=sent_data,
        macro=macro_data,
    )


# ─── Запуск ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MOEX Metals Volatility Dashboard")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, port=5000)

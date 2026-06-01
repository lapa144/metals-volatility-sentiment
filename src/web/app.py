from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from datetime import date, datetime, timedelta
from flask import Flask, render_template, request, Response

from services import (
    MoexError,
    MoexIssClient,
    SECIDS,
    INTERVAL_MAP,
    fetch_candles_by_dates,
    plot_close_png,
    validate_secid,
    parse_interval_label,
)


app = Flask(__name__)


@app.get("/")
def index():
    today = date.today()
    default_from = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    default_till = today.strftime("%Y-%m-%d")
    return render_template(
        "index.html",
        secids=SECIDS,
        default_secid="GLDRUB_TOM",
        default_date_from=default_from,
        default_date_till=default_till,
        default_interval="1d",
        intervals=list(INTERVAL_MAP.keys()),
    )


@app.post("/run")
def run():
    secid = (request.form.get("secid") or "").strip()
    date_from = (request.form.get("date_from") or "").strip()
    date_till = (request.form.get("date_till") or "").strip()
    interval_label = (request.form.get("interval") or "1d").strip()

    if not date_from or not date_till:
        return render_template("error.html", message="Укажите обе даты: с и по"), 400

    try:
        secid = validate_secid(secid)
        parse_interval_label(interval_label)
    except MoexError as e:
        return render_template("error.html", message=str(e)), 400

    try:
        datetime.strptime(date_from, "%Y-%m-%d")
        datetime.strptime(date_till, "%Y-%m-%d")
    except ValueError:
        return render_template("error.html", message="Неверный формат даты. Используйте ГГГГ-ММ-ДД"), 400

    try:
        candles = fetch_candles_by_dates(secid, date_from, date_till, interval_label)
    except MoexError as e:
        return render_template("error.html", message=str(e)), 400

    interval_int = INTERVAL_MAP.get(interval_label, 24)

    # Таблица последних 50 свечей
    table_rows = []
    for c in candles[-50:]:
        table_rows.append({
            "begin": c.begin.strftime("%Y-%m-%d %H:%M"),
            "open": f"{c.open:.2f}",
            "high": f"{c.high:.2f}",
            "low": f"{c.low:.2f}",
            "close": f"{c.close:.2f}",
            "volume": f"{c.volume:.0f}" if c.volume is not None else "—",
            "value": f"{c.value:.0f}" if c.value is not None else "—",
        })

    # URL для графика
    plot_url = (
        f"/plot?secid={secid}&from={date_from}&till={date_till}&interval={interval_int}"
    )

    return render_template(
        "result.html",
        secid=secid,
        secid_name=SECIDS.get(secid, secid),
        date_from=date_from,
        date_till=date_till,
        interval=interval_label,
        n=len(candles),
        table_rows=table_rows,
        plot_url=plot_url,
        has_data=len(candles) > 0,
    )


@app.get("/plot")
def plot():
    secid = request.args.get("secid", "").strip()
    date_from = request.args.get("from", "")
    date_till = request.args.get("till", "")
    interval_raw = request.args.get("interval", "24")

    try:
        secid = validate_secid(secid)
        interval_int = int(interval_raw)
        if interval_int not in INTERVAL_MAP.values():
            interval_int = 24
    except (MoexError, ValueError):
        return _empty_plot_response()

    if not date_from or not date_till:
        return _empty_plot_response()

    try:
        client = MoexIssClient()
        candles = client.fetch_candles(secid, date_from, date_till, interval_int)
    except MoexError:
        return _empty_plot_response()

    png_bytes = plot_close_png(candles)
    return Response(png_bytes, mimetype="image/png")


def _empty_plot_response() -> Response:
    """Возвращает PNG с текстом «Нет данных»."""
    png_bytes = plot_close_png([])
    return Response(png_bytes, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, port=5001)

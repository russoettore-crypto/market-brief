#!/usr/bin/env python3
"""
Market Brief — report giornaliero automatico di mercati + sentiment notizie,
con track record del segnale e calcolatore di size sui tuoi parametri di rischio.
Fonti: Google News RSS + Yahoo Finance (nessuna registrazione) — sentiment via FinBERT
chiamato online tramite Hugging Face Inference API (richiede un token HF gratuito
in variabile d'ambiente HF_TOKEN, nessun modello scaricato su questa macchina).
Uso: python3 main.py  ->  scrive docs/index.html, docs/archive/<data>.html
                          e docs/archive/data/<data>.json + track_record.json
"""
import glob
import html
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import feedparser
import requests
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
ARCHIVE_DIR = os.path.join(DOCS_DIR, "archive")
DATA_DIR = os.path.join(ARCHIVE_DIR, "data")
TRACK_RECORD_PATH = os.path.join(DATA_DIR, "track_record.json")
DISPLAY_TZ = ZoneInfo("Europe/Rome")

# Sentiment via Hugging Face Inference API (FinBERT gira sui server HF, non in locale:
# nessun modello scaricato qui). Serve un token gratuito, vedi README.
HF_MODEL = "ProsusAI/finbert"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ---------- Raccolta dati ----------

class ConfigError(Exception):
    """Sollevato quando watchlist.json manca di qualcosa — con un messaggio leggibile,
    non uno stack trace, perché l'errore più probabile è una modifica manuale sbagliata."""


def validate_config(config):
    errors = []

    def require(cond, msg):
        if not cond:
            errors.append(msg)

    require(isinstance(config.get("watchlist"), list) and config["watchlist"],
            "'watchlist' deve essere una lista non vuota")
    for i, item in enumerate(config.get("watchlist", [])):
        for key in ("ticker", "name", "query"):
            require(isinstance(item.get(key), str) and item[key],
                    f"watchlist[{i}] ({item.get('ticker', '?')}): manca o è vuoto il campo '{key}'")

    require(isinstance(config.get("indices"), list), "'indices' deve essere una lista")
    for i, item in enumerate(config.get("indices", [])):
        for key in ("ticker", "name"):
            require(isinstance(item.get(key), str) and item[key],
                    f"indices[{i}]: manca o è vuoto il campo '{key}'")

    require(isinstance(config.get("macro_queries"), list) and config["macro_queries"],
            "'macro_queries' deve essere una lista non vuota")
    require(isinstance(config.get("news_per_item"), int) and config["news_per_item"] > 0,
            "'news_per_item' deve essere un numero intero positivo")

    lang = config.get("language", {})
    for key in ("hl", "gl", "ceid"):
        require(isinstance(lang.get(key), str) and lang[key], f"'language.{key}' mancante")

    risk = config.get("risk_settings")
    if risk is not None:
        for key in ("account_size", "risk_per_trade_pct", "default_stop_loss_pct"):
            require(isinstance(risk.get(key), (int, float)) and risk[key] > 0,
                    f"'risk_settings.{key}' deve essere un numero positivo")

    if errors:
        raise ConfigError("watchlist.json non è valido:\n  - " + "\n  - ".join(errors))


def load_config():
    path = os.path.join(BASE_DIR, "watchlist.json")
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Non trovo watchlist.json in {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"watchlist.json ha un errore di sintassi JSON: {e}") from e
    validate_config(config)
    return config


def fetch_news(query, max_items, lang_cfg):
    """Interroga Google News RSS (nessuna API key richiesta) per una query, ultime 24h."""
    q = urllib.parse.quote(f"{query} when:1d")
    url = (
        f"https://news.google.com/rss/search?q={q}"
        f"&hl={lang_cfg['hl']}&gl={lang_cfg['gl']}&ceid={lang_cfg['ceid']}"
    )
    items = []
    try:
        feed = feedparser.parse(url)
        seen_titles = set()
        for entry in feed.entries[: max_items * 2]:
            title = entry.get("title", "").strip()
            if not title:
                continue
            norm = title.lower()[:60]
            if norm in seen_titles:
                continue
            seen_titles.add(norm)
            source = ""
            if "source" in entry and hasattr(entry.source, "title"):
                source = entry.source.title
            items.append({
                "title": title,
                "link": entry.get("link", "#"),
                "source": source or "Fonte sconosciuta",
            })
            if len(items) >= max_items:
                break
    except Exception as e:
        print(f"  [warn] news fetch fallita per '{query}': {e}", file=sys.stderr)
    return items


def sentiment_score(text):
    """
    Sentiment -1..+1 (positivo - negativo) via FinBERT, chiamato ONLINE tramite la
    Hugging Face Inference API — nessun modello scaricato o eseguito su questa macchina.
    Se il token manca o l'API non risponde dopo i tentativi, ritorna 0.0 (neutro) e
    logga un avviso: un report parziale è meglio di un report che si blocca.
    """
    if not HF_TOKEN:
        print("  [warn] HF_TOKEN non impostato: sentiment neutro per questo testo", file=sys.stderr)
        return 0.0

    headers = {"Authorization": f"Bearer {HF_TOKEN}", "x-wait-for-model": "true"}
    for attempt in range(3):
        try:
            resp = requests.post(HF_API_URL, headers=headers,
                                  json={"inputs": text[:512]}, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data and isinstance(data[0], list):
                    data = data[0]  # alcune risposte arrivano annidate: [[{...}]]
                scores = {d["label"].lower(): d["score"] for d in data
                          if isinstance(d, dict) and "label" in d and "score" in d}
                return scores.get("positive", 0.0) - scores.get("negative", 0.0)
            if resp.status_code in (503, 429):  # modello in caricamento / rate limit
                time.sleep(5 * (attempt + 1))
                continue
            print(f"  [warn] Hugging Face API errore {resp.status_code}", file=sys.stderr)
            return 0.0
        except Exception as e:
            print(f"  [warn] Hugging Face API fallita: {e}", file=sys.stderr)
            time.sleep(3)
    return 0.0


def sentiment_label(score):
    if score >= 0.15:
        return "Positivo"
    if score <= -0.15:
        return "Negativo"
    return "Misto"


def fetch_ticker_metrics(ticker, atr_period=14):
    """
    Un'unica chiamata a Yahoo Finance per titolo: ultimo prezzo, variazione % di oggi,
    e ATR (Average True Range, media mobile semplice) come % del prezzo — una misura
    di volatilità recente, usata per uno stop loss su misura invece di un numero fisso
    identico per Bitcoin e per un titolo blue-chip.
    Ritorna (prezzo, variazione_%, atr_%) — ciascuno None se non calcolabile.
    """
    try:
        hist = yf.Ticker(ticker).history(period="2mo", interval="1d")
        closes = hist["Close"].dropna()
        if len(closes) < 2:
            return None, None, None
        last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
        pct = (last - prev) / prev * 100 if prev else None

        atr_pct = None
        ohlc = hist.dropna(subset=["High", "Low", "Close"])
        if len(ohlc) >= atr_period + 1:
            highs, lows, cl = ohlc["High"].values, ohlc["Low"].values, ohlc["Close"].values
            trs = [max(highs[i] - lows[i], abs(highs[i] - cl[i - 1]), abs(lows[i] - cl[i - 1]))
                   for i in range(1, len(ohlc))]
            atr = sum(trs[-atr_period:]) / atr_period
            atr_pct = (atr / last * 100) if last else None

        return last, pct, atr_pct
    except Exception as e:
        print(f"  [warn] dati falliti per '{ticker}': {e}", file=sys.stderr)
        return None, None, None


# ---------- Track record (pagella reale del tool, non un backtest) ----------
# Non esiste un archivio gratuito di notizie storiche, quindi non possiamo simulare
# "cosa avrebbe detto il sentiment 6 mesi fa". Quello che possiamo fare onestamente è
# registrare ogni segnale reale e, N esecuzioni dopo, controllare cosa ha fatto davvero
# il prezzo. Statistiche vere, costruite giorno per giorno, non previsioni.

def build_snapshot(data, date_str):
    return {
        "date": date_str,
        "watchlist": [
            {"ticker": w["ticker"], "price": w["price"], "sentiment": w["sentiment"], "label": w["label"]}
            for w in data["watchlist"]
        ],
    }


def save_snapshot(snapshot):
    with open(os.path.join(DATA_DIR, f"{snapshot['date']}.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False)


def load_snapshots():
    snapshots = []
    for fp in sorted(glob.glob(os.path.join(DATA_DIR, "20*.json"))):
        try:
            with open(fp, encoding="utf-8") as f:
                snapshots.append(json.load(f))
        except Exception as e:
            print(f"  [warn] snapshot illeggibile '{fp}': {e}", file=sys.stderr)
    return snapshots


# ---------- Trend del sentiment (usa l'archivio che già salviamo ogni giorno) ----------

def load_sentiment_history(snapshots, lookback=5):
    """ticker -> lista di sentiment score delle ultime esecuzioni PRECEDENTI a oggi."""
    history = {}
    for snap in snapshots[-lookback:]:
        for w in snap["watchlist"]:
            if w["sentiment"] is not None:
                history.setdefault(w["ticker"], []).append(w["sentiment"])
    return history


def trend_indicator(ticker, today_score, history):
    past = history.get(ticker, [])
    if len(past) < 2:
        return None  # non c'è ancora abbastanza storia per dire qualcosa di sensato
    diff = today_score - (sum(past) / len(past))
    if diff >= 0.15:
        return {"arrow": "▲", "cls": "pos", "text": f"in miglioramento sulle ultime {len(past)} esecuzioni"}
    if diff <= -0.15:
        return {"arrow": "▼", "cls": "neg", "text": f"in peggioramento sulle ultime {len(past)} esecuzioni"}
    return {"arrow": "–", "cls": "flat", "text": f"stabile sulle ultime {len(past)} esecuzioni"}


def update_track_record(snapshots, horizon_runs):
    """Confronta ogni segnale passato con il prezzo N esecuzioni dopo e accumula il risultato."""
    try:
        with open(TRACK_RECORD_PATH, encoding="utf-8") as f:
            record = json.load(f)
    except Exception:
        record = {"entries": []}

    already = {(e["signal_date"], e["ticker"]) for e in record["entries"]}

    for i in range(len(snapshots) - horizon_runs):
        signal_snap, eval_snap = snapshots[i], snapshots[i + horizon_runs]
        eval_prices = {w["ticker"]: w["price"] for w in eval_snap["watchlist"] if w["price"] is not None}

        for w in signal_snap["watchlist"]:
            key = (signal_snap["date"], w["ticker"])
            if key in already or not w["price"] or w["label"] == "Misto":
                continue
            price_now = eval_prices.get(w["ticker"])
            if not price_now:
                continue
            ret = (price_now - w["price"]) / w["price"] * 100
            hit = (ret > 0) if w["label"] == "Positivo" else (ret < 0)
            record["entries"].append({
                "signal_date": signal_snap["date"], "eval_date": eval_snap["date"],
                "ticker": w["ticker"], "label": w["label"],
                "price_then": w["price"], "price_now": price_now,
                "return_pct": ret, "hit": hit,
            })
            already.add(key)

    with open(TRACK_RECORD_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    return record["entries"]


def compute_track_stats(entries):
    stats = {"total": len(entries)}
    for label in ("Positivo", "Negativo"):
        subset = [e for e in entries if e["label"] == label]
        n = len(subset)
        hits = sum(1 for e in subset if e["hit"])
        stats[label] = {
            "n": n,
            "hit_rate": (hits / n * 100) if n else None,
            "avg_return": (sum(e["return_pct"] for e in subset) / n) if n else None,
        }
    return stats


# ---------- Position sizing (aritmetica sui TUOI parametri, non una decisione del tool) ----------

def compute_position_size(price, risk_cfg, atr_pct=None):
    account = risk_cfg.get("account_size", 0)
    risk_pct = risk_cfg.get("risk_per_trade_pct", 0)
    multiplier = risk_cfg.get("atr_multiplier", 2.0)
    use_atr = risk_cfg.get("stop_loss_mode", "atr") == "atr"

    if use_atr and atr_pct:
        stop_pct, stop_source = atr_pct * multiplier, f"{multiplier}× ATR"
    else:
        stop_pct = risk_cfg.get("default_stop_loss_pct", 0)
        stop_source = "fisso (ATR non disponibile)" if use_atr else "fisso"

    if not price or not account or not stop_pct:
        return None

    risk_amount = account * (risk_pct / 100)
    position_value = risk_amount / (stop_pct / 100)
    capped = position_value > account
    position_value = min(position_value, account)
    return {
        "position_value": position_value,
        "position_pct": position_value / account * 100,
        "shares": position_value / price,
        "capped": capped,
        "stop_pct": stop_pct,
        "stop_source": stop_source,
    }


def build_report(config, snapshots_before=None):
    lang_cfg = config["language"]
    n_news = config["news_per_item"]
    history = load_sentiment_history(snapshots_before or [])

    print("Recupero indici e prezzi...")
    indices = []
    for item in config["indices"]:
        price, pct, _ = fetch_ticker_metrics(item["ticker"])
        indices.append({**item, "price": price, "pct": pct})

    print("Recupero notizie, prezzi e volatilità per la watchlist...")
    watchlist = []
    for item in config["watchlist"]:
        news = fetch_news(item["query"], n_news, lang_cfg)
        scores = [sentiment_score(n["title"]) for n in news]
        news_scored = list(zip(news, scores))  # evita di richiamare l'API due volte per lo stesso titolo
        avg_score = sum(scores) / len(scores) if scores else 0.0
        price, pct, atr_pct = fetch_ticker_metrics(item["ticker"])

        divergence = False
        if pct is not None:
            if pct <= -1.0 and avg_score >= 0.2:
                divergence = True
            elif pct >= 1.0 and avg_score <= -0.2:
                divergence = True

        top_news = max(news_scored, key=lambda ns: abs(ns[1]), default=(None, 0))[0]
        watchlist.append({
            **item, "price": price, "pct": pct, "atr_pct": atr_pct,
            "sentiment": avg_score, "label": sentiment_label(avg_score),
            "top_news": top_news, "divergence": divergence, "n_news": len(news),
            "trend": trend_indicator(item["ticker"], avg_score, history),
        })
    # ordina i più rilevanti (sentiment più marcato, positivo o negativo) in cima
    watchlist.sort(key=lambda w: abs(w["sentiment"]), reverse=True)

    print("Recupero notizie macro...")
    macro = []
    for query in config["macro_queries"]:
        news = fetch_news(query, n_news, lang_cfg)
        scores = [sentiment_score(n["title"]) for n in news]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        macro.append({"query": query, "news": news, "sentiment": avg_score,
                       "label": sentiment_label(avg_score)})

    all_scores = [w["sentiment"] for w in watchlist if w["n_news"] > 0] + \
                 [m["sentiment"] for m in macro if m["news"]]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "generated_at": datetime.now(timezone.utc),
        "indices": indices,
        "watchlist": watchlist,
        "macro": macro,
        "overall": overall,
        "overall_label": sentiment_label(overall),
    }


# ---------- Rendering HTML ----------

def esc(s):
    return html.escape(str(s), quote=True)


def fmt_pct(pct):
    if pct is None:
        return "—"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def pct_class(pct):
    if pct is None:
        return "flat"
    return "pos" if pct >= 0.05 else ("neg" if pct <= -0.05 else "flat")


def sent_class(label):
    return {"Positivo": "pos", "Negativo": "neg", "Misto": "flat"}.get(label, "flat")


def headline_takeaway(data):
    ov = data["overall_label"]
    n_div = sum(1 for w in data["watchlist"] if w["divergence"])
    if ov == "Positivo":
        base = "Il tono prevalente delle notizie è costruttivo."
    elif ov == "Negativo":
        base = "Il tono prevalente delle notizie è difensivo."
    else:
        base = "Il tono delle notizie è contrastato, senza una direzione chiara."
    if n_div:
        base += f" {n_div} titol{'o' if n_div == 1 else 'i'} in watchlist mostr{'a' if n_div==1 else 'ano'} una divergenza tra prezzo e sentiment — vale la pena guardarl{'o' if n_div==1 else 'i'} da vicino."
    return base


def render_ticker_strip(indices):
    chips = []
    for idx in indices:
        cls = pct_class(idx["pct"])
        price_txt = f"{idx['price']:,.2f}" if idx["price"] is not None else "—"
        chips.append(
            f'<div class="chip {cls}"><span class="chip-name">{esc(idx["name"])}</span>'
            f'<span class="chip-price">{esc(price_txt)}</span>'
            f'<span class="chip-pct">{esc(fmt_pct(idx["pct"]))}</span></div>'
        )
    return "".join(chips)


def render_watchlist_rows(watchlist):
    rows = []
    for w in watchlist:
        pcls = pct_class(w["pct"])
        scls = sent_class(w["label"])
        price_txt = f"{w['price']:,.2f}" if w["price"] is not None else "—"
        if w["top_news"]:
            news_html = (f'<a href="{esc(w["top_news"]["link"])}" target="_blank" rel="noopener">'
                         f'{esc(w["top_news"]["title"])}</a> '
                         f'<span class="src">— {esc(w["top_news"]["source"])}</span>')
        else:
            news_html = '<span class="src">nessuna notizia recente</span>'
        div_flag = '<span class="flag" title="Prezzo e sentiment divergono">⚠ divergenza</span>' if w["divergence"] else ""
        trend_html = ""
        if w.get("trend"):
            t = w["trend"]
            trend_html = f' <span class="trend {t["cls"]}" title="{esc(t["text"])}">{t["arrow"]}</span>'
        rows.append(f'''
        <tr>
          <td class="tk" data-label="Titolo">{esc(w["ticker"])}<div class="tk-name">{esc(w["name"])}</div></td>
          <td class="num" data-label="Prezzo">{esc(price_txt)}</td>
          <td class="num {pcls}" data-label="Var.">{esc(fmt_pct(w["pct"]))}</td>
          <td class="{scls}" data-label="Sentiment">{esc(w["label"])}{trend_html} {div_flag}</td>
          <td class="news" data-label="Notizia principale">{news_html}</td>
        </tr>''')
    return "".join(rows)


def render_macro_section(macro):
    blocks = []
    for m in macro:
        scls = sent_class(m["label"])
        items = "".join(
            f'<li><a href="{esc(n["link"])}" target="_blank" rel="noopener">{esc(n["title"])}</a> '
            f'<span class="src">— {esc(n["source"])}</span></li>'
            for n in m["news"]
        ) or "<li class='src'>Nessuna notizia trovata per questo tema oggi.</li>"
        blocks.append(f'''
        <div class="macro-block">
          <h3>{esc(m["query"])} <span class="tag {scls}">{esc(m["label"])}</span></h3>
          <ul>{items}</ul>
        </div>''')
    return "".join(blocks)


def render_archive_links():
    files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.html")), reverse=True)
    links = "".join(
        f'<a href="archive/{esc(os.path.basename(f))}">{esc(os.path.basename(f).replace(".html",""))}</a>'
        for f in files[:30]
    )
    return links or '<span class="src">Nessun report precedente ancora salvato.</span>'


def render_track_record_section(stats, horizon_runs):
    if stats["total"] == 0:
        return (f'<p class="src">Ancora nessun segnale valutato: il tool confronta ogni segnale con il prezzo '
                f'{horizon_runs} esecuzioni dopo, quindi servono almeno {horizon_runs + 1} run del workflow prima '
                f'del primo dato. La statistica si costruisce da sola, run dopo run — torna tra qualche settimana.</p>')

    cards = []
    for label, cls in (("Positivo", "pos"), ("Negativo", "neg")):
        s = stats[label]
        if not s["n"]:
            cards.append(f'<div class="tr-card"><div class="tr-label">{label}</div>'
                         f'<div class="src">nessun segnale valutato finora</div></div>')
            continue
        cards.append(f'''
        <div class="tr-card">
          <div class="tr-label">{label}</div>
          <div class="tr-stat {cls}">{s["hit_rate"]:.0f}%</div>
          <div class="tr-sub">direzione indovinata su {s["n"]} segnal{"e" if s["n"]==1 else "i"}</div>
          <div class="tr-sub">rendimento medio: {s["avg_return"]:+.2f}%</div>
        </div>''')

    warn = ""
    if stats["total"] < 30:
        warn = (f'<p class="src" style="margin-top:14px;">Campione ancora piccolo ({stats["total"]} segnali '
                'in totale): numeri indicativi, non statisticamente affidabili finché non crescono nel tempo.</p>')
    return f'<div class="tr-grid">{"".join(cards)}</div>{warn}'


def render_sizing_section(watchlist, risk_cfg):
    if not risk_cfg or not risk_cfg.get("account_size"):
        return '<p class="src">Imposta "risk_settings" in watchlist.json per attivare questa sezione.</p>'

    rows = []
    for w in watchlist:
        sz = compute_position_size(w["price"], risk_cfg, w.get("atr_pct"))
        if sz is None:
            continue
        cap = ' <span class="flag" title="Il calcolo puro richiederebbe leva; limitato al 100% del capitale">⚠ limitato</span>' if sz["capped"] else ""
        rows.append(f'''
        <tr>
          <td class="tk" data-label="Titolo">{esc(w["ticker"])}</td>
          <td class="num" data-label="Size">{sz["position_pct"]:.1f}% del capitale{cap}</td>
          <td class="num" data-label="Azioni">{sz["shares"]:.2f}</td>
          <td class="num" data-label="Stop ipotetico">{sz["stop_pct"]:.1f}%<div class="tk-name">{esc(sz["stop_source"])}</div></td>
          <td class="num" data-label="Controvalore">{esc(risk_cfg.get("currency","EUR"))} {sz["position_value"]:,.0f}</td>
        </tr>''')
    if not rows:
        return '<p class="src">Prezzi non disponibili in questa esecuzione per calcolare la size.</p>'

    return f'''
    <table class="size-table">
      <thead><tr><th>Titolo</th><th>Size</th><th>Azioni</th><th>Stop ipotetico</th><th>Controvalore</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    <p class="src" style="margin-top:12px;">
      Calcolo aritmetico sui TUOI parametri (capitale {esc(risk_cfg.get("currency","EUR"))} {risk_cfg["account_size"]:,.0f},
      rischio {risk_cfg["risk_per_trade_pct"]}% a operazione). Lo stop ipotetico usa l'ATR (volatilità reale
      delle ultime settimane) per ogni titolo quando disponibile, invece di un numero identico per tutti —
      altrimenti torna al valore fisso in "default_stop_loss_pct". Non è collegato al sentiment del giorno
      e non è un consiglio su quale titolo aprire. Modifica i parametri in watchlist.json con i tuoi numeri
      reali; sono placeholder finché non lo fai.
    </p>'''


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Brief — {date_title}</title>
<style>
  :root {{
    --board-bg: #14171A; --board-line: #262B31; --paper-bg: #FAF8F3; --paper-line: #E4E0D6;
    --ink: #1E2024; --ink-soft: #6B6A63; --ink-inv: #E9E7E0; --ink-inv-soft: #9AA0A6;
    --pos: #4FAE7D; --neg: #D9584F; --flat: #9AA0A6; --amber: #C68A2E;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--paper-bg); color:var(--ink);
    font-family:-apple-system,"Segoe UI",Roboto,sans-serif; -webkit-font-smoothing:antialiased; }}
  a {{ color:inherit; }}
  .wrap {{ max-width:920px; margin:0 auto; }}
  .board {{ background:var(--board-bg); color:var(--ink-inv); padding:28px 20px 22px; }}
  .eyebrow {{ font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-inv-soft); margin:0 0 10px; }}
  .headline {{ font-family:Georgia,"Iowan Old Style","Times New Roman",serif; font-size:26px;
    line-height:1.35; margin:0 0 20px; max-width:640px; font-weight:500; }}
  .strip {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:26px; }}
  .chip {{ font-family:"SF Mono","IBM Plex Mono",Consolas,"Courier New",monospace; font-size:12.5px;
    border:1px solid var(--board-line); border-radius:3px; padding:7px 10px; display:flex; gap:8px; align-items:baseline; }}
  .chip-name {{ color:var(--ink-inv-soft); }}
  .chip-price {{ color:var(--ink-inv); }}
  .chip.pos .chip-pct {{ color:var(--pos); }}
  .chip.neg .chip-pct {{ color:var(--neg); }}
  .chip.flat .chip-pct {{ color:var(--flat); }}
  table {{ width:100%; border-collapse:collapse; font-family:"SF Mono","IBM Plex Mono",Consolas,"Courier New",monospace; font-size:13px; }}
  .main-table {{ table-layout:fixed; }}
  .main-table th:nth-child(1), .main-table td:nth-child(1) {{ width:19%; }}
  .main-table th:nth-child(2), .main-table td:nth-child(2) {{ width:13%; }}
  .main-table th:nth-child(3), .main-table td:nth-child(3) {{ width:11%; }}
  .main-table th:nth-child(4), .main-table td:nth-child(4) {{ width:16%; }}
  .main-table th:nth-child(5), .main-table td:nth-child(5) {{ width:41%; }}
  td[data-label]::before {{ content:none; }}
  th {{ text-align:left; font-family:-apple-system,"Segoe UI",Roboto,sans-serif; font-size:11px; letter-spacing:.05em;
    text-transform:uppercase; color:var(--ink-inv-soft); font-weight:500; padding:0 10px 10px; border-bottom:1px solid var(--board-line); }}
  td {{ padding:12px 10px; border-bottom:1px solid var(--board-line); vertical-align:top; }}
  .tk {{ font-weight:600; white-space:nowrap; }}
  .tk-name {{ font-family:-apple-system,"Segoe UI",Roboto,sans-serif; color:var(--ink-inv-soft); font-size:11px; font-weight:400; }}
  .num {{ text-align:right; white-space:nowrap; }}
  .pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }} .flat {{ color:var(--flat); }}
  .news {{ font-family:-apple-system,"Segoe UI",Roboto,sans-serif; font-size:12.5px; color:var(--ink-inv); }}
  .news a {{ text-decoration:underline; text-decoration-color:var(--board-line); }}
  .src {{ color:var(--ink-inv-soft); }}
  .flag {{ font-size:11px; color:var(--amber); font-family:-apple-system,"Segoe UI",Roboto,sans-serif; margin-left:4px; }}
  .paper {{ padding:30px 20px 40px; }}
  .paper h2 {{ font-family:Georgia,"Iowan Old Style","Times New Roman",serif; font-size:19px; margin:0 0 16px; }}
  .paper th {{ color:var(--ink-soft); }}
  .paper .src, .paper .tk-name {{ color:var(--ink-soft); }}
  .trend {{ font-weight:700; margin-left:2px; }}
  .section {{ margin-bottom:36px; }}
  .tr-grid {{ display:flex; gap:14px; flex-wrap:wrap; }}
  .tr-card {{ flex:1; min-width:160px; border:1px solid var(--paper-line); border-radius:8px; padding:14px 16px; }}
  .tr-label {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--ink-soft); margin-bottom:8px; }}
  .tr-stat {{ font-family:"SF Mono","IBM Plex Mono",Consolas,"Courier New",monospace; font-size:30px; font-weight:600; line-height:1; margin-bottom:8px; }}
  .tr-sub {{ font-size:12px; color:var(--ink-soft); line-height:1.6; }}
  .size-table th, .size-table td {{ border-bottom:1px solid var(--paper-line); }}
  .macro-block {{ margin-bottom:22px; padding-bottom:18px; border-bottom:1px solid var(--paper-line); }}
  .macro-block:last-child {{ border-bottom:none; }}
  .macro-block h3 {{ font-size:14px; margin:0 0 8px; font-weight:600; }}
  .tag {{ font-size:10.5px; font-weight:500; padding:2px 7px; border-radius:10px; margin-left:6px; letter-spacing:.02em; }}
  .tag.pos {{ background:rgba(79,174,125,.15); color:#2F7B4F; }}
  .tag.neg {{ background:rgba(217,88,79,.15); color:#A63A2E; }}
  .tag.flat {{ background:rgba(154,160,166,.18); color:#6B6A63; }}
  .macro-block ul {{ margin:0; padding-left:18px; color:var(--ink-soft); font-size:13.5px; line-height:1.7; }}
  .macro-block a {{ text-decoration-color:var(--paper-line); }}
  .archive {{ font-size:12.5px; color:var(--ink-soft); }}
  .archive a {{ margin-right:12px; display:inline-block; text-decoration-color:var(--paper-line); }}
  footer {{ padding:20px 20px 40px; font-size:12px; color:var(--ink-soft); line-height:1.6; border-top:1px solid var(--paper-line); }}
  @media (max-width:640px) {{
    table, thead, tbody, th, td, tr {{ display:block; width:auto !important; }}
    thead {{ display:none; }}
    tr {{ padding:12px 0; border-bottom:1px solid var(--board-line); }}
    tbody tr:last-child {{ border-bottom:none; }}
    td {{ border:none; padding:3px 0; }}
    td[data-label]::before {{ content:attr(data-label) ": "; color:var(--ink-inv-soft);
      font-family:-apple-system,"Segoe UI",Roboto,sans-serif; font-size:11px; }}
    td.tk[data-label]::before {{ content:none; }}
    .num {{ text-align:left; }}
    .headline {{ font-size:22px; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="board">
    <p class="eyebrow">Market Brief &middot; {date_full}, {time_full} CET</p>
    <p class="headline">{takeaway}</p>
    <div class="strip">{ticker_strip}</div>
    <table class="main-table">
      <thead><tr><th>Titolo</th><th>Prezzo</th><th>Var.</th><th>Sentiment notizie</th><th>Notizia principale</th></tr></thead>
      <tbody>{watchlist_rows}</tbody>
    </table>
  </div>
  <div class="paper">
    <div class="section">
      <h2>Eventi macro di oggi</h2>
      {macro_section}
    </div>
    <div class="section">
      <h2>Track record del segnale</h2>
      {track_record_section}
    </div>
    <div class="section">
      <h2>Calcolatore di size</h2>
      {sizing_section}
    </div>
    <div class="section">
      <h2>Report precedenti</h2>
      <p class="archive">{archive_links}</p>
    </div>
  </div>
  <footer>
    Generato automaticamente da notizie pubbliche (Google News) e prezzi Yahoo Finance.
    Non è consulenza finanziaria: sentiment, track record e size sono informazioni e aritmetica,
    non un consiglio di acquisto o vendita — nessuna sezione di questa pagina ti dice cosa fare.
    Verifica sempre le fonti originali prima di ogni decisione.
  </footer>
</div>
</body>
</html>
"""


def render_html(data, track_stats, horizon_runs, risk_cfg):
    dt_local = data["generated_at"].astimezone(DISPLAY_TZ)
    return PAGE_TEMPLATE.format(
        date_title=dt_local.strftime("%d/%m/%Y"),
        date_full=dt_local.strftime("%A %d %B %Y"),
        time_full=dt_local.strftime("%H:%M"),
        takeaway=esc(headline_takeaway(data)),
        ticker_strip=render_ticker_strip(data["indices"]),
        watchlist_rows=render_watchlist_rows(data["watchlist"]),
        macro_section=render_macro_section(data["macro"]),
        track_record_section=render_track_record_section(track_stats, horizon_runs),
        sizing_section=render_sizing_section(data["watchlist"], risk_cfg),
        archive_links=render_archive_links(),
    )


def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    config = load_config()

    snapshots_before = load_snapshots()
    data = build_report(config, snapshots_before)

    date_str = data["generated_at"].astimezone(DISPLAY_TZ).strftime("%Y-%m-%d")
    snapshot = build_snapshot(data, date_str)
    save_snapshot(snapshot)

    print("Aggiorno il track record...")
    horizon = config.get("track_record_horizon_runs", 5)
    entries = update_track_record(snapshots_before + [snapshot], horizon)
    track_stats = compute_track_stats(entries)

    out_html = render_html(data, track_stats, horizon, config.get("risk_settings"))

    with open(os.path.join(DOCS_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(out_html)

    archive_name = date_str + ".html"
    with open(os.path.join(ARCHIVE_DIR, archive_name), "w", encoding="utf-8") as f:
        f.write(out_html)

    print(f"OK — report scritto in docs/index.html e docs/archive/{archive_name}")
    print(f"Track record: {track_stats['total']} segnali valutati finora.")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        print(f"\n❌ ERRORE DI CONFIGURAZIONE\n{e}\n", file=sys.stderr)
        sys.exit(1)

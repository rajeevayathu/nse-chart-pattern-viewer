#!/usr/bin/env python3.10
"""
Real-time price updater for the Minervini dashboard — India + US.

India (NSE, IST):
  • NSE bulk API (pre-open 9:00-9:15 AM) or yfinance 5m bars during market hours
  • Writes live_prices.json, patches india_data.json
  • Re-scans + regenerates charts every 30 min

US (NYSE/NASDAQ, ET):
  • yfinance 5m bars during US market hours (9:30 AM–4:00 PM ET)
  • Writes live_prices_us.json, patches us_data.json
  • Re-scans + regenerates US charts every 30 min

Usage:
  python3.10 price_updater.py            # normal (market-hours-gated)
  python3.10 price_updater.py --force    # run even outside market hours (testing)
"""

import json, os, sys, time, threading, subprocess, logging
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import pandas as pd
from nsepython import nsefetch

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR  = os.path.dirname(HERE)
SCANNER_DIR  = os.path.join(EXAMPLE_DIR, 'minervini-trend-template-scanner')

INDIA_DATA      = os.path.join(HERE, 'india_data.json')
US_DATA         = os.path.join(HERE, 'us_data.json')
LIVE_PRICES     = os.path.join(HERE, 'live_prices.json')
LIVE_PRICES_US  = os.path.join(HERE, 'live_prices_us.json')
RESULTS_JSON    = os.path.join(SCANNER_DIR, 'results.json')
SCANNER_PY      = os.path.join(SCANNER_DIR, 'scanner.py')
SCANNER_US_PY   = os.path.join(SCANNER_DIR, 'scanner_us.py')
GENcharts_PY    = os.path.join(HERE, 'generate_charts.py')
PYTHON          = sys.executable

# ── Config ────────────────────────────────────────────────────────────────────
PRICE_INTERVAL_S   = 60          # price fetch cadence (seconds)
RESCAN_INTERVAL_S  = 30 * 60    # full re-scan cadence (30 min)
BATCH_SIZE         = 200         # stocks per yfinance call
MAX_WORKERS        = 1           # sequential — yfinance rate-limits parallel calls
IST                = timezone(timedelta(hours=5, minutes=30))   # UTC+5:30
ET                 = timezone(timedelta(hours=-4))              # EDT (UTC-4); -5 in EST

# India (NSE) market hours in IST
MARKET_OPEN  = (9,  0)
MARKET_CLOSE = (16, 15)

# US (NYSE/NASDAQ) market hours in ET
US_MARKET_OPEN  = (9,  30)
US_MARKET_CLOSE = (16,  0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('price_updater')

# ── Helpers ───────────────────────────────────────────────────────────────────
def ist_now() -> datetime:
    return datetime.now(IST)

def et_now() -> datetime:
    return datetime.now(ET)

def market_open() -> bool:
    if '--force' in sys.argv:
        return True
    now = ist_now()
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t <= MARKET_CLOSE

def us_market_open() -> bool:
    if '--force' in sys.argv:
        return True
    now = et_now()
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return US_MARKET_OPEN <= t <= US_MARKET_CLOSE

def get_screen_tickers() -> list[str]:
    """Return unique tickers in india_data.json."""
    tickers = set()
    try:
        data = json.load(open(INDIA_DATA))
        for scr in data.values():
            for s in scr.get('stocks', []):
                t = s.get('ticker', '')
                if t:
                    tickers.add(t)
    except Exception:
        pass
    return sorted(tickers)

NSE_BULK_URL = "https://www.nseindia.com/api/market-data-pre-open?key=ALL"

def fetch_nse_bulk() -> dict:
    """
    Single NSE API call → prices for ~2200 stocks in <1s.
    Returns {ticker: {price, prev_close, change_abs, change_pct}}.
    This endpoint updates in near-real-time during market hours (~1 min lag).
    """
    result = {}
    try:
        data = nsefetch(NSE_BULK_URL)
        for item in data.get('data', []):
            meta = item.get('metadata', {})
            sym  = meta.get('symbol', '')
            p    = meta.get('lastPrice')
            pc   = meta.get('previousClose')
            if not sym or p is None:
                continue
            p  = float(p)
            pc = float(pc) if pc else p
            result[sym] = {
                'price':      round(p, 2),
                'prev_close': round(pc, 2),
                'change_abs': round(p - pc, 2),
                'change_pct': round((p - pc) / pc * 100, 2) if pc else 0.0,
            }
        log.info(f'NSE bulk: {len(result)} stocks in one call')
    except Exception as e:
        log.warning(f'NSE bulk fetch failed: {e}')
    return result

def _yf_batch(syms: list[str], interval: str, period: str) -> dict:
    """Download one batch from yfinance, return {sym: (price, prev_close)}."""
    out = {}
    try:
        df = yf.download(syms, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=True)
        if df.empty:
            return out
        close_df = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df[['Close']]
        if isinstance(close_df, pd.Series):
            close_df = close_df.to_frame(syms[0])
        for sym in syms:
            col = close_df.get(sym)
            if col is None:
                continue
            valid = col.dropna()
            if valid.empty:
                continue
            p  = float(valid.iloc[-1])
            pc = float(valid.iloc[-2]) if len(valid) >= 2 else p
            out[sym] = (p, pc)
    except Exception as e:
        log.warning(f'yfinance batch error: {e}')
    return out

def fetch_yf_prices(tickers: list[str]) -> dict:
    """
    Always uses interval='5m', period='2d'.
    - During market hours   → live 5-min bars, ~5 min lag
    - After close / overnight → last bar of the session (3:25 PM closing bar)
    - Never uses interval='1d' which returns yesterday's data until next morning
    """
    result = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for batch in batches:
        syms = [t + '.NS' for t in batch]
        prices = _yf_batch(syms, '5m', '2d')
        for sym, ticker in zip(syms, batch):
            if sym not in prices:
                continue
            p, pc = prices[sym]
            result[ticker] = {
                'price':      round(p, 2),
                'prev_close': round(pc, 2),
                'change_abs': round(p - pc, 2),
                'change_pct': round((p - pc) / pc * 100, 2) if pc else 0.0,
            }
    return result

def fetch_all_prices(tickers: list[str]) -> dict:
    """
    Strategy:
      1. NSE pre-open bulk  — single call, <1s, covers ~2200 stocks (pre-open IEP / last traded)
      2. yfinance           — 5m intervals during market hours (~5 min lag), daily otherwise
         Used for tickers missing from NSE bulk AND as the primary source during intraday
         when NSE bulk may be stale (it freezes at 9:08 AM pre-open).
    During market hours we prefer yfinance 5m over NSE pre-open (which is stale intraday).
    """
    now   = ist_now()
    mins  = now.hour * 60 + now.minute
    pre_open_window = (9*60) <= mins < (9*60+15)

    if pre_open_window:
        # 9:00–9:15 AM: NSE pre-open gives live IEP — use it as primary
        prices = fetch_nse_bulk()
        missing = [t for t in tickers if t not in prices]
        if missing:
            prices.update(fetch_yf_prices(missing))
    else:
        # After 9:15 AM (market open) or outside hours: yfinance is more reliable
        prices = fetch_yf_prices(tickers)
        # Fill any gaps with NSE bulk
        missing = [t for t in tickers if t not in prices]
        if missing:
            nse = fetch_nse_bulk()
            for t in missing:
                if t in nse:
                    prices[t] = nse[t]

    return {t: prices[t] for t in tickers if t in prices}

def write_live_prices(prices: dict):
    """Write live_prices.json (read by chart_viewer.html every 30 s)."""
    now_ist = ist_now()
    payload = {
        'updated_at':    now_ist.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'updated_str':   now_ist.strftime('%I:%M %p IST'),
        'market_open':   market_open(),
        'count':         len(prices),
        'prices':        prices,
    }
    tmp = LIVE_PRICES + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    os.replace(tmp, LIVE_PRICES)

def patch_india_data(prices: dict):
    """
    Patch prices in india_data.json — ONLY during market hours.
    Outside hours the scanner's closing prices are authoritative; don't overwrite them.
    """
    now  = ist_now()
    mins = now.hour * 60 + now.minute
    in_session = (9*60 + 15) <= mins <= (15*60 + 35)   # 9:15 AM – 3:35 PM IST
    if not (market_open() and in_session):
        return   # leave scanner's closing prices intact
    try:
        data = json.load(open(INDIA_DATA))
    except Exception:
        return
    changed = 0
    for scr in data.values():
        for s in scr.get('stocks', []):
            t = s.get('ticker', '')
            if t in prices:
                s['price'] = prices[t]['price']
                changed += 1
    if changed:
        tmp = INDIA_DATA + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        os.replace(tmp, INDIA_DATA)

# ── US price functions ────────────────────────────────────────────────────────
def get_us_tickers() -> list[str]:
    """Return unique tickers in us_data.json."""
    tickers = set()
    try:
        data = json.load(open(US_DATA))
        for scr in data.values():
            for s in scr.get('stocks', []):
                t = s.get('ticker', '')
                if t:
                    tickers.add(t)
    except Exception:
        pass
    return sorted(tickers)

def fetch_us_prices(tickers: list[str]) -> dict:
    """Fetch US stock prices via yfinance 5m bars (no .NS suffix)."""
    result = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for batch in batches:
        prices = _yf_batch(batch, '5m', '2d')
        for sym in batch:
            if sym not in prices:
                continue
            p, pc = prices[sym]
            result[sym] = {
                'price':      round(p, 2),
                'prev_close': round(pc, 2),
                'change_abs': round(p - pc, 2),
                'change_pct': round((p - pc) / pc * 100, 2) if pc else 0.0,
            }
    return result

def write_live_us_prices(prices: dict):
    now_et = et_now()
    payload = {
        'updated_at':  now_et.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'updated_str': now_et.strftime('%I:%M %p ET'),
        'market_open': us_market_open(),
        'count':       len(prices),
        'prices':      prices,
    }
    tmp = LIVE_PRICES_US + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    os.replace(tmp, LIVE_PRICES_US)

def patch_us_data(prices: dict):
    """Patch us_data.json with live prices — only during US market hours."""
    now  = et_now()
    mins = now.hour * 60 + now.minute
    in_session = (9*60 + 30) <= mins <= (16*60)
    if not (us_market_open() and in_session):
        return
    try:
        data = json.load(open(US_DATA))
    except Exception:
        return
    changed = 0
    for scr in data.values():
        for s in scr.get('stocks', []):
            t = s.get('ticker', '')
            if t in prices:
                s['price'] = prices[t]['price']
                changed += 1
    if changed:
        tmp = US_DATA + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        os.replace(tmp, US_DATA)

# ── Price loop ────────────────────────────────────────────────────────────────
def price_loop():
    log.info('Price updater started (India + US) — press Ctrl+C to stop')
    while True:
        india_open = market_open()
        us_open    = us_market_open()

        if not india_open and not us_open:
            now_ist = ist_now()
            now_et  = et_now()
            log.info(f'Both markets closed (IST {now_ist.strftime("%H:%M")} / ET {now_et.strftime("%H:%M")}) — sleeping 5 min')
            time.sleep(300)
            continue

        t0 = time.time()

        # India prices
        if india_open:
            tickers = get_screen_tickers()
            log.info(f'[India] Fetching {len(tickers)} prices…')
            prices = fetch_all_prices(tickers)
            write_live_prices(prices)
            patch_india_data(prices)
            log.info(f'[India] Updated {len(prices)}/{len(tickers)} in {time.time()-t0:.1f}s')

        # US prices
        if us_open:
            t1 = time.time()
            us_tickers = get_us_tickers()
            if us_tickers:
                log.info(f'[US] Fetching {len(us_tickers)} prices…')
                us_prices = fetch_us_prices(us_tickers)
                write_live_us_prices(us_prices)
                patch_us_data(us_prices)
                log.info(f'[US] Updated {len(us_prices)}/{len(us_tickers)} in {time.time()-t1:.1f}s')

        elapsed = time.time() - t0
        time.sleep(max(0, PRICE_INTERVAL_S - elapsed))

# ── Re-scan loop (every 30 min during market hours) ──────────────────────────
_rescan_lock = threading.Lock()

MANIFEST_PATH = os.path.join(HERE, 'chart_manifest.json')

def clear_manifest_today():
    """Remove today's entries from manifest so --daily regenerates fresh charts."""
    try:
        today = str(__import__('datetime').date.today())
        manifest = json.load(open(MANIFEST_PATH)) if os.path.exists(MANIFEST_PATH) else {}
        cleared = {k: v for k, v in manifest.items() if v.get('date') != today}
        with open(MANIFEST_PATH, 'w') as f:
            json.dump(cleared, f)
        removed = len(manifest) - len(cleared)
        if removed:
            log.info(f'Cleared {removed} today-dated entries from chart manifest')
    except Exception as e:
        log.warning(f'Could not clear manifest: {e}')

def _run_rescan(label, scanner_py, scanner_cwd, chart_market):
    t0 = time.time()
    log.info(f'=== [{label}] Starting re-scan ===')
    try:
        subprocess.run([PYTHON, scanner_py], cwd=scanner_cwd, timeout=600)
        log.info(f'[{label}] Scanner done in {time.time()-t0:.0f}s — regenerating charts…')
        clear_manifest_today()
        subprocess.run(
            [PYTHON, GENcharts_PY, '--market', chart_market, '--daily'],
            cwd=HERE, timeout=600
        )
        log.info(f'=== [{label}] Re-scan complete ({time.time()-t0:.0f}s total) ===')
    except Exception as e:
        log.error(f'[{label}] Re-scan failed: {e}')

def rescan_loop():
    time.sleep(RESCAN_INTERVAL_S)  # stagger first run
    while True:
        india_open = market_open()
        us_open    = us_market_open()
        if not india_open and not us_open:
            time.sleep(60)
            continue
        with _rescan_lock:
            if india_open:
                _run_rescan('India', SCANNER_PY, SCANNER_DIR, 'india')
            if us_open:
                _run_rescan('US', SCANNER_US_PY, SCANNER_DIR, 'us')
        time.sleep(RESCAN_INTERVAL_S)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rescan_thread = threading.Thread(target=rescan_loop, daemon=True)
    rescan_thread.start()

    try:
        price_loop()
    except KeyboardInterrupt:
        log.info('Stopped.')

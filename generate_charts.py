#!/usr/bin/env python3.10
"""
Chart Pattern Viewer — generates annotated candlestick chart images
with AI-detected patterns (trendlines, breakouts, VCP, support/resistance)
and builds a Finviz-style HTML grid viewer.

Usage:
  python3.10 generate_charts.py            # India (default)
  python3.10 generate_charts.py --market us
  python3.10 generate_charts.py --screen vcp_setup
  python3.10 generate_charts.py --market us --screen near_breakout --max 60
"""

import os, sys, json, pickle, argparse, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────────────────────
SCANNER_DIR  = os.path.expanduser('~/Desktop/example/minervini-trend-template-scanner')
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INDIA_JSON   = os.path.join(SCANNER_DIR, 'results.json')
US_JSON      = os.path.join(SCANNER_DIR, 'results_us.json')
INDIA_CACHE  = os.path.join(SCANNER_DIR, '.cache')
US_CACHE     = os.path.join(SCANNER_DIR, '.cache_us')
OUT_INDIA    = os.path.join(SCRIPT_DIR, 'charts_india')
OUT_US       = os.path.join(SCRIPT_DIR, 'charts_us')
HTML_OUT     = os.path.join(SCRIPT_DIR, 'chart_viewer.html')

CHART_W, CHART_H = 8, 4.5   # inches per chart
DPI = 100                    # lower = faster, higher = sharper

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
BG       = '#0d1117'
GRID_C   = '#1c2333'
UP_C     = '#3fb950'
DN_C     = '#f85149'
VOL_UP   = '#1e4620'
VOL_DN   = '#4a1010'
MA20_C   = '#fbbf24'    # amber
MA50_C   = '#60a5fa'    # blue
MA200_C  = '#a78bfa'    # violet
TLINE_C  = '#94a3b8'    # slate (support/resistance lines)
PIVOT_C  = '#fb923c'    # orange (breakout pivot)
TEXT_C   = '#e6edf3'

# ── PATTERN DETECTION ─────────────────────────────────────────────────────────

def find_pivot_highs(high, left=10, right=10, lookback=150):
    """
    Pivot high: bar must be the highest in [left] bars before AND [right] bars after.
    Minervini/TradingView standard: left=10, right=10.
    """
    h = np.array(high[-lookback:] if len(high) > lookback else high, dtype=float)
    offset = max(0, len(high) - lookback)
    pivots = []
    for i in range(left, len(h) - right):
        window = h[i - left: i + right + 1]
        if float(h[i]) >= float(np.max(window)):
            pivots.append((offset + i, float(h[i])))
    return pivots


def find_pivot_lows(low, left=10, right=10, lookback=150):
    """
    Pivot low: bar must be the lowest in [left] bars before AND [right] bars after.
    Minervini/TradingView standard: left=10, right=10.
    """
    l = np.array(low[-lookback:] if len(low) > lookback else low, dtype=float)
    offset = max(0, len(low) - lookback)
    pivots = []
    for i in range(left, len(l) - right):
        window = l[i - left: i + right + 1]
        if float(l[i]) <= float(np.min(window)):
            pivots.append((offset + i, float(l[i])))
    return pivots


def fit_trendline(points):
    """Fit a line through (index, price) pivot points. Returns (slope, intercept) or None."""
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    coeffs = np.polyfit(xs, ys, 1)
    return coeffs  # [slope, intercept]


def detect_vcp_contractions(close, high, low, lookback=60):
    """
    Simple VCP contraction detector.
    Returns list of (start_idx, end_idx, range_pct) for contracting price swings.
    """
    h = np.array(high[-lookback:], dtype=float)
    l = np.array(low[-lookback:], dtype=float)
    offset = max(0, len(close) - lookback)

    ph = find_pivot_highs(h, left=5, right=5, lookback=lookback)
    pl = find_pivot_lows(l,  left=5, right=5, lookback=lookback)

    # pair each pivot high with nearest following pivot low
    contractions = []
    for i, (hi_idx, hi_val) in enumerate(ph):
        # find the first pivot low after this pivot high
        following_lows = [(li, lv) for li, lv in pl if li > hi_idx]
        if not following_lows:
            continue
        lo_idx, lo_val = following_lows[0]
        rng = (hi_val - lo_val) / hi_val * 100
        contractions.append((hi_idx, lo_idx, hi_val, lo_val, round(rng, 1)))

    # keep only contracting (each range < previous)
    if len(contractions) < 2:
        return contractions
    filtered = [contractions[0]]
    for c in contractions[1:]:
        if c[4] < filtered[-1][4] * 0.95:   # at least 5% tighter
            filtered.append(c)
    return filtered


def detect_breakout(close, high, pivot_high, lookback=3):
    """True if recent close crossed above pivot_high with recency."""
    if pivot_high is None or pivot_high <= 0:
        return False
    recent = close[-lookback:]
    return any(float(c) >= pivot_high * 0.995 for c in recent)


# ── EXTRA PATTERN DETECTORS ───────────────────────────────────────────────────

def detect_ipo_base(df):
    """IPO stock (< 2 yrs, > 45 trading days) in or near first base."""
    n = len(df)
    if n < 45 or n > 480:
        return False
    c = df['Close'].values
    h = df['High'].values
    ath = float(np.max(h))
    price = float(c[-1])
    if ath == 0 or (price / ath - 1) * 100 < -25:
        return False
    if n >= 20 and price < float(np.mean(c[-20:])) * 0.95:
        return False
    return True


def detect_ath_consolidation(df):
    """Within 8% of ATH after a clearly defined consolidation base."""
    c = df['Close'].values
    h = df['High'].values
    if len(c) < 60:
        return False
    ath   = float(np.max(h))
    price = float(c[-1])
    if ath == 0 or (price / ath - 1) * 100 < -8:
        return False
    # Require a tight consolidation window (≤18% range) in the 10-45 bars before now
    for w in [15, 20, 30]:
        start = -(w + 12)
        end   = -10
        if len(c) < abs(start):
            continue
        seg = c[start:end]
        seg_rng = (float(np.max(seg)) - float(np.min(seg))) / float(np.mean(seg)) * 100
        if seg_rng <= 18:
            return True
    return False


def detect_weekly_pivot_break(df):
    """Price breaking above or recently broke a key weekly pivot high."""
    if len(df) < 80:
        return False
    df2 = df.copy()
    if not isinstance(df2.index, pd.DatetimeIndex):
        df2.index = pd.to_datetime(df2.index)
    wkly = df2['High'].resample('W').max()
    if len(wkly) < 12:
        return False
    pivot_region = wkly.iloc[-20:-3]
    if len(pivot_region) < 4:
        return False
    pivot_level = float(pivot_region.max())
    price = float(df['Close'].iloc[-1])
    pct   = (price / pivot_level - 1) * 100
    if -3 <= pct <= 12:
        if len(df) >= 50:
            ma50 = float(df['Close'].rolling(50).mean().iloc[-1])
            return price > ma50 * 0.95
    return False


def detect_pocket_pivot(df):
    """Up day with vol > any down-day vol in last 10 days, near 10-day MA."""
    c = df['Close'].values
    v = df['Volume'].values
    if len(c) < 22:
        return False
    if float(c[-1]) <= float(c[-2]):
        return False
    today_vol = float(v[-1])
    max_dn_vol = max(
        (float(v[i]) for i in range(-11, -1) if float(c[i]) < float(c[i-1])),
        default=0.0
    )
    if max_dn_vol == 0 or today_vol < max_dn_vol:
        return False
    ma10 = float(np.mean(c[-10:]))
    pct_ma10 = (float(c[-1]) / ma10 - 1) * 100
    if not (-5 <= pct_ma10 <= 8):
        return False
    if len(c) >= 50:
        ma50 = float(np.mean(c[-50:]))
        return float(c[-1]) > ma50 * 0.95
    return True


def detect_power_play(df, rs_rank=0):
    """RS≥80 stock with ≥4% single-day surge on 1.5× volume, price held."""
    c = df['Close'].values
    v = df['Volume'].values
    if len(c) < 30 or rs_rank < 80:
        return False
    avg_vol = float(np.mean(v[-30:-1]))
    for i in range(-10, 0):
        pct_move = (float(c[i]) - float(c[i-1])) / float(c[i-1]) * 100
        if pct_move >= 4 and avg_vol > 0 and float(v[i]) >= avg_vol * 1.5:
            if float(c[-1]) >= float(c[i]) * 0.93:
                return True
    return False


# ── HORIZONTAL S/R CLUSTERING ─────────────────────────────────────────────────

def cluster_sr_levels(high, low, lookback=120, price_tol=0.018, min_touches=2):
    """
    Find horizontal S/R levels by clustering pivot-point price touches.
    Returns list of (price_level, touch_count) strongest first.
    """
    h = np.array(high[-lookback:] if len(high) > lookback else high, dtype=float)
    l = np.array(low[-lookback:] if len(low) > lookback else low, dtype=float)
    # Collect pivot highs and lows (looser 3/3 for clustering — we want density)
    candidates = []
    for i in range(3, len(h) - 3):
        if h[i] >= np.max(h[i-3:i+4]):
            candidates.append(float(h[i]))
        if l[i] <= np.min(l[i-3:i+4]):
            candidates.append(float(l[i]))
    if len(candidates) < 2:
        return []
    candidates.sort()
    # Merge candidates within price_tol of each other
    clusters = []
    used = [False] * len(candidates)
    for i, p in enumerate(candidates):
        if used[i]:
            continue
        grp = [p]
        for j in range(i + 1, len(candidates)):
            if not used[j] and abs(candidates[j] - np.mean(grp)) / np.mean(grp) <= price_tol:
                grp.append(candidates[j])
                used[j] = True
        if len(grp) >= min_touches:
            clusters.append((float(np.mean(grp)), len(grp)))
    clusters.sort(key=lambda x: -x[1])
    return clusters[:6]


def detect_base_box(close, high, low, lookback=50):
    """
    Detect a consolidation base. Returns (base_high, base_low, start_idx) or None.
    Scans for the tightest multi-bar range ending at the current bar.
    """
    c = np.array(close, dtype=float)
    h = np.array(high, dtype=float)
    l = np.array(low, dtype=float)
    best = None
    for window in range(min(lookback, len(c)-1), 9, -1):
        seg_h = float(np.max(h[-window:]))
        seg_l = float(np.min(l[-window:]))
        rng   = (seg_h - seg_l) / seg_l * 100 if seg_l > 0 else 99
        if rng <= 20:
            best = (seg_h, seg_l, len(c) - window)
    return best


def identify_chart_pattern(close, high, low, volume, ma20_arr, ma50_arr):
    """
    Identify the dominant chart pattern. Returns (name, confidence_pct, note).
    """
    c  = np.array(close,  dtype=float)
    h  = np.array(high,   dtype=float)
    l  = np.array(low,    dtype=float)
    v  = np.array(volume, dtype=float)
    n  = len(c)
    price = float(c[-1])

    if n < 25:
        return 'WATCH', 40, ''

    # ── Breakout ─────────────────────────────────────────────────────────────
    if n >= 25:
        hi20_prior = float(np.max(h[-25:-5]))
        if price > hi20_prior * 0.998:
            avg_vol = float(np.mean(v[-20:-1])) if len(v) >= 20 else 0
            vol_conf = avg_vol > 0 and float(v[-1]) >= avg_vol * 1.3
            return ('BREAKOUT' if vol_conf else 'NEAR BREAKOUT',
                    90 if vol_conf else 75,
                    'Vol surge' if vol_conf else 'Low-vol test — watch')

    # ── Cup & Handle ─────────────────────────────────────────────────────────
    if n >= 55:
        left_lip  = float(np.max(h[-55:-38]))
        cup_bot   = float(np.min(l[-38:-12]))
        right_lip = float(np.max(h[-18:-5]))
        handle_lo = float(np.min(l[-10:]))
        cup_depth   = (left_lip - cup_bot)  / left_lip  * 100
        handle_pct  = (right_lip - handle_lo) / right_lip * 100
        lip_diff    = abs(right_lip - left_lip) / left_lip * 100
        if (10 <= cup_depth <= 45 and handle_pct <= cup_depth * 0.65
                and lip_diff <= 10 and price > cup_bot * 1.08):
            return 'CUP & HANDLE', 83, f'Cup {cup_depth:.0f}% Handle {handle_pct:.0f}%'

    # ── Double Bottom ─────────────────────────────────────────────────────────
    if n >= 35:
        seg_l = l[-35:]
        bot_idx = np.argsort(seg_l)[:6]
        bot_idx.sort()
        if len(bot_idx) >= 2:
            b1, b2 = int(bot_idx[0]), int(bot_idx[-1])
            if b2 - b1 >= 8:
                lo1, lo2 = float(seg_l[b1]), float(seg_l[b2])
                if abs(lo1 - lo2) / max(lo1, lo2) <= 0.04:
                    mid_h = float(np.max(c[-35:][b1:b2])) if b2 > b1 else 0
                    recovery = (mid_h - min(lo1,lo2)) / min(lo1,lo2) * 100 if min(lo1,lo2)>0 else 0
                    if recovery >= 5 and price > min(lo1, lo2) * 1.04:
                        return 'DOUBLE BOTTOM', 79, f'Support ~{min(lo1,lo2):,.0f}'

    # ── Ascending Triangle ────────────────────────────────────────────────────
    if n >= 25:
        res_level = float(np.max(h[-25:]))
        res_tests = sum(1 for hh in h[-25:] if abs(hh - res_level)/res_level <= 0.015)
        pl_recent = [float(l[i]) for i in range(n-25, n-1)
                     if i > 2 and float(l[i]) <= float(l[i-1]) and float(l[i]) <= float(l[i+1])]
        if res_tests >= 2 and len(pl_recent) >= 3:
            rising = all(pl_recent[i] < pl_recent[i+1] for i in range(len(pl_recent)-1))
            if rising and price >= res_level * 0.96:
                return 'ASCENDING TRIANGLE', 81, f'Res {res_level:,.0f}'

    # ── Bull Flag / Pennant ───────────────────────────────────────────────────
    if n >= 30:
        pole_hi = float(np.max(h[-30:-10]))
        pole_lo = float(np.min(l[-30:-10]))
        flag_hi = float(np.max(h[-10:]))
        flag_lo = float(np.min(l[-10:]))
        pole_pct = (pole_hi - pole_lo) / pole_lo * 100 if pole_lo > 0 else 0
        flag_pct = (flag_hi - flag_lo) / flag_lo * 100 if flag_lo > 0 else 99
        if pole_pct >= 12 and flag_pct <= pole_pct * 0.45 and price > flag_lo * 1.01:
            return 'BULL FLAG', 78, f'Pole {pole_pct:.0f}% Flag {flag_pct:.0f}%'

    # ── VCP / Tight Base ─────────────────────────────────────────────────────
    if n >= 20:
        rng20 = (float(np.max(h[-20:])) - float(np.min(l[-20:]))) / float(np.min(l[-20:])) * 100
        ma20_v = float(ma20_arr[-1]) if ma20_arr is not None and not np.isnan(ma20_arr[-1]) else None
        if rng20 <= 8 and ma20_v and price > ma20_v:
            return 'TIGHT BASE', 74, f'Range {rng20:.1f}%'

    # ── Stage 2 / Stage 4 ────────────────────────────────────────────────────
    if ma50_arr is not None and len(ma50_arr) and not np.isnan(ma50_arr[-1]):
        ma50_v = float(ma50_arr[-1])
        if price > ma50_v * 1.03:
            return 'STAGE 2', 62, 'Above MA50, uptrend'
        if price < ma50_v * 0.97:
            return 'STAGE 4', 55, 'Below MA50, downtrend'

    return 'WATCH', 45, 'Building setup'


# ── VISUAL AI ANALYSIS (Ollama vision) ────────────────────────────────────────
#
# Sends the actual chart PNG to a local vision LLM — no API key needed.
#
# Vision models (require Ollama pull):
#   llava:7b          — 4.7 GB, works with current Ollama  ← DEFAULT
#   llava:13b         — 8.0 GB, better accuracy
#   llama3.2-vision   — 7.9 GB, needs Ollama ≥ 0.4
#
# Text-only fallback (if no vision model found):
#   llama3.2:latest / llama3.1:8b — reads OHLCV numbers instead of image
#
# Usage:
#   python3.10 generate_charts.py --daily --ai-analysis
#   ollama serve   (must be running)

import urllib.request as _urllib_req
import base64 as _b64

_AI_ENABLED     = False
_OLLAMA_URL     = 'http://localhost:11434'
_OLLAMA_MODEL   = 'llava:7b'
_VISION_CAPABLE = False   # True when selected model can process images

# Models that support image input
_VISION_MODELS = ['llava:7b', 'llava:13b', 'llava:latest', 'llava',
                  'llama3.2-vision:11b', 'llama3.2-vision',
                  'minicpm-v', 'moondream', 'bakllava']


def _build_ohlcv_summary(c, h, l, v, ma20, ma50, ma200, entry):
    """Compact OHLCV text context — appended to vision prompt for extra grounding."""
    n = len(c)
    pct_chg  = [(float(c[i]) - float(c[i-1])) / float(c[i-1]) * 100 for i in range(max(1, n-20), n)]
    avg_vol   = float(np.mean(v[-20:])) if len(v) >= 20 else float(np.mean(v))
    vol_ratio = float(v[-1]) / avg_vol if avg_vol > 0 else 1.0
    hi52  = float(np.max(h[-252:])) if len(h) >= 252 else float(np.max(h))
    lo52  = float(np.min(l[-252:])) if len(l) >= 252 else float(np.min(l))
    price = float(c[-1])
    ma200_v = float(ma200[-1]) if ma200 is not None and len(ma200) and not np.isnan(ma200[-1]) else None
    return (
        f"Price={price:.2f}  52wHigh={hi52:.2f}({(price/hi52-1)*100:+.1f}%)  "
        f"52wLow={lo52:.2f}({(price/lo52-1)*100:+.1f}%)\n"
        f"MA20={float(ma20[-1]):.2f}  MA50={float(ma50[-1]):.2f}  "
        f"MA200={f'{ma200_v:.2f}' if ma200_v else 'N/A'}\n"
        f"Last10 returns: {' '.join(f'{x:+.1f}%' for x in pct_chg[-10:])}\n"
        f"VolRatio={vol_ratio:.2f}x  RS={entry.get('rs_rank','?')}  "
        f"Criteria={entry.get('passed','?')}/8  "
        f"PctFromHigh={entry.get('pct_from_high','?')}%"
    )


def init_ai_client():
    """Connect to Ollama, pick the best available model, warm it up."""
    global _AI_ENABLED, _OLLAMA_MODEL, _VISION_CAPABLE
    import json as _json
    try:
        req  = _urllib_req.urlopen(f'{_OLLAMA_URL}/api/tags', timeout=5)
        data = _json.loads(req.read())
        names = [m['name'] for m in data.get('models', [])]

        # 1. Prefer a vision-capable model
        for preferred in _VISION_MODELS:
            if preferred in names:
                _OLLAMA_MODEL   = preferred
                _VISION_CAPABLE = True
                break
        else:
            # 2. Fall back to text-only
            for preferred in ['llama3.2:latest', 'llama3.1:8b', 'phi3:latest']:
                if preferred in names:
                    _OLLAMA_MODEL   = preferred
                    _VISION_CAPABLE = False
                    break
            else:
                print(f"  Ollama: no usable model found. Available: {names}")
                return False

        mode = 'VISUAL (looks at chart image)' if _VISION_CAPABLE else 'TEXT-ONLY (reads OHLCV numbers)'
        print(f"  Warming up {_OLLAMA_MODEL} [{mode}]…", flush=True)
        body = _json.dumps({
            'model': _OLLAMA_MODEL, 'prompt': 'Say OK',
            'stream': False, 'options': {'num_predict': 4}
        }).encode()
        wreq = _urllib_req.Request(f'{_OLLAMA_URL}/api/generate', data=body,
                                   headers={'Content-Type': 'application/json'})
        _urllib_req.urlopen(wreq, timeout=90)
        _AI_ENABLED = True
        print(f"  ✓ AI ready — {_OLLAMA_MODEL}  [{mode}]")
        return True
    except Exception as e:
        print(f"  Ollama not reachable ({e}).  Run: ollama serve")
    return False


def _call_ollama(prompt, img_path=None, num_predict=220):
    """Low-level Ollama call. Attaches image as base64 if img_path given."""
    import json as _json
    payload = {
        'model':   _OLLAMA_MODEL,
        'prompt':  prompt,
        'stream':  False,
        'options': {'temperature': 0.1, 'num_predict': num_predict},
    }
    if img_path and _VISION_CAPABLE and os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            payload['images'] = [_b64.b64encode(f.read()).decode()]
    body = _json.dumps(payload).encode()
    req  = _urllib_req.Request(f'{_OLLAMA_URL}/api/generate', data=body,
                               headers={'Content-Type': 'application/json'})
    resp = _urllib_req.urlopen(req, timeout=90)
    return _json.loads(resp.read()).get('response', '').strip()


def claude_analyze(ticker, ohlcv_summary, timeframe='daily', img_path=None):
    """
    Visual AI analysis: sends chart PNG to Ollama vision model.
    Falls back to text-only OHLCV analysis if no vision model available.
    Returns dict: {pattern, conf, support, resistance, note, action} or None.
    """
    if not _AI_ENABLED:
        return None
    import json as _json

    tf = 'weekly' if timeframe == 'weekly' else 'daily'

    if _VISION_CAPABLE and img_path and os.path.exists(img_path):
        # ── VISUAL MODE: model looks at the actual chart image ────────────────
        prompt = (
            f"You are a professional stock trader using Mark Minervini's SEPA method.\n"
            f"Look at this {tf} candlestick chart for {ticker}.\n\n"
            f"Additional context:\n{ohlcv_summary}\n\n"
            f"Visually identify the chart pattern by examining:\n"
            f"- Candlestick shape and trend (are prices making higher highs/lows?)\n"
            f"- Moving average alignment (MA20 above MA50 above MA200?)\n"
            f"- Volume pattern (rising on up days, falling on down days?)\n"
            f"- Any classic pattern: Cup & Handle, VCP, Bull Flag, Double Bottom, "
            f"Ascending Triangle, Tight Base, or Breakout\n"
            f"- Support and resistance levels visible on the chart\n\n"
            f"Reply with ONLY valid JSON on one line, no markdown, no explanation:\n"
            f'{{"pattern":"<CUP & HANDLE|VCP|BREAKOUT|BULL FLAG|DOUBLE BOTTOM|'
            f'ASCENDING TRIANGLE|TIGHT BASE|STAGE 2|STAGE 4|WATCH>",'
            f'"conf":<0-100>,"support":<price>,"resistance":<price>,'
            f'"note":"<max 55 chars: what you visually see>","action":"<BUY|WATCH|AVOID>"}}'
        )
    else:
        # ── TEXT-ONLY FALLBACK: model reads OHLCV numbers ─────────────────────
        prompt = (
            f"You are a technical analyst using Mark Minervini's SEPA method.\n"
            f"Analyze {ticker} on a {tf} chart from this data:\n\n{ohlcv_summary}\n\n"
            f"Reply with ONLY valid JSON on one line:\n"
            f'{{"pattern":"<CUP & HANDLE|VCP|BREAKOUT|BULL FLAG|DOUBLE BOTTOM|'
            f'ASCENDING TRIANGLE|TIGHT BASE|STAGE 2|WATCH>",'
            f'"conf":<0-100>,"support":<price>,"resistance":<price>,'
            f'"note":"<max 55 chars>","action":"<BUY|WATCH|AVOID>"}}'
        )

    try:
        text = _call_ollama(prompt, img_path=img_path if _VISION_CAPABLE else None)
        s, e = text.find('{'), text.rfind('}') + 1
        if s >= 0 and e > s:
            result = _json.loads(text[s:e])
            if 'confidence' in result and 'conf' not in result:
                v = result.pop('confidence')
                result['conf'] = int(float(v) * 100 if float(v) <= 1.0 else float(v))
            result['vision'] = _VISION_CAPABLE   # flag so viewer can show "Visual AI"
            if 'pattern' in result and 'action' in result:
                return result
    except Exception:
        pass
    return None


_AI_CLIENT = None   # legacy alias, unused


# ── CCI DETECTORS ─────────────────────────────────────────────────────────────

def _calc_cci(series_high, series_low, series_close, period=34):
    tp  = (series_high + series_low + series_close) / 3
    ma  = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


def detect_cci_daily_cross(df, period=34, threshold=100):
    """Daily CCI just crossed above threshold (previous bar was below)."""
    if len(df) < period + 3:
        return False
    cci = _calc_cci(df['High'], df['Low'], df['Close'], period).dropna()
    if len(cci) < 2:
        return False
    return float(cci.iloc[-1]) >= threshold and float(cci.iloc[-2]) < threshold


def detect_cci_daily_above(df, period=34, threshold=100):
    """Daily CCI above threshold and still rising (momentum continuation)."""
    if len(df) < period + 3:
        return False
    cci = _calc_cci(df['High'], df['Low'], df['Close'], period).dropna()
    if len(cci) < 3:
        return False
    v = float(cci.iloc[-1])
    return v >= threshold and v > float(cci.iloc[-3])


def detect_cci_weekly_cross(df, period=34, threshold=100):
    """Weekly CCI just crossed above threshold."""
    if len(df) < 80:
        return False
    df2 = df.copy()
    if not isinstance(df2.index, pd.DatetimeIndex):
        df2.index = pd.to_datetime(df2.index)
    wkly = df2.resample('W').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    if len(wkly) < period + 3:
        return False
    cci = _calc_cci(wkly['High'], wkly['Low'], wkly['Close'], period).dropna()
    if len(cci) < 2:
        return False
    return float(cci.iloc[-1]) >= threshold and float(cci.iloc[-2]) < threshold


def detect_cci_weekly_above(df, period=34, threshold=100):
    """Weekly CCI above threshold and rising."""
    if len(df) < 80:
        return False
    df2 = df.copy()
    if not isinstance(df2.index, pd.DatetimeIndex):
        df2.index = pd.to_datetime(df2.index)
    wkly = df2.resample('W').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    if len(wkly) < period + 3:
        return False
    cci = _calc_cci(wkly['High'], wkly['Low'], wkly['Close'], period).dropna()
    if len(cci) < 3:
        return False
    v = float(cci.iloc[-1])
    return v >= threshold and v > float(cci.iloc[-3])


def detect_cci_pullback(df, period=34):
    """CCI pulled back to 50-100 zone after being above 100 — buy-the-dip setup."""
    if len(df) < period + 10:
        return False
    cci = _calc_cci(df['High'], df['Low'], df['Close'], period).dropna()
    if len(cci) < 6:
        return False
    recent = [float(cci.iloc[i]) for i in range(-6, 0)]
    was_above = any(v >= 100 for v in recent[:-2])
    now_pullback = 30 <= float(cci.iloc[-1]) <= 100
    return was_above and now_pullback


# ── ADDITIONAL PATTERN DETECTORS ──────────────────────────────────────────────

def detect_tight_consolidation(df):
    """Volatility squeeze: price in ≤8% range for 10+ days, Bollinger Bands narrow."""
    c = df['Close'].values
    if len(c) < 25:
        return False
    seg = c[-15:]
    rng = (float(np.max(seg)) - float(np.min(seg))) / float(np.mean(seg)) * 100
    if rng > 8:
        return False
    # Bollinger Band width < 5%
    ma20 = float(np.mean(c[-20:]))
    std20 = float(np.std(c[-20:]))
    if ma20 > 0 and (std20 * 4 / ma20 * 100) < 8:
        return float(c[-1]) > ma20 * 0.97
    return False


def detect_ma_tightening(df):
    """Price, MA20, MA50 all within 5% of each other — coiling for a move."""
    c = df['Close'].values
    if len(c) < 55:
        return False
    price = float(c[-1])
    ma20  = float(np.mean(c[-20:]))
    ma50  = float(np.mean(c[-50:]))
    spread = (max(price, ma20, ma50) - min(price, ma20, ma50)) / min(price, ma20, ma50) * 100
    return spread <= 5 and price > ma50 * 0.97


def detect_high_vol_accumulation(df):
    """Up days have significantly higher volume than down days (institutional accumulation)."""
    c = df['Close'].values
    v = df['Volume'].values
    if len(c) < 20:
        return False
    up_vols = [float(v[i]) for i in range(-20, 0) if float(c[i]) > float(c[i-1])]
    dn_vols  = [float(v[i]) for i in range(-20, 0) if float(c[i]) <= float(c[i-1])]
    if not up_vols or not dn_vols:
        return False
    ratio = np.mean(up_vols) / np.mean(dn_vols)
    return ratio >= 1.5 and float(c[-1]) > float(np.mean(c[-50:])) if len(c) >= 50 else ratio >= 1.5


def detect_rs_new_high(df, rs_rank=0):
    """RS rank ≥ 85 and price near 52W high — relative strength leader."""
    if rs_rank < 85:
        return False
    h = df['High'].values
    c = df['Close'].values
    if len(h) < 252:
        return False
    hi52 = float(np.max(h[-252:]))
    price = float(c[-1])
    return hi52 > 0 and (price / hi52 - 1) * 100 >= -5


def detect_momentum_breakout(df):
    """Price broke above 20-day high in last 3 sessions on above-avg volume."""
    c = df['Close'].values
    v = df['Volume'].values
    if len(c) < 25:
        return False
    hi20_prev = float(np.max(c[-23:-3]))
    avg_vol   = float(np.mean(v[-20:-1]))
    for i in (-3, -2, -1):
        if float(c[i]) > hi20_prev and avg_vol > 0 and float(v[i]) >= avg_vol * 1.2:
            return True
    return False


def detect_first_pullback(df):
    """First pullback to MA20 after a strong trending move — continuation entry."""
    c = df['Close'].values
    if len(c) < 30:
        return False
    ma20  = float(np.mean(c[-20:]))
    ma50  = float(np.mean(c[-50:])) if len(c) >= 50 else None
    price = float(c[-1])
    pct_from_ma20 = (price / ma20 - 1) * 100
    # Near MA20 (within -2% to +2%)
    if not (-2 <= pct_from_ma20 <= 2):
        return False
    # Prior trend: 10-30 bars ago price was well above MA20 (>5%)
    prev_prices = c[-30:-10]
    prev_ma20s  = [float(np.mean(c[max(0,i-20):i])) for i in range(len(c)-30, len(c)-10)]
    was_extended = any(
        (float(p) / float(m) - 1) * 100 > 5
        for p, m in zip(prev_prices, prev_ma20s) if m > 0
    )
    if not was_extended:
        return False
    return ma50 is None or price > ma50


EXTRA_SCREENS_DEF = {
    # CCI screens — weekly variants get weekly charts
    'cci_daily_cross':   {'label': 'CCI Daily Crossed 100',        'fn': lambda df, rs: detect_cci_daily_cross(df),   'weekly': False},
    'cci_daily_above':   {'label': 'CCI Daily Above 100 (Rising)', 'fn': lambda df, rs: detect_cci_daily_above(df),   'weekly': False},
    'cci_weekly_cross':  {'label': 'CCI Weekly Crossed 100',       'fn': lambda df, rs: detect_cci_weekly_cross(df),  'weekly': True},
    'cci_weekly_above':  {'label': 'CCI Weekly Above 100 (Rising)','fn': lambda df, rs: detect_cci_weekly_above(df),  'weekly': True},
    'cci_pullback':      {'label': 'CCI Pullback (50-100 Zone)',   'fn': lambda df, rs: detect_cci_pullback(df),      'weekly': False},
    # Price pattern screens — weekly_pivot gets weekly chart
    'ipo_base':          {'label': 'IPO Base Breakout',            'fn': lambda df, rs: detect_ipo_base(df),          'weekly': False},
    'ath_breakout':      {'label': 'ATH After Consolidation',      'fn': lambda df, rs: detect_ath_consolidation(df), 'weekly': False},
    'weekly_pivot':      {'label': 'Weekly Pivot Break',           'fn': lambda df, rs: detect_weekly_pivot_break(df),'weekly': True},
    'pocket_pivot':      {'label': 'Pocket Pivot',                 'fn': lambda df, rs: detect_pocket_pivot(df),      'weekly': False},
    'power_play':        {'label': 'Power Play (4%+ Surge)',       'fn': lambda df, rs: detect_power_play(df, rs),    'weekly': False},
    'tight_squeeze':     {'label': 'Tight Consolidation Squeeze',  'fn': lambda df, rs: detect_tight_consolidation(df),'weekly': False},
    'ma_coil':           {'label': 'MA Tightening (Coiling)',      'fn': lambda df, rs: detect_ma_tightening(df),     'weekly': False},
    'accumulation':      {'label': 'High Volume Accumulation',     'fn': lambda df, rs: detect_high_vol_accumulation(df),'weekly': False},
    'rs_leader_high':    {'label': 'RS Leader Near 52W High',      'fn': lambda df, rs: detect_rs_new_high(df, rs),   'weekly': False},
    'momentum_bo':       {'label': 'Momentum Breakout (20D High)', 'fn': lambda df, rs: detect_momentum_breakout(df), 'weekly': False},
    'first_pullback':    {'label': '1st Pullback to MA20',         'fn': lambda df, rs: detect_first_pullback(df),    'weekly': False},
}


def build_extra_screens(results_json, cache_dir, out_dir, bars, is_us,
                        manifest=None, existing_data=None):
    """
    Scan all tickers, run extra pattern detectors, generate charts.
    manifest      — if provided, use smart incremental (only regenerate stale charts)
    existing_data — preserve AI results from previous run when in daily mode
    """
    try:
        with open(results_json) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}

    seen = {}
    for scr in data.get('screens', {}).values():
        for s in scr.get('stocks', []):
            t = s.get('ticker', '')
            if t and t not in seen:
                seen[t] = s

    print(f"  Scanning {len(seen)} unique tickers for {len(EXTRA_SCREENS_DEF)} patterns…")
    buckets = {k: [] for k in EXTRA_SCREENS_DEF}

    for ticker, entry in seen.items():
        df = load_df(ticker, cache_dir, is_us)
        if df is None:
            continue
        rs = entry.get('rs_rank', 0)
        for key, defn in EXTRA_SCREENS_DEF.items():
            try:
                if defn['fn'](df, rs):
                    buckets[key].append(entry)
            except Exception:
                pass

    os.makedirs(out_dir, exist_ok=True)
    result = {}
    for key, defn in EXTRA_SCREENS_DEF.items():
        stocks     = sorted(buckets[key], key=lambda s: -s.get('rs_rank', 0))
        use_weekly = defn.get('weekly', False)
        suffix     = '_W' if use_weekly else ''

        # Build old AI lookup for this screen (preserve across daily runs)
        old_ai = {}
        if existing_data and key in existing_data:
            old_ai = {e['ticker']: e.get('ai')
                      for e in existing_data[key].get('stocks', []) if e.get('ai')}

        if not stocks:
            result[key] = {'label': defn['label'], 'stocks': []}
            continue

        print(f"  → {key} ({defn['label']}): {len(stocks)} stocks"
              f"{' [weekly]' if use_weekly else ''}")
        generated = []
        new_count  = 0
        for s in stocks:
            t     = s.get('ticker', '')
            entry = dict(s)
            entry['img'] = t + suffix + '.png'
            # Restore previous AI result
            if t in old_ai:
                entry['ai'] = old_ai[t]

            if manifest is not None:
                # Daily mode: only regenerate stale charts
                if is_stale(t, s, manifest, out_dir, suffix):
                    out_path = generate_chart(t, s, cache_dir, out_dir,
                                              is_us=is_us, bars=bars, weekly=use_weekly)
                    if out_path:
                        update_manifest(manifest, t, s, suffix)
                        generated.append(entry)
                        new_count += 1
                    # else: no cached data, skip
                else:
                    generated.append(entry)
            else:
                # Full mode: regenerate if PNG missing
                png = os.path.join(out_dir, t + suffix + '.png')
                if os.path.exists(png):
                    generated.append(entry)
                else:
                    out_path = generate_chart(t, s, cache_dir, out_dir,
                                              is_us=is_us, bars=bars, weekly=use_weekly)
                    if out_path:
                        generated.append(entry)
                        print(f"    ✓ {t}{suffix} (new)", flush=True)

        if new_count:
            print(f"    {new_count} charts regenerated", flush=True)
        result[key] = {'label': defn['label'], 'stocks': generated}
    return result


# ── CHART GENERATOR ───────────────────────────────────────────────────────────

def load_df(ticker, cache_dir, is_us=False):
    """Load cached OHLCV DataFrame for a ticker."""
    if is_us:
        fname = ticker.replace('-','_').replace('.','_') + '.pkl'
    else:
        fname = ticker + '_NS.pkl'
    path = os.path.join(cache_dir, fname)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            df = pickle.load(f)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open','High','Low','Close','Volume']].dropna()
        if len(df) < 60:
            return None
        return df
    except Exception:
        return None


def resample_weekly(df):
    """Convert daily OHLCV DataFrame to weekly OHLCV."""
    df2 = df.copy()
    if not isinstance(df2.index, pd.DatetimeIndex):
        df2.index = pd.to_datetime(df2.index)
    weekly = df2.resample('W').agg({
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum',
    }).dropna()
    return weekly


def generate_chart(ticker, entry, cache_dir, out_dir, is_us=False, bars=90, weekly=False):
    """
    Generate annotated candlestick chart PNG.
    weekly=True → resample to weekly bars, use _W.png suffix.
    Returns output path or None if failed.
    """
    df_daily = load_df(ticker, cache_dir, is_us)
    if df_daily is None:
        return None

    if weekly:
        df = resample_weekly(df_daily)
        suffix = '_W'
        bars_w = 78   # ~18 months of weekly bars
        df = df.tail(bars_w).copy()
    else:
        df = df_daily
        suffix = ''
        df = df.tail(bars).copy()
    n   = len(df)
    xs  = np.arange(n)
    op  = df['Open'].values.astype(float)
    hi  = df['High'].values.astype(float)
    lo  = df['Low'].values.astype(float)
    cl  = df['Close'].values.astype(float)
    vol = df['Volume'].values.astype(float)

    # MAs — computed on the same timeframe (weekly if weekly mode)
    full_src = resample_weekly(df_daily) if weekly else df_daily
    full_c = full_src['Close'].values.astype(float)
    # For weekly: MA10/26/52 (≈ MA20/50/200 on weekly); daily: MA20/50/200
    ma_p1, ma_p2, ma_p3 = (10, 26, 52) if weekly else (20, 50, 200)
    ma20_full  = pd.Series(full_c).rolling(ma_p1).mean().values
    ma50_full  = pd.Series(full_c).rolling(ma_p2).mean().values
    ma200_full = pd.Series(full_c).rolling(ma_p3).mean().values
    ma20  = ma20_full[-n:]
    ma50  = ma50_full[-n:]
    ma200 = ma200_full[-n:]

    # Pattern detection on full timeframe data
    full_hi = full_src['High'].values.astype(float)
    full_lo = full_src['Low'].values.astype(float)
    full_cl = full_src['Close'].values.astype(float)
    disp_bars = len(df)

    ph_full = find_pivot_highs(full_hi, left=10, right=10, lookback=disp_bars + 20)
    pl_full = find_pivot_lows(full_lo,  left=10, right=10, lookback=disp_bars + 20)

    offset_full = len(full_hi) - n
    def to_disp(idx): return idx - offset_full

    ph_disp = [(to_disp(i), v) for i, v in ph_full if 0 <= to_disp(i) < n]
    pl_disp = [(to_disp(i), v) for i, v in pl_full if 0 <= to_disp(i) < n]

    vcps = detect_vcp_contractions(full_cl, full_hi, full_lo, lookback=disp_bars+20)
    vcps_disp = [(to_disp(s), to_disp(e), hv, lv, rng)
                 for s, e, hv, lv, rng in vcps
                 if 0 <= to_disp(s) < n]

    # Horizontal S/R clusters (replace slanted trendlines)
    sr_levels = cluster_sr_levels(hi, lo, lookback=min(n, 120))

    # Base / consolidation box
    base_box = detect_base_box(cl, hi, lo, lookback=min(n, 50))

    # Breakout level
    pivot_high = entry.get('pivot_high')
    pivot_crossed = entry.get('pivot_crossed', False)

    # Pattern identification
    pct_hi   = entry.get('pct_from_high', -99)
    pct_m20  = entry.get('pct_from_ma20')
    passed   = entry.get('passed', 0)
    pat_name, pat_conf, pat_note = identify_chart_pattern(cl, hi, lo, vol, ma20, ma50)
    pattern_label = pat_name

    # ── FIGURE SETUP ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(CHART_W, CHART_H), facecolor=BG)
    ax_price = fig.add_axes([0.0, 0.22, 1.0, 0.78], facecolor=BG)
    ax_vol   = fig.add_axes([0.0, 0.00, 1.0, 0.20], facecolor=BG, sharex=ax_price)

    # ── CANDLESTICKS ──────────────────────────────────────────────────────────
    W = 0.6
    for i in range(n):
        color = UP_C if cl[i] >= op[i] else DN_C
        ax_price.plot([i, i], [lo[i], hi[i]], color=color, linewidth=0.6, zorder=2)
        ax_price.add_patch(plt.Rectangle(
            (i - W/2, min(op[i], cl[i])), W, abs(cl[i] - op[i]),
            color=color, zorder=2
        ))

    # ── VOLUME ────────────────────────────────────────────────────────────────
    vol_ma = pd.Series(vol).rolling(20).mean().values
    for i in range(n):
        vc = VOL_UP if cl[i] >= op[i] else VOL_DN
        ax_vol.bar(i, vol[i], color=vc, width=W, zorder=2)
    ax_vol.plot(xs, vol_ma, color='#64748b', linewidth=0.8, zorder=3)

    # ── MOVING AVERAGES ───────────────────────────────────────────────────────
    valid20  = ~np.isnan(ma20)
    valid50  = ~np.isnan(ma50)
    valid200 = ~np.isnan(ma200)
    if valid20.any():  ax_price.plot(xs[valid20],  ma20[valid20],  color=MA20_C,  linewidth=1.0, zorder=3, label='MA20')
    if valid50.any():  ax_price.plot(xs[valid50],  ma50[valid50],  color=MA50_C,  linewidth=1.0, zorder=3, label='MA50')
    if valid200.any(): ax_price.plot(xs[valid200], ma200[valid200],color=MA200_C, linewidth=0.8, zorder=3, label='MA200', linestyle='--')

    # ── HORIZONTAL S/R LEVELS ────────────────────────────────────────────────
    price_now = float(cl[-1])
    for lvl, touches in sr_levels:
        above = lvl > price_now
        lc = '#f8514980' if above else '#3fb95080'   # red zone = resistance, green = support
        lw = 0.5 + min(touches * 0.15, 0.6)
        ax_price.axhline(lvl, color=lc, linewidth=lw, linestyle='--', alpha=0.85, zorder=4)
        ax_price.text(n - 1, lvl,
                      f"  {'R' if above else 'S'}{touches}× {lvl:,.0f}",
                      color=lc, fontsize=5.5, va='center', ha='left', zorder=5,
                      clip_on=False)

    # ── BASE / CONSOLIDATION BOX ─────────────────────────────────────────────
    if base_box is not None:
        bh, bl, bstart = base_box
        bx0, bx1 = max(0, bstart), n - 1
        brng = (bh - bl) / bl * 100 if bl > 0 else 0
        ax_price.add_patch(mpatches.FancyBboxPatch(
            (bx0, bl), bx1 - bx0, bh - bl,
            boxstyle='square,pad=0', linewidth=0.7,
            edgecolor='#60a5fa55', facecolor='#60a5fa08', zorder=3
        ))
        ax_price.text(bx0 + 0.5, bh, f' Base {brng:.0f}%',
                      color='#60a5fa', fontsize=5.5, va='bottom', ha='left', zorder=5)

    # ── PIVOT DOTS ────────────────────────────────────────────────────────────
    if ph_disp:
        ax_price.scatter([p[0] for p in ph_disp], [p[1] for p in ph_disp],
                         color=DN_C, s=18, zorder=5, marker='v')
    if pl_disp:
        ax_price.scatter([p[0] for p in pl_disp], [p[1] for p in pl_disp],
                         color=UP_C, s=18, zorder=5, marker='^')

    # ── VCP CONTRACTION BRACKETS ──────────────────────────────────────────────
    for s_i, e_i, hv, lv, rng in vcps_disp[-3:]:
        ax_price.annotate('', xy=(e_i, lv), xytext=(s_i, hv),
                          arrowprops=dict(arrowstyle='<->', color='#fbbf24', lw=0.8))
        mid_x = (s_i + e_i) / 2
        mid_y = (hv + lv) / 2
        ax_price.text(mid_x, mid_y, f'{rng:.0f}%', color='#fbbf24',
                      fontsize=5.5, ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.15', fc='#0d1117', ec='#fbbf2440', lw=0.5))

    # ── BREAKOUT / PIVOT HIGH LEVEL ───────────────────────────────────────────
    if pivot_high and pivot_high > 0:
        piv_disp = bars - (entry.get('pivot_bars_ago') or bars)
        if 0 <= piv_disp < n:
            ax_price.axhline(pivot_high, color=PIVOT_C, linewidth=1.0,
                             linestyle='-', alpha=0.85, zorder=4)
            ax_price.text(n - 1, pivot_high, f' Pivot {pivot_high:,.0f}',
                          color=PIVOT_C, fontsize=6.5, va='center', ha='left')
            if pivot_crossed:
                ax_price.annotate('▲ BREAKOUT', xy=(n-1, cl[-1]),
                                  xytext=(n - 10, cl[-1] * 1.02),
                                  color=UP_C, fontsize=7, fontweight='bold',
                                  arrowprops=dict(arrowstyle='->', color=UP_C, lw=0.8))

    # ── AXES STYLING ──────────────────────────────────────────────────────────
    for ax in [ax_price, ax_vol]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_C, labelsize=6)
        ax.spines[:].set_color(GRID_C)
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

    ax_price.grid(True, color=GRID_C, linewidth=0.3, alpha=0.6)
    ax_vol.grid(True, color=GRID_C, linewidth=0.3, alpha=0.4)
    ax_price.set_xlim(-1, n + 2)
    ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x:,.0f}' if x >= 1000 else f'{x:.1f}'
    ))
    ax_vol.set_yticks([])
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # Date x-ticks on volume axis
    step = max(1, n // 6)
    tick_idx = list(range(0, n, step))
    ax_vol.set_xticks(tick_idx)
    ax_vol.set_xticklabels(
        [df.index[i].strftime('%b %d') for i in tick_idx],
        color=TEXT_C, fontsize=5.5
    )

    # ── HEADER TEXT ───────────────────────────────────────────────────────────
    price_str  = f"{'$' if is_us else '₹'}{cl[-1]:,.2f}"
    pct_h_str  = f"{pct_hi:+.1f}%" if pct_hi is not None else ''
    rs_str     = f"RS {entry.get('rs_rank', '?')}"
    passed_str = f"{passed}/8"
    tf_str     = 'W' if weekly else 'D'

    # Map pattern name → colour (6-char hex only so we can append alpha suffix)
    _PAT_COLORS = {
        'BREAKOUT':           '#3fb950',
        'NEAR BREAKOUT':      '#26a641',
        'CUP & HANDLE':       '#fbbf24',
        'DOUBLE BOTTOM':      '#60a5fa',
        'ASCENDING TRIANGLE': '#a78bfa',
        'BULL FLAG':          '#fb923c',
        'TIGHT BASE':         '#fbbf24',
        'STAGE 2':            '#a78bfa',
        'VCP':                '#fbbf24',
        'WATCH':              '#94a3b8',
    }
    label_color = _PAT_COLORS.get(pat_name, '#94a3b8')

    ax_price.text(0.01, 0.97, ticker, transform=ax_price.transAxes,
                  color=TEXT_C, fontsize=11, fontweight='bold', va='top')
    ax_price.text(0.01, 0.87, price_str,
                  transform=ax_price.transAxes, color=TEXT_C, fontsize=8, va='top',
                  fontfamily='monospace')
    ax_price.text(0.01, 0.80,
                  f'{pct_h_str} from high  {rs_str}  {passed_str}  [{tf_str}]',
                  transform=ax_price.transAxes, color='#94a3b8', fontsize=6.5, va='top')
    # Pattern badge (top-right)
    badge_text = f'{pat_name}  {pat_conf}%'
    ax_price.text(0.99, 0.97, badge_text, transform=ax_price.transAxes,
                  color=label_color, fontsize=7, fontweight='bold', va='top', ha='right',
                  bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec=label_color + '55', lw=0.6))
    # Pattern note (below badge)
    if pat_note:
        ax_price.text(0.99, 0.88, pat_note, transform=ax_price.transAxes,
                      color='#94a3b8', fontsize=5.5, va='top', ha='right')

    # MA legend
    ma_label1 = 'MA10' if weekly else 'MA20'
    ma_label2 = 'MA26' if weekly else 'MA50'
    ma_label3 = 'MA52' if weekly else 'MA200'
    legend_els = [
        Line2D([0],[0], color=MA20_C,  lw=1,   label=ma_label1),
        Line2D([0],[0], color=MA50_C,  lw=1,   label=ma_label2),
        Line2D([0],[0], color=MA200_C, lw=0.8, label=ma_label3, linestyle='--'),
    ]
    ax_price.legend(handles=legend_els, loc='lower left', fontsize=5.5,
                    facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT_C, framealpha=0.8)

    # ── AI VISUAL ANALYSIS OVERLAY (enabled by --ai-analysis flag) ──────────────
    # Note: chart is saved FIRST (so vision model can read the PNG), then overlay
    # is skipped here — AI results are stored in JSON and shown in the viewer instead.
    # The overlay on-chart is done in run_ai_on_screen() after the PNG exists.

    # ── SAVE ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, f'{ticker}{suffix}.png')
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0,
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    return out_path


# ── HTML GENERATOR ────────────────────────────────────────────────────────────

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chart Pattern Viewer</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;
       --muted:#8b949e;--accent:#3fb950;--amber:#fbbf24;--blue:#60a5fa;
       --violet:#a78bfa;--orange:#fb923c;--red:#f85149}}
body{{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif;min-height:100vh}}

/* TOP BAR */
.topbar{{background:var(--surface);border-bottom:1px solid var(--border);
         padding:10px 20px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100}}
.logo{{font-size:15px;font-weight:800;color:var(--accent);letter-spacing:-.5px;white-space:nowrap}}
.market-tabs{{display:flex;gap:4px}}
.mtab{{padding:5px 14px;border:1px solid var(--border);border-radius:6px;background:transparent;
       color:var(--muted);cursor:pointer;font-size:12px;font-weight:600;font-family:inherit;transition:all .15s}}
.mtab.active{{background:var(--accent);border-color:var(--accent);color:#fff}}
.screen-tabs{{display:flex;gap:4px;flex-wrap:wrap}}
.stab{{padding:4px 11px;border:1px solid var(--border);border-radius:20px;background:transparent;
       color:var(--muted);cursor:pointer;font-size:11px;font-weight:600;font-family:inherit;transition:all .15s;white-space:nowrap}}
.stab.active{{background:var(--blue);border-color:var(--blue);color:#fff}}
.stats-bar{{margin-left:auto;display:flex;gap:12px;font-size:11px;color:var(--muted);white-space:nowrap}}
.stats-bar b{{color:var(--text)}}

/* FILTER BAR */
.filterbar{{background:var(--surface);border-bottom:1px solid var(--border);
            padding:8px 20px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
.filterbar input{{background:var(--bg);border:1px solid var(--border);border-radius:6px;
                  color:var(--text);padding:5px 10px;font-size:12px;font-family:inherit;width:180px}}
.filterbar input::placeholder{{color:var(--muted)}}
.sort-sel{{background:var(--bg);border:1px solid var(--border);border-radius:6px;
           color:var(--text);padding:5px 8px;font-size:12px;font-family:inherit;cursor:pointer}}
.filter-chips{{display:flex;gap:5px;flex-wrap:wrap}}
.chip{{padding:3px 10px;border:1px solid var(--border);border-radius:12px;font-size:10px;
       font-weight:600;cursor:pointer;color:var(--muted);background:transparent;font-family:inherit;transition:all .15s}}
.chip.active{{background:rgba(96,165,250,.15);border-color:var(--blue);color:var(--blue)}}
.grid-size{{display:flex;gap:4px;margin-left:auto}}
.gsz{{padding:4px 8px;border:1px solid var(--border);border-radius:5px;font-size:11px;
      cursor:pointer;background:transparent;color:var(--muted);font-family:inherit;transition:all .15s}}
.gsz.active{{background:var(--border);color:var(--text)}}

/* CHART GRID */
.grid-container{{padding:16px 20px}}
.grid{{display:grid;gap:10px;grid-template-columns:repeat(4,1fr)}}
.grid.cols-3{{grid-template-columns:repeat(3,1fr)}}
.grid.cols-5{{grid-template-columns:repeat(5,1fr)}}
.grid.cols-2{{grid-template-columns:repeat(2,1fr)}}

.card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;
       overflow:hidden;cursor:pointer;transition:border-color .15s,transform .1s;position:relative}}
.card:hover{{border-color:#58a6ff;transform:translateY(-1px)}}
.card img{{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}}
.card-footer{{padding:6px 8px 7px;display:flex;align-items:center;justify-content:space-between;
              border-top:1px solid var(--border)}}
.card-ticker{{font-size:12px;font-weight:700;color:var(--text);font-family:'SF Mono',monospace}}
.card-price{{font-size:11px;color:var(--muted);font-family:'SF Mono',monospace}}
.card-badge{{position:absolute;top:6px;right:6px;font-size:9px;font-weight:800;padding:2px 6px;
             border-radius:10px;letter-spacing:.04em}}
.badge-BREAKOUT{{background:rgba(63,185,80,.2);color:#3fb950;border:1px solid #3fb95055}}
.badge-VCP{{background:rgba(251,191,36,.15);color:#fbbf24;border:1px solid #fbbf2455}}
.badge-NEAR.HIGH{{background:rgba(96,165,250,.15);color:#60a5fa;border:1px solid #60a5fa55}}
.badge-ON.20MA{{background:rgba(251,146,60,.15);color:#fb923c;border:1px solid #fb923c55}}
.badge-STAGE.2{{background:rgba(167,139,250,.15);color:#a78bfa;border:1px solid #a78bfa55}}
.badge-WATCH{{background:rgba(139,148,158,.1);color:#8b949e;border:1px solid #8b949e44}}

.rs-pill{{font-size:9px;font-weight:700;padding:1px 6px;border-radius:8px;
          background:rgba(96,165,250,.12);color:var(--blue);border:1px solid rgba(96,165,250,.25)}}

/* MODAL */
.modal-bg{{display:none;position:fixed;inset:0;background:#00000099;z-index:1000;
           align-items:center;justify-content:center}}
.modal-bg.open{{display:flex}}
.modal{{background:var(--surface);border:1px solid var(--border);border-radius:12px;
        max-width:95vw;max-height:92vh;overflow:auto;padding:0}}
.modal-header{{padding:12px 16px;border-bottom:1px solid var(--border);
               display:flex;align-items:center;justify-content:space-between}}
.modal-title{{font-size:14px;font-weight:700;color:var(--text)}}
.modal-close{{background:transparent;border:none;color:var(--muted);font-size:18px;cursor:pointer}}
.modal img{{width:100%;display:block;border-radius:0 0 12px 12px}}
.modal-meta{{padding:10px 16px;display:flex;gap:16px;flex-wrap:wrap;border-top:1px solid var(--border)}}
.meta-item{{display:flex;flex-direction:column;gap:2px}}
.meta-label{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:600}}
.meta-val{{font-size:13px;font-weight:700;color:var(--text);font-family:'SF Mono',monospace}}

/* NO RESULTS */
.empty{{text-align:center;padding:60px 20px;color:var(--muted);font-size:14px}}

/* RESPONSIVE */
@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,1fr)!important}}}}
@media(max-width:560px){{.grid{{grid-template-columns:1fr!important}}}}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">📊 Chart Pattern Viewer</div>
  <div class="market-tabs">
    <button class="mtab active" id="tab-india" onclick="switchMarket('india')">🇮🇳 India</button>
    <button class="mtab" id="tab-us" onclick="switchMarket('us')">🇺🇸 US</button>
  </div>
  <div class="screen-tabs" id="screen-tabs"></div>
  <div class="stats-bar" id="stats-bar"><span>Loading…</span></div>
</div>

<div class="filterbar">
  <input type="text" id="search-inp" placeholder="Search ticker…" oninput="applyFilter()">
  <select class="sort-sel" id="sort-sel" onchange="applyFilter()">
    <option value="rs_rank_desc">RS ↓</option>
    <option value="pct_from_high_desc">Closest to High ↓</option>
    <option value="passed_desc">Criteria ↓</option>
    <option value="pct_from_ma20_asc">Closest to 20MA</option>
    <option value="ticker_asc">A–Z</option>
  </select>
  <div class="filter-chips">
    <button class="chip active" data-f="all"    onclick="setChip(this)">All</button>
    <button class="chip" data-f="bo"   onclick="setChip(this)">Breakout</button>
    <button class="chip" data-f="vcp"  onclick="setChip(this)">VCP</button>
    <button class="chip" data-f="ma20" onclick="setChip(this)">On 20MA</button>
    <button class="chip" data-f="rs85" onclick="setChip(this)">RS ≥ 85</button>
    <button class="chip" data-f="ft"   onclick="setChip(this)">Full Template</button>
  </div>
  <div class="grid-size">
    <button class="gsz" data-cols="2" onclick="setCols(this)">2</button>
    <button class="gsz active" data-cols="4" onclick="setCols(this)">4</button>
    <button class="gsz" data-cols="5" onclick="setCols(this)">5</button>
  </div>
</div>

<div class="grid-container">
  <div class="grid" id="grid"></div>
</div>

<!-- MODAL -->
<div class="modal-bg" id="modal-bg" onclick="closeModal(event)">
  <div class="modal" id="modal">
    <div class="modal-header">
      <div class="modal-title" id="modal-title">—</div>
      <button class="modal-close" onclick="document.getElementById('modal-bg').classList.remove('open')">✕</button>
    </div>
    <img id="modal-img" src="" alt="">
    <div class="modal-meta" id="modal-meta"></div>
  </div>
</div>

<script>
var DATA = CHART_DATA_PLACEHOLDER;

var market       = 'india';
var currentScreen= null;
var chipFilter   = 'all';
var displayedStocks = [];

var SCREENS_INDIA = {SCREENS_INDIA_PLACEHOLDER};
var SCREENS_US    = {SCREENS_US_PLACEHOLDER};

function switchMarket(m){
  market = m;
  document.getElementById('tab-india').classList.toggle('active', m==='india');
  document.getElementById('tab-us').classList.toggle('active', m==='us');
  buildScreenTabs();
  var screens = m==='india' ? SCREENS_INDIA : SCREENS_US;
  var firstKey = Object.keys(screens)[0];
  setScreen(firstKey);
}

function buildScreenTabs(){
  var screens = market==='india' ? SCREENS_INDIA : SCREENS_US;
  var html = '';
  Object.entries(screens).forEach(function([k,v],i){
    html += '<button class="stab'+(i===0?' active':'')+'" data-key="'+k+'" onclick="setScreen(\''+k+'\')">'+v.label+' <span style="opacity:.6;font-weight:400">('+v.stocks.length+')</span></button>';
  });
  document.getElementById('screen-tabs').innerHTML = html;
}

function setScreen(key){
  currentScreen = key;
  document.querySelectorAll('.stab').forEach(function(b){
    b.classList.toggle('active', b.dataset.key === key);
  });
  applyFilter();
}

function setChip(btn){
  document.querySelectorAll('.chip').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active');
  chipFilter = btn.dataset.f;
  applyFilter();
}

function setCols(btn){
  document.querySelectorAll('.gsz').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active');
  var grid = document.getElementById('grid');
  grid.className = 'grid cols-'+btn.dataset.cols;
}

function applyFilter(){
  var screens = market==='india' ? SCREENS_INDIA : SCREENS_US;
  var scr = screens[currentScreen];
  if(!scr){ document.getElementById('grid').innerHTML = '<div class="empty">No data for this screen.</div>'; return; }

  var stocks = scr.stocks.slice();
  var query  = document.getElementById('search-inp').value.trim().toUpperCase();
  if(query) stocks = stocks.filter(function(s){ return s.ticker.includes(query); });

  if(chipFilter === 'bo')   stocks = stocks.filter(function(s){ return s.pivot_crossed; });
  if(chipFilter === 'vcp')  stocks = stocks.filter(function(s){ return s.vcp; });
  if(chipFilter === 'ma20') stocks = stocks.filter(function(s){ return s.pct_from_ma20!=null && Math.abs(s.pct_from_ma20)<=3; });
  if(chipFilter === 'rs85') stocks = stocks.filter(function(s){ return (s.rs_rank||0)>=85; });
  if(chipFilter === 'ft')   stocks = stocks.filter(function(s){ return s.passed===8; });

  var sort = document.getElementById('sort-sel').value;
  stocks.sort(function(a,b){
    if(sort==='rs_rank_desc')       return (b.rs_rank||0)-(a.rs_rank||0);
    if(sort==='pct_from_high_desc') return (b.pct_from_high||{})-(a.pct_from_high||{});
    if(sort==='passed_desc')        return (b.passed||0)-(a.passed||0);
    if(sort==='pct_from_ma20_asc')  return Math.abs(a.pct_from_ma20||99)-Math.abs(b.pct_from_ma20||99);
    if(sort==='ticker_asc')         return a.ticker.localeCompare(b.ticker);
    return 0;
  });

  displayedStocks = stocks;
  document.getElementById('stats-bar').innerHTML =
    '<span><b>'+stocks.length+'</b> stocks</span>';
  renderGrid(stocks);
}

function patternLabel(s){
  if(s.pivot_crossed)                               return 'BREAKOUT';
  if(s.vcp)                                         return 'VCP';
  if(s.pct_from_high!=null && s.pct_from_high>=-5) return 'NEAR HIGH';
  if(s.pct_from_ma20!=null && Math.abs(s.pct_from_ma20)<=3) return 'ON 20MA';
  if(s.passed===8)                                  return 'STAGE 2';
  return 'WATCH';
}

function renderGrid(stocks){
  var cur = market==='india' ? '₹' : '$';
  var imgDir = market==='india' ? 'charts_india' : 'charts_us';
  var html = '';
  if(!stocks.length){
    html = '<div class="empty">No stocks match the current filter.</div>';
    document.getElementById('grid').innerHTML = html;
    return;
  }
  stocks.forEach(function(s, idx){
    var lbl      = patternLabel(s);
    var price    = s.price!=null ? cur+(+s.price).toLocaleString('en',{{minimumFractionDigits:2,maximumFractionDigits:2}}) : '—';
    var pctH     = s.pct_from_high!=null ? (s.pct_from_high>=0?'+':'')+s.pct_from_high.toFixed(1)+'%' : '';
    var badgeCls = 'badge-'+lbl.replace(/ /g,'.');
    var imgSrc   = imgDir+'/'+(s.img||s.ticker+'.png');
    // AI action pill
    var aiHtml = '';
    if(s.ai && s.ai.action){
      var aCol = {{BUY:'#3fb950',WATCH:'#fbbf24',AVOID:'#f85149'}}[s.ai.action]||'#8b949e';
      aiHtml = '<span style="font-size:8px;font-weight:800;padding:1px 5px;border-radius:8px;'
        +'background:'+aCol+'22;color:'+aCol+';border:1px solid '+aCol+'44;margin-left:4px">'
        +s.ai.action+'</span>';
    }
    html += '<div class="card" onclick="openModal('+idx+')">'
      +'<img src="'+imgSrc+'" alt="'+s.ticker+'" loading="lazy" onerror="this.classList.add(\'img-missing\')">'
      +'<span class="card-badge '+badgeCls+'">'+lbl+'</span>'
      +'<div class="card-footer">'
      +'<span class="card-ticker">'+s.ticker+aiHtml+'</span>'
      +'<span class="card-price">'+price+' <span style="color:'+(s.pct_from_high!=null&&s.pct_from_high>=-5?\'#3fb950\':\'#8b949e\')+'">'+(pctH)+'</span></span>'
      +'<span class="rs-pill">RS '+s.rs_rank+'</span>'
      +'</div>'
      +'</div>';
  });
  document.getElementById('grid').innerHTML = html;
}

function openModal(idx){
  var s      = displayedStocks[idx];
  var cur    = market==='india' ? '₹' : '$';
  var imgDir = market==='india' ? 'charts_india' : 'charts_us';
  var lbl    = patternLabel(s);
  var imgSrc = imgDir+'/'+(s.img||s.ticker+'.png');
  document.getElementById('modal-title').textContent = s.ticker+' — '+lbl;
  document.getElementById('modal-img').src = imgSrc;
  var meta = [
    {{l:'Price',    v: s.price!=null?cur+(+s.price).toLocaleString('en',{{minimumFractionDigits:2}}):'—'}},
    {{l:'RS Rank',  v: s.rs_rank}},
    {{l:'Criteria', v: (s.passed||'?')+'/8'}},
    {{l:'From High',v: s.pct_from_high!=null?(s.pct_from_high>=0?'+':'')+s.pct_from_high.toFixed(1)+'%':'—'}},
    {{l:'From 20MA',v: s.pct_from_ma20!=null?(s.pct_from_ma20>=0?'+':'')+s.pct_from_ma20.toFixed(1)+'%':'—'}},
    {{l:'From MA50',v: s.pct_from_ma50!=null?(s.pct_from_ma50>=0?'+':'')+s.pct_from_ma50.toFixed(1)+'%':'—'}},
    {{l:'Pivot High',v: s.pivot_high!=null?cur+s.pivot_high.toLocaleString('en'):'—'}},
    {{l:'Vol Ratio', v: s.vol_ratio!=null?s.vol_ratio.toFixed(2)+'×':'—'}},
  ];
  // Append AI row if available
  if(s.ai){
    var aCol = {{BUY:'#3fb950',WATCH:'#fbbf24',AVOID:'#f85149'}}[s.ai.action]||'#8b949e';
    meta.push({{l:'AI Pattern', v: (s.ai.pattern||'')+'  '+( s.ai.conf||'')+( s.ai.conf?'%':'')}});
    meta.push({{l:'AI Action',  v: '<span style="color:'+aCol+';font-weight:800">'+(s.ai.action||'')+'</span>'}});
    if(s.ai.note) meta.push({{l:'AI Note', v: s.ai.note}});
    if(s.ai.support) meta.push({{l:'AI Support', v: cur+(+s.ai.support).toLocaleString('en')}});
    if(s.ai.resistance) meta.push({{l:'AI Resist.', v: cur+(+s.ai.resistance).toLocaleString('en')}});
  }
  document.getElementById('modal-meta').innerHTML = meta.map(function(m){{
    return '<div class="meta-item"><div class="meta-label">'+m.l+'</div><div class="meta-val">'+m.v+'</div></div>';
  }}).join('');
  document.getElementById('modal-bg').classList.add('open');
}

function closeModal(e){
  if(e.target === document.getElementById('modal-bg'))
    document.getElementById('modal-bg').classList.remove('open');
}

// Init
buildScreenTabs();
setScreen(Object.keys(SCREENS_INDIA)[0]);
</script>
</body>
</html>
'''


# ── MANIFEST (smart incremental rebuild) ──────────────────────────────────────

MANIFEST_PATH = os.path.join(SCRIPT_DIR, 'chart_manifest.json')

def load_manifest():
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def save_manifest(manifest):
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

def is_stale(ticker, entry, manifest, out_dir, suffix='', price_threshold=0.02):
    """
    Returns True if the chart should be regenerated.
    Stale when: PNG missing, ticker new, price changed ≥ threshold, or date changed.
    """
    png = os.path.join(out_dir, ticker + suffix + '.png')
    if not os.path.exists(png):
        return True
    key = ticker + suffix
    if key not in manifest:
        return True
    m = manifest[key]
    today = str(datetime.date.today())
    if m.get('date') != today:
        # Price moved significantly since last generation
        last_price = float(m.get('price', 0))
        cur_price  = float(entry.get('price', entry.get('close', 0)) or 0)
        if last_price > 0 and abs(cur_price - last_price) / last_price >= price_threshold:
            return True
        # Also stale if it's been more than 1 day (ensure daily freshness)
        return True
    return False

def update_manifest(manifest, ticker, entry, suffix=''):
    key = ticker + suffix
    manifest[key] = {
        'price': entry.get('price', entry.get('close', 0)),
        'date':  str(datetime.date.today()),
    }


# ── AI BATCH ANALYSIS (top-N per screen, results stored in JSON) ───────────────

# Priority screens where AI adds the most value
AI_PRIORITY_SCREENS = {
    'near_breakout', 'vcp_setup', 'cci_daily_cross', 'momentum_bo',
    'ipo_base', 'ath_breakout', 'pocket_pivot', 'power_play',
    'weekly_pivot', 'first_pullback', 'rs_leader_high',
}

def run_ai_on_screen(screen_data, cache_dir, out_dir, is_us, bars, ai_top, weekly=False):
    """
    Run visual AI analysis on top-N stocks: sends the chart PNG to the vision model.
    Skips stocks that already have an AI result from today.
    Updates 'ai' key in each stock entry dict in-place.
    """
    if not _AI_ENABLED:
        return
    today  = str(datetime.date.today())
    suffix = '_W' if weekly else ''
    ma_p1, ma_p2, ma_p3 = (10, 26, 52) if weekly else (20, 50, 200)
    stocks = screen_data.get('stocks', [])

    # Only analyse stocks not yet done today
    candidates = [s for s in stocks
                  if not (isinstance(s.get('ai'), dict) and s['ai'].get('date') == today)]
    candidates = candidates[:ai_top]
    if not candidates:
        print(f"    (all {len(stocks)} already analysed today)", flush=True)
        return

    mode = 'visual' if _VISION_CAPABLE else 'text'
    print(f"    Running {mode} AI on {len(candidates)} stocks…", flush=True)

    for s in candidates:
        ticker   = s.get('ticker', '')
        img_path = os.path.join(out_dir, ticker + suffix + '.png')
        df_daily = load_df(ticker, cache_dir, is_us)
        if df_daily is None:
            continue
        try:
            src      = resample_weekly(df_daily) if weekly else df_daily
            src      = src.tail(bars).copy()
            c        = src['Close'].values.astype(float)
            h        = src['High'].values.astype(float)
            l        = src['Low'].values.astype(float)
            v        = src['Volume'].values.astype(float)
            ma20_arr = pd.Series(c).rolling(ma_p1).mean().values
            ma50_arr = pd.Series(c).rolling(ma_p2).mean().values
            ma200_arr= pd.Series(c).rolling(ma_p3).mean().values
            ohlcv_sum = _build_ohlcv_summary(c, h, l, v, ma20_arr, ma50_arr, ma200_arr, s)

            result = claude_analyze(
                ticker, ohlcv_sum,
                timeframe='weekly' if weekly else 'daily',
                img_path=img_path          # ← vision model reads the actual chart PNG
            )
            if result:
                result['date'] = today
                s['ai'] = result
                vis_tag = '👁' if result.get('vision') else '📊'
                print(f"    {vis_tag} {ticker}: {result.get('pattern')} "
                      f"{result.get('conf')}% → {result.get('action')}", flush=True)
        except Exception:
            pass


# ── EXCEL EXPORT ──────────────────────────────────────────────────────────────

def export_excel(screens_data, out_path, market='India'):
    """One Excel workbook, one sheet per screen, plus a Summary sheet."""
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("  ⚠ openpyxl not installed (pip install openpyxl)")
        return

    cur   = '₹' if market == 'India' else '$'
    today = datetime.date.today().strftime('%Y-%m-%d')
    wb    = openpyxl.Workbook()
    wb.remove(wb.active)

    thin  = Side(border_style='thin', color='1E2733')
    bdr   = Border(left=thin, right=thin, top=thin, bottom=thin)
    hdr_fill = PatternFill('solid', fgColor='0D1117')
    alt_fill = PatternFill('solid', fgColor='111820')
    buy_fill  = PatternFill('solid', fgColor='1A3A1A')
    wtch_fill = PatternFill('solid', fgColor='3D2B00')
    avd_fill  = PatternFill('solid', fgColor='3A1010')

    COLS = [
        ('Ticker',10),('Price',10),('RS',6),('Criteria',8),
        ('%Hi',8),('%MA20',8),('%MA50',8),('Pivot',10),('VolRatio',8),
        ('AI Pattern',16),('AI Conf',8),('AI Action',9),
        ('AI Note',30),('AI Support',10),('AI Resist',10),('Date',11),
    ]

    for key, scr in screens_data.items():
        stocks = scr.get('stocks', [])
        if not stocks:
            continue
        label = scr.get('label', key)
        safe  = ''.join(c for c in label if c not in r'\/*?:[]\x00')[:31]
        ws    = wb.create_sheet(title=safe or key[:31])
        ws.sheet_properties.tabColor = '3FB950'
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = f'A1:{get_column_letter(len(COLS))}1'

        # Header
        for ci, (h, _) in enumerate(COLS, 1):
            c = ws.cell(row=1, column=ci, value=h)
            c.fill = hdr_fill
            c.font = Font(bold=True, color='8B949E', size=9, name='Consolas')
            c.alignment = Alignment(horizontal='center', vertical='center')
            c.border = bdr
        ws.row_dimensions[1].height = 24

        # Data
        for ri, s in enumerate(stocks, 2):
            ai  = s.get('ai') or {}
            act = ai.get('action', '')
            row = [
                s.get('ticker',''), s.get('price'), s.get('rs_rank'),
                f"{s.get('passed','?')}/8",
                s.get('pct_from_high'), s.get('pct_from_ma20'), s.get('pct_from_ma50'),
                s.get('pivot_high'), s.get('vol_ratio'),
                ai.get('pattern',''), ai.get('conf'), act,
                ai.get('note',''), ai.get('support'), ai.get('resistance'), today,
            ]
            rfill = alt_fill if ri % 2 == 0 else None
            for ci, val in enumerate(row, 1):
                cell = ws.cell(row=ri, column=ci, value=val)
                cell.border    = bdr
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.font      = Font(size=9, name='Consolas', color='C9D1D9')
                if rfill: cell.fill = rfill
                # AI action colour
                if ci == 12 and act:
                    cell.fill = {'BUY':buy_fill,'WATCH':wtch_fill,'AVOID':avd_fill}.get(act, rfill or PatternFill())
                    cell.font = Font(bold=True, size=9, name='Consolas',
                                     color={'BUY':'3FB950','WATCH':'FBBF24','AVOID':'F85149'}.get(act,'C9D1D9'))
                # % columns
                if ci in (5,6,7) and isinstance(val,(int,float)):
                    cell.value = f"{val:+.1f}%"
                    cell.font  = Font(size=9, name='Consolas',
                                      color='2ECC71' if val>=0 else 'E74C3C')
                # Price columns
                if ci in (2,8,14,15) and isinstance(val,(int,float)):
                    cell.number_format = '#,##0.00'

        for ci,(_, w) in enumerate(COLS, 1):
            ws.column_dimensions[get_column_letter(ci)].width = w

    # Summary sheet
    ws0 = wb.create_sheet('Summary', 0)
    ws0.sheet_properties.tabColor = '60A5FA'
    ws0.column_dimensions['A'].width = 32
    for col, w in [('B',8),('C',8),('D',10)]:
        ws0.column_dimensions[col].width = w
    ws0['A1'] = f'{market} Screens — {today}'
    ws0['A1'].font = Font(bold=True, size=11, color='E6EDF3', name='Consolas')
    ws0.merge_cells('A1:D1')
    for col, hdr in zip('ABCD', ['Screen','Stocks','AI BUY','AI WATCH']):
        c = ws0[f'{col}3']
        c.value = hdr
        c.font  = Font(bold=True, color='8B949E', size=9, name='Consolas')
        c.fill  = hdr_fill
        c.border = bdr
    for ri, (key, scr) in enumerate(screens_data.items(), 4):
        stks   = scr.get('stocks', [])
        n_buy  = sum(1 for s in stks if (s.get('ai') or {}).get('action')=='BUY')
        n_wtch = sum(1 for s in stks if (s.get('ai') or {}).get('action')=='WATCH')
        ws0.cell(ri,1,scr.get('label',key)).font = Font(size=9,color='DCE6F0',name='Consolas')
        ws0.cell(ri,2,len(stks)).font             = Font(size=9,color='DCE6F0',name='Consolas')
        ws0.cell(ri,3,n_buy).font                 = Font(size=9,color='3FB950',bold=True,name='Consolas')
        ws0.cell(ri,4,n_wtch).font                = Font(size=9,color='FBBF24',bold=True,name='Consolas')

    wb.save(out_path)
    print(f"  ✓ Excel → {os.path.basename(out_path)}  ({len(wb.sheetnames)-1} screens)")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def load_screen_stocks(results_json, screen_keys):
    """Load stocks from specified screen keys. Returns dict of {key: {label, stocks}}."""
    try:
        with open(results_json) as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR loading {results_json}: {e}")
        return {}
    screens = data.get('screens', {})
    out = {}
    for key in screen_keys:
        if key in screens:
            out[key] = {
                'label':  screens[key].get('label', key),
                'stocks': screens[key].get('stocks', [])
            }
    return out


def _regen_screen(stocks, cache_dir, out_dir, is_us, bars, manifest,
                  weekly=False, label='', ai_top=0):
    """
    Generate charts for a list of stocks, skipping up-to-date ones.
    Updates manifest in-place. Returns list of stock entries (with 'img' set).
    """
    suffix  = '_W' if weekly else ''
    today   = str(datetime.date.today())
    generated = []
    new_count = 0
    for s in stocks:
        ticker = s.get('ticker', '')
        entry  = dict(s)
        entry['img'] = ticker + suffix + '.png'
        if is_stale(ticker, s, manifest, out_dir, suffix):
            out = generate_chart(ticker, s, cache_dir, out_dir,
                                 is_us=is_us, bars=bars, weekly=weekly)
            if out:
                update_manifest(manifest, ticker, s, suffix)
                generated.append(entry)
                new_count += 1
                print(f"  ↺ {ticker}{suffix}", end='  ', flush=True)
            # else: no cache data, skip silently
        else:
            generated.append(entry)
    if new_count:
        print(f"\n  → {label}: {len(generated)} stocks ({new_count} regenerated)")
    return generated


def main():
    parser = argparse.ArgumentParser(
        description='Chart Pattern Viewer — generates annotated chart PNGs + JSON data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  (default)      Full rebuild — regenerates every chart
  --daily        Smart incremental: only new/changed charts + AI on top stocks
  --new-only     Skip base screens, only run extra pattern screens
  --ai-analysis  Enable local Ollama AI (llama3.2) on generated charts

Daily workflow (run every morning after scanner):
  python3.10 generate_charts.py --daily --market both
  python3.10 generate_charts.py --daily --market both --ai-analysis
"""
    )
    parser.add_argument('--market',      default='both', choices=['india','us','both'])
    parser.add_argument('--max',         type=int, default=9999,
                        help='Max charts per base screen')
    parser.add_argument('--bars',        type=int, default=90,
                        help='Candles to display per chart')
    parser.add_argument('--daily',       action='store_true',
                        help='Incremental mode: only regenerate changed charts (fast, for daily use)')
    parser.add_argument('--new-only',    action='store_true',
                        help='Skip base screens, only run extra pattern screens')
    parser.add_argument('--ai-analysis', action='store_true',
                        help='Run Ollama AI analysis (llama3.2, free, offline)')
    parser.add_argument('--ai-top',      type=int, default=25,
                        help='AI analysis: top N stocks per priority screen (default 25)')
    args = parser.parse_args()

    # ── AI setup ─────────────────────────────────────────────────────────────
    if args.ai_analysis:
        if init_ai_client():
            print(f"✓ Ollama AI ready ({_OLLAMA_MODEL}) — top {args.ai_top} per priority screen")
        else:
            print("⚠  Ollama not reachable. Start with: ollama serve")

    INDIA_SCREENS = ['full_template','near_breakout','vcp_setup','high_ma20','rs_leaders','new_highs']
    US_SCREENS    = ['full_template','near_breakout','vcp_setup','high_ma20','rs_leaders','near_base_pivot']

    # ── Load manifest + existing JSON ────────────────────────────────────────
    manifest = load_manifest() if args.daily else {}

    def load_existing_json(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}

    preserve = args.new_only or args.daily
    india_data = load_existing_json(os.path.join(SCRIPT_DIR, 'india_data.json')) if preserve else {}
    us_data    = load_existing_json(os.path.join(SCRIPT_DIR, 'us_data.json'))    if preserve else {}

    t0 = datetime.datetime.now()

    # ── INDIA base screens ────────────────────────────────────────────────────
    if args.market in ('india', 'both') and not args.new_only:
        print("\n=== INDIA CHARTS ===")
        os.makedirs(OUT_INDIA, exist_ok=True)
        india_screens = load_screen_stocks(INDIA_JSON, INDIA_SCREENS)
        for key, scr in india_screens.items():
            stocks = sorted(scr['stocks'], key=lambda s: -s.get('rs_rank', 0))[:args.max]
            if args.daily:
                generated = _regen_screen(stocks, INDIA_CACHE, OUT_INDIA, False,
                                          args.bars, manifest, label=scr['label'])
            else:
                generated = []
                for s in stocks:
                    ticker = s.get('ticker', '')
                    out = generate_chart(ticker, s, INDIA_CACHE, OUT_INDIA, is_us=False, bars=args.bars)
                    if out:
                        generated.append(s)
                        print(f"  ✓ {ticker}", end='  ', flush=True)
                print(f"\n  → {key}: {len(generated)} charts")
            # Preserve existing AI results for stocks already in JSON
            if args.daily and key in india_data:
                old_ai = {e['ticker']: e.get('ai') for e in india_data[key].get('stocks', []) if e.get('ai')}
                for e in generated:
                    if e.get('ticker') in old_ai and not e.get('ai'):
                        e['ai'] = old_ai[e['ticker']]
            india_data[key] = {'label': scr['label'], 'stocks': generated}

    # ── US base screens ───────────────────────────────────────────────────────
    if args.market in ('us', 'both') and not args.new_only:
        print("\n=== US CHARTS ===")
        os.makedirs(OUT_US, exist_ok=True)
        us_screens = load_screen_stocks(US_JSON, US_SCREENS)
        for key, scr in us_screens.items():
            stocks = sorted(scr['stocks'], key=lambda s: -s.get('rs_rank', 0))[:args.max]
            if args.daily:
                generated = _regen_screen(stocks, US_CACHE, OUT_US, True,
                                          args.bars, manifest, label=scr['label'])
            else:
                generated = []
                for s in stocks:
                    ticker = s.get('ticker', '')
                    out = generate_chart(ticker, s, US_CACHE, OUT_US, is_us=True, bars=args.bars)
                    if out:
                        generated.append(s)
                        print(f"  ✓ {ticker}", end='  ', flush=True)
                print(f"\n  → {key}: {len(generated)} charts")
            if args.daily and key in us_data:
                old_ai = {e['ticker']: e.get('ai') for e in us_data[key].get('stocks', []) if e.get('ai')}
                for e in generated:
                    if e.get('ticker') in old_ai and not e.get('ai'):
                        e['ai'] = old_ai[e['ticker']]
            us_data[key] = {'label': scr['label'], 'stocks': generated}

    # ── Extra pattern screens ─────────────────────────────────────────────────
    if args.market in ('india', 'both'):
        print("\n=== INDIA — Extra Pattern Screens ===")
        extra_india = build_extra_screens(
            INDIA_JSON, INDIA_CACHE, OUT_INDIA, args.bars, is_us=False,
            manifest=manifest if args.daily else None,
            existing_data=india_data if args.daily else None,
        )
        india_data.update(extra_india)

    if args.market in ('us', 'both'):
        print("\n=== US — Extra Pattern Screens ===")
        extra_us = build_extra_screens(
            US_JSON, US_CACHE, OUT_US, args.bars, is_us=True,
            manifest=manifest if args.daily else None,
            existing_data=us_data if args.daily else None,
        )
        us_data.update(extra_us)

    # ── AI batch analysis on priority screens ─────────────────────────────────
    if args.ai_analysis and _AI_ENABLED and args.ai_top > 0:
        print(f"\n=== AI ANALYSIS (top {args.ai_top} per priority screen) ===")
        for key, scr in india_data.items():
            if key in AI_PRIORITY_SCREENS:
                weekly = EXTRA_SCREENS_DEF.get(key, {}).get('weekly', False)
                print(f"  {key} ({scr['label']})…")
                run_ai_on_screen(scr, INDIA_CACHE, OUT_INDIA, False, args.bars, args.ai_top, weekly)
        for key, scr in us_data.items():
            if key in AI_PRIORITY_SCREENS:
                weekly = EXTRA_SCREENS_DEF.get(key, {}).get('weekly', False)
                print(f"  {key} US ({scr['label']})…")
                run_ai_on_screen(scr, US_CACHE, OUT_US, True, args.bars, args.ai_top, weekly)

    # ── Save manifest ─────────────────────────────────────────────────────────
    if args.daily:
        save_manifest(manifest)

    # ── Write JSON data files ─────────────────────────────────────────────────
    with open(os.path.join(SCRIPT_DIR, 'india_data.json'), 'w') as f:
        json.dump(india_data, f, default=str)
    with open(os.path.join(SCRIPT_DIR, 'us_data.json'), 'w') as f:
        json.dump(us_data, f, default=str)

    # ── Excel export (one sheet per screen) ───────────────────────────────────
    export_excel(india_data, os.path.join(SCRIPT_DIR, 'india_screens.xlsx'), market='India')
    export_excel(us_data,    os.path.join(SCRIPT_DIR, 'us_screens.xlsx'),    market='US')

    elapsed = (datetime.datetime.now() - t0).seconds
    total   = sum(len(v['stocks']) for v in india_data.values()) + \
              sum(len(v['stocks']) for v in us_data.values())
    print(f"\n✓ Done in {elapsed//60}m {elapsed%60}s — "
          f"{total} stocks across {len(india_data)+len(us_data)} screens")
    print(f"  Reload:  http://localhost:8765/chart_viewer.html")
    print(f"  Excel:   india_screens.xlsx  /  us_screens.xlsx")


if __name__ == '__main__':
    main()

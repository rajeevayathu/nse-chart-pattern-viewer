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

import os, re, sys, json, pickle, argparse, datetime, warnings
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

CHART_W, CHART_H = 14, 7.875  # inches  (16:9, bigger canvas for sharpness)
WEEKLY_BARS = 260                # 5 years of weekly bars — the long-term trend view
DPI = 150                       # sharper output

# ── COLOUR PALETTE  (Gandiva design system — see assets/gandiva.css) ─────────
# Light theme. Values mirror the UI tokens exactly, so a chart sits on the
# dashboard's paper ground as one continuous surface.
BG       = '#FFFFFF'   # figure background — --g-raise-1
PANEL    = '#FDFDFE'   # chart panel
GRID_C   = '#E7EBEF'   # grid lines, barely there on purpose
UP_C     = '#1E7A52'   # --g-up
DN_C     = '#B4392E'   # --g-down
VOL_UP   = '#7FB39B'   # volume reads quieter than price
VOL_DN   = '#D3A09A'
# Volume has to stay legible on paper: dry-up and surge are both primary
# Minervini signals, so the bars carry real weight rather than a faint tint.
VOL_C    = '#C3CDD6'   # neutral volume bar
VOL_MA_C = '#71808E'   # volume MA — a reference line, not a signal
# The three moving averages are one hue stepped across three values, so the
# eye reads them as a family (fast/medium/slow) rather than three competing
# signals. On paper the ramp runs light-to-dark the other way, so the slow
# MA is the heaviest mark — the steps stay separable where they converge.
MA20_C   = '#D9AE46'   # bright brass (fast)
MA50_C   = '#9A7B28'   # true brass   (medium) — --g-brass
MA200_C  = '#5E4B1B'   # deep brass   (slow)
SR_RES_C = '#B4392E'   # resistance
SR_SUP_C = '#1E7A52'   # support
PIVOT_C  = '#8F6212'   # --g-warn
TEXT_C   = '#12171D'   # --g-text
MUTED_C  = '#6A7582'   # --g-text-3
AVWAP_C  = ('#2C5E92', '#7A3FA0')   # anchored VWAP from HL1 (info blue) / HL2 (violet)
LABEL_BG = '#FFFFFF'   # --g-raise-2

# ── PATTERN DETECTION ─────────────────────────────────────────────────────────

def find_pivot_highs(high, left=10, right=10, lookback=150):
    """
    Pine Script ta.pivothigh(left, right): bar i is a pivot high iff high[i] is the
    UNIQUE maximum of the window [i-left .. i+right] — strictly greater than all
    surrounding bars with no ties allowed.
    """
    h = np.array(high[-lookback:] if len(high) > lookback else high, dtype=float)
    offset = max(0, len(high) - lookback)
    pivots = []
    for i in range(left, len(h) - right):
        window = h[i - left: i + right + 1]
        wmax = float(np.max(window))
        if float(h[i]) == wmax and np.sum(window == wmax) == 1:
            pivots.append((offset + i, float(h[i])))
    return pivots


def find_pivot_lows(low, left=10, right=10, lookback=150):
    """
    Pine Script ta.pivotlow(left, right): bar i is a pivot low iff low[i] is the
    UNIQUE minimum of the window [i-left .. i+right] — strictly lower than all
    surrounding bars with no ties allowed.
    """
    l = np.array(low[-lookback:] if len(low) > lookback else low, dtype=float)
    offset = max(0, len(low) - lookback)
    pivots = []
    for i in range(left, len(l) - right):
        window = l[i - left: i + right + 1]
        wmin = float(np.min(window))
        if float(l[i]) == wmin and np.sum(window == wmin) == 1:
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

    ph = find_pivot_highs(h, left=10, right=10, lookback=lookback)
    pl = find_pivot_lows(l,  left=10, right=10, lookback=lookback)

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

# Minervini RAG — loaded lazily; works without the index (graceful degradation)
try:
    import minervini_rag as _mrag
    _RAG_AVAILABLE = os.path.exists(_mrag.INDEX_PATH)
except ImportError:
    _mrag = None
    _RAG_AVAILABLE = False

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
        try:
            from PIL import Image as _PILImage
            import io as _io
            with _PILImage.open(img_path) as _im:
                _im.thumbnail((900, 506), _PILImage.LANCZOS)  # ~half-res for fast inference
                _buf = _io.BytesIO()
                _im.save(_buf, format='PNG', optimize=True)
                payload['images'] = [_b64.b64encode(_buf.getvalue()).decode()]
        except Exception:
            with open(img_path, 'rb') as f:
                payload['images'] = [_b64.b64encode(f.read()).decode()]
    body = _json.dumps(payload).encode()
    req  = _urllib_req.Request(f'{_OLLAMA_URL}/api/generate', data=body,
                               headers={'Content-Type': 'application/json'})
    resp = _urllib_req.urlopen(req, timeout=600)  # 10 min — vision inference can be slow
    return _json.loads(resp.read()).get('response', '').strip()


def claude_analyze(ticker, ohlcv_summary, timeframe='daily', img_path=None, entry=None):
    """
    Visual AI analysis grounded in Minervini's book via RAG.
    Sends chart PNG to Ollama vision model; falls back to text-only if needed.
    Returns dict: {pattern, conf, support, resistance, note, action} or None.
    """
    if not _AI_ENABLED:
        return None
    import json as _json

    tf  = 'weekly' if timeframe == 'weekly' else 'daily'
    ent = entry or {}

    # ── RAG: retrieve relevant Minervini principles ───────────────────────────
    rag_ctx = ''
    if _RAG_AVAILABLE and _mrag is not None:
        try:
            rag_ctx = _mrag.get_minervini_context(
                pattern   = str(ent.get('ai', {}).get('pattern', '')),
                stage     = 2 if ent.get('passed', 0) >= 6 else None,
                pivot_pct = float(ent.get('pct_from_pivot') or ent.get('pct_from_pivot_w') or 99),
                rs        = int(ent.get('rs_rank') or 0),
            )
        except Exception:
            rag_ctx = ''

    rag_block = f"\n{rag_ctx}\n" if rag_ctx else ''

    # ── BUILD PROMPT ──────────────────────────────────────────────────────────
    persona = (
        "You are Mark Minervini, champion stock trader and author of "
        "'Trade Like a Stock Market Wizard'. "
        "You apply your SEPA (Specific Entry Point Analysis) methodology strictly.\n"
        "Your rules: Stage 2 uptrend required, VCP base preferred, volume must dry up "
        "in the base and surge on breakout, always cut losses at 7-10%, "
        "only buy within 5% of a proper pivot point.\n"
    )

    price_now = float(ent.get('price') or 0)
    json_schema = (
        'Reply with ONLY valid JSON on one line, no markdown, no explanation:\n'
        '{"pattern":"<VCP|CUP & HANDLE|BREAKOUT|BULL FLAG|DOUBLE BOTTOM|'
        'ASCENDING TRIANGLE|TIGHT BASE|FLAT BASE|STAGE 2|STAGE 4|CHOPPY>",'
        '"conf":<0-100>,'
        '"support":<nearest support price>,'
        '"resistance":<nearest resistance price>,'
        f'"entry":<exact pivot/buy-point price — current price is {price_now:.2f}>,'
        '"stop":<stop loss price — just below support or recent pivot low>,'
        '"target":<price target — minimum 2:1 risk-reward from entry>,'
        '"entry_rule":"<max 60 chars: exact Minervini entry trigger>","note":"<max 60 chars: key observation>",'
        '"action":"<BUY|WATCH|AVOID>"}'
    )

    if _VISION_CAPABLE and img_path and os.path.exists(img_path):
        # ── VISUAL MODE ───────────────────────────────────────────────────────
        prompt = (
            f"{persona}"
            f"{rag_block}"
            f"Now look at this {tf} candlestick chart for {ticker}.\n\n"
            f"Numerical context:\n{ohlcv_summary}\n\n"
            f"Apply your SEPA checklist visually:\n"
            f"1. Stage: Is price above MA50, MA50 above MA150, MA150 above MA200? (Stage 2)\n"
            f"2. VCP: Do you see tightening price contractions with declining volume in the base?\n"
            f"3. Pivot: Is there a clear buy point (handle high, base high, tight area high)?\n"
            f"4. Volume: Volume drying up in base, then a surge on the breakout bar?\n"
            f"5. Moving averages: Price stacked above MA20 > MA50 > MA200?\n"
            f"6. RS: Is this stock trending up relative to the market?\n\n"
            f"{json_schema}"
        )
    else:
        # ── TEXT-ONLY FALLBACK ────────────────────────────────────────────────
        prompt = (
            f"{persona}"
            f"{rag_block}"
            f"Analyze {ticker} ({tf}) from this data:\n\n{ohlcv_summary}\n\n"
            f"Apply SEPA: check stage, base quality, volume pattern, pivot proximity.\n\n"
            f"{json_schema}"
        )

    try:
        text = _call_ollama(prompt, img_path=img_path if _VISION_CAPABLE else None,
                            num_predict=280)
        s, e = text.find('{'), text.rfind('}') + 1
        if s >= 0 and e > s:
            result = _json.loads(text[s:e])
            if 'confidence' in result and 'conf' not in result:
                v = result.pop('confidence')
                result['conf'] = int(float(v) * 100 if float(v) <= 1.0 else float(v))
            result['vision'] = _VISION_CAPABLE
            result['rag']    = bool(rag_ctx)
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


def detect_higher_low_pivot(df):
    """
    Strong uptrend stock that just formed a confirmed new higher low pivot.
    Conditions:
      - Price > MA50 > MA200 (uptrend structure intact)
      - Two recent confirmed pivot lows where the newer one is >2% higher
      - The new pivot low confirmed within the last 20 bars (fresh)
      - Price has bounced above the pivot (not still sitting at the low)
      - Not too extended from the pivot yet (<15%), catching the early bounce
    """
    c  = df['Close'].values
    lo = df['Low'].values
    if len(c) < 60:
        return False

    price = float(c[-1])
    ma50  = float(np.mean(c[-50:])) if len(c) >= 50 else None
    ma200 = float(np.mean(c[-200:])) if len(c) >= 200 else None

    # Must be in uptrend: price above MA50, MA50 above MA200
    if ma50 is None or price < ma50:
        return False
    if ma200 is not None and ma50 < ma200:
        return False

    # Find pivot lows — left=10, right=10 filters out minor pullbacks (matches chart drawing standard)
    pls = find_pivot_lows(lo, left=10, right=10, lookback=200)
    if len(pls) < 2:
        return False

    pl_new  = pls[-1]  # most recent confirmed pivot low (abs_index, price)
    pl_prev = pls[-2]  # prior pivot low

    # Genuine higher low: new pivot must be >2% above previous
    if (pl_new[1] / pl_prev[1] - 1) * 100 < 2:
        return False

    # New pivot must be fresh — confirmed within last 20 bars
    bars_since_new = len(c) - pl_new[0]
    if bars_since_new > 20:
        return False

    # Prior pivot must be within 120 bars (same trend leg, not ancient history)
    bars_since_prev = len(c) - pl_prev[0]
    if bars_since_prev > 120:
        return False

    # Price bouncing: must be above the new pivot low
    if price <= pl_new[1] * 1.005:
        return False

    # Not already too extended from the pivot (early bounce only, <15%)
    pct_above = (price / pl_new[1] - 1) * 100
    if pct_above > 15:
        return False

    return True


def detect_cci_best_setups(df, rs=0):
    """
    Minervini candidate (full 8/8 template OR within 5% of 52W high)
    AND CCI34 >= 100 on daily OR weekly timeframe.
    Mirrors scanner 'cci34_best_setups' screen.
    """
    c = df['Close'].values.flatten().astype(float)
    h = df['High'].values.flatten().astype(float)
    l = df['Low'].values.flatten().astype(float)
    if len(c) < 210:
        return False

    price    = c[-1]
    ma50     = float(np.mean(c[-50:]))
    ma150    = float(np.mean(c[-150:]))
    ma200    = float(np.mean(c[-200:]))
    ma200_1m = float(np.mean(c[-221:-21])) if len(c) >= 221 else ma200
    hi52     = float(np.max(h[-252:])) if len(h) >= 252 else float(np.max(h))
    lo52     = float(np.min(l[-252:])) if len(l) >= 252 else float(np.min(l))

    # Full template (c1–c8)
    c1 = price > ma150 and price > ma200
    c2 = ma150 > ma200
    c3 = ma200 > ma200_1m
    c4 = ma50 > ma150 and ma50 > ma200
    c5 = price > ma50
    c6 = lo52 > 0 and price >= lo52 * 1.25
    c7 = hi52 > 0 and price >= hi52 * 0.75
    c8 = rs >= 70
    full = c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8

    # Near breakout: within 5% of 52W high (even without full template)
    near_bo = hi52 > 0 and price >= hi52 * 0.95

    if not (full or near_bo):
        return False

    # CCI34 >= 100 on daily OR weekly
    return detect_cci_daily_above(df) or detect_cci_weekly_above(df)


def detect_ath_leaders(df):
    """
    Price within 5% of all-time high (full history).
    Mirrors the scanner dashboard 'All-Time Highs' view: pct_from_ath >= -5%.
    Minimal trend filter: price > MA200 to exclude pump-and-dump spikes.
    """
    c = df['Close'].values.flatten().astype(float)
    h = df['High'].values.flatten().astype(float)
    if len(c) < 60:
        return False

    price = c[-1]
    ma200 = float(np.mean(c[-200:])) if len(c) >= 200 else float(np.mean(c))
    if price < ma200:
        return False

    ath = float(np.max(h))
    return ath > 0 and price >= ath * 0.95


def detect_higher_low_pivot_any_ext(df):
    """
    Same as detect_higher_low_pivot but no upper extension limit —
    catches stocks that have already moved well above their pivot low.
    """
    c  = df['Close'].values
    lo = df['Low'].values
    if len(c) < 60:
        return False

    price = float(c[-1])
    ma50  = float(np.mean(c[-50:])) if len(c) >= 50 else None
    ma200 = float(np.mean(c[-200:])) if len(c) >= 200 else None

    if ma50 is None or price < ma50:
        return False
    if ma200 is not None and ma50 < ma200:
        return False

    pls = find_pivot_lows(lo, left=10, right=10, lookback=200)
    if len(pls) < 2:
        return False

    pl_new  = pls[-1]
    pl_prev = pls[-2]

    if (pl_new[1] / pl_prev[1] - 1) * 100 < 2:
        return False
    if (len(c) - pl_new[0]) > 20:
        return False
    if (len(c) - pl_prev[0]) > 120:
        return False
    if price <= pl_new[1] * 1.005:
        return False

    # No upper extension limit
    return True


# ── LOCALLY-COMPUTED SCREENS ──────────────────────────────────────────────────
# The chart viewer mirrors the scanner exactly (see mirrored_screen_keys below);
# these two are the ONLY screens it computes itself, because the scanner has no
# equivalent. Everything else must match the scanner stock-for-stock, so adding
# a detector here would reintroduce the drift this design removes.
#
# The other detect_* functions remain defined above but are intentionally not
# wired in — they are kept so a screen can be reinstated without rewriting it.
EXTRA_SCREENS_DEF = {
    'higher_low_pivot':  {'label': 'Higher Low Pivot',            'fn': lambda df, rs: detect_higher_low_pivot(df),         'weekly': False},
    'hl_pivot_any_ext':  {'label': 'Higher Low Pivot (Any Ext)',  'fn': lambda df, rs: detect_higher_low_pivot_any_ext(df), 'weekly': False},
}

# Mirrored screens whose charts should be drawn on the weekly timeframe, since
# the underlying scanner screen is a weekly signal.
# ── SIDEBAR LAYOUT ────────────────────────────────────────────────────────────
# Section headings and their screens, mirroring the scanner dashboard's own
# sidebar so both views read the same way. Emitted into the data JSON under
# "_sidebar"; the viewer renders from it and falls back to a flat list if it
# is absent. Screens present in the data but missing from this layout are
# appended under "Other" rather than being silently hidden.
SIDEBAR_LAYOUT = [
    ('Market Overview',          ['market_breadth', 'newly_added']),
    ('Individual Criteria',      ['c1','c2','c3','c4','c5','c6','c7','c8']),
    ('Composite Screens',        ['full_template','near_breakout','rs_leaders',
                                  'vcp_setup','new_highs','ath_highs','ath_within_10']),
    ('RS Line',                  ['rs_trending','rs_trending_minervini']),
    ('Pivot Zones',              ['daily_near_pivot','weekly_near_pivot','near_base_pivot',
                                  'ai_near_pivot','watch_list','c1_to_c6']),
    ('Higher Low Pivot',         ['higher_low_pivot','hl_pivot_any_ext']),
    ('Anchored VWAP',            ['avwap_hl_stage2','avwap_hl_near']),
    ('Live Signals',             ['v4_live_signal','breakout_alert']),
    ('Minervini Trades',         ['minervini_picks','minervini_backtest']),
    ('Recent IPOs',              ['ipo_base','ipo_near_pivot','ipo_watch']),
    ('CCI(34) Scanner',          ['cci34_best_setups','cci34_daily_cross_100',
                                  'cci34_weekly_cross_100','cci34_daily_100',
                                  'cci34_weekly_100','cci34_daily_neg100',
                                  'cci34_weekly_neg100','strong_earnings']),
    ('HTF & Trend Setups',       ['htf_setup','htf_potential','ma_pullback','hhhl_pullback',
                                  'weekly_hl_pivot','high_ma20','high_ma50',
                                  'primary_base_new','primary_base_10yr']),
    ('Youth Screens',            ['young_3yr','young_10yr']),
    ('F&O',                      ['fo_momentum','fo_strong_uptrend']),
    ('Chartink Live (Full NSE)', ['ck_mtf_100','ck_daily_100','ck_weekly_100',
                                  'ck_daily_cross','ck_weekly_cross']),
]


def build_sidebar(data):
    """Sections in display order, keeping only screens this market actually has."""
    used, out = set(), []
    for title, keys in SIDEBAR_LAYOUT:
        present = [k for k in keys if k in data]
        if present:
            out.append({'title': title, 'screens': present})
            used.update(present)
    leftover = [k for k in data if k not in used and not k.startswith('_')]
    if leftover:
        out.append({'title': 'Other', 'screens': sorted(leftover)})
    return out


WEEKLY_MIRROR_SCREENS = {
    'cci34_weekly_100', 'cci34_weekly_cross_100', 'cci34_weekly_neg100',
    'weekly_hl_pivot', 'ck_weekly_100', 'ck_weekly_cross',
    'near_base_pivot', 'ai_near_pivot',
    # Derived screen: its signal is measured on the weekly pivot, so a daily
    # chart shows the wrong timeframe for what put the stock on the list.
    'weekly_near_pivot',
}
# Note: c6/c7/new_highs mention "52-week", but that is a lookback window on
# daily bars, not a weekly timeframe — they stay daily.


def mirrored_screen_keys(results_json):
    """
    Every screen the scanner produced, in its original order.

    Read dynamically rather than hardcoded so the viewer tracks the scanner
    automatically — a screen added to scanner.py appears here on the next run
    with no change to this file.
    """
    try:
        with open(results_json) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ERROR reading {results_json}: {e}")
        return []
    return [k for k in data.get('screens', {}) if k not in EXTRA_SCREENS_DEF]


# ── DERIVED SCREENS ───────────────────────────────────────────────────────────
# A handful of dashboard screens are not stored in results.json — dashboard.html
# builds them in the browser by filtering the union of every other screen. The
# viewer has to reproduce them from the same source, or its sidebar would be
# missing screens the scanner clearly shows.
#
# Each filter below is a direct port of the matching _build*Data() function in
# dashboard.html. If one of those changes, change its twin here.
#
# Deliberately NOT ported: market_breadth (a chart, not a stock list) and
# minervini_picks (LLM output, not a deterministic filter).

def _union_of_screens(data):
    """Every stock across every screen, de-duplicated by ticker, first wins."""
    seen, out = set(), []
    for scr in data.get('screens', {}).values():
        for s in scr.get('stocks', []):
            t = s.get('ticker')
            if t and t not in seen:
                seen.add(t)
                out.append(s)
    return out


def _num(v):
    return v if isinstance(v, (int, float)) else None


def _ath_from_cache(ticker, cache_dir):
    """
    All-time high and latest close straight from the local price cache.

    Needed because Chartink-sourced tickers arrive as symbol-only — the
    scanner never prices them, so they carry no pct_from_ath and are
    invisible to any ATH filter unless we compute it ourselves.
    Returns (pct_from_ath, price) or (None, None).
    """
    path = os.path.join(cache_dir, ticker + '_NS.pkl')
    if not os.path.exists(path):
        return None, None
    try:
        import pickle
        with open(path, 'rb') as f:
            df = pickle.load(f)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or len(df) < 20:
            return None, None
        ath   = float(df['High'].max())
        price = float(df['Close'].iloc[-1])
        if ath <= 0:
            return None, None
        return (price - ath) / ath * 100, price
    except Exception:
        return None, None


def build_derived_screens(results_json, cache_dir=None):
    """Return {key: {'label', 'stocks'}} for the browser-computed screens."""
    try:
        with open(results_json) as f:
            data = json.load(f)
    except Exception:
        return {}

    allst = _union_of_screens(data)
    rs    = lambda s: s.get('rs_rank') or 0

    # ath_highs — mirrors _buildAthData()
    ath = [s for s in allst
           if (_num(s.get('pct_from_ath')) is not None and s['pct_from_ath'] >= -5)
           or (_num(s.get('pct_from_ath')) is None
               and _num(s.get('pct_from_high')) is not None and s['pct_from_high'] >= -2)]
    ath.sort(key=rs, reverse=True)

    # rs_trending / rs_trending_minervini — mirrors _buildRsTrendingData()
    rst = [s for s in allst if s.get('rs_trending_up') is True]
    rst.sort(key=rs, reverse=True)
    rst_min = [s for s in rst if (s.get('passed') or 0) >= 7]

    # daily_near_pivot — mirrors _buildDailyPivotData()
    dp = [s for s in allst
          if _num(s.get('pct_from_pivot')) is not None
          and -5 <= s['pct_from_pivot'] <= 5
          and s.get('pivot_high') is not None]
    dp.sort(key=lambda s: abs(s.get('pct_from_pivot') or 0))

    # weekly_near_pivot — mirrors _buildWeeklyPivotData()
    wp = [s for s in allst
          if _num(s.get('pct_from_pivot_w')) is not None
          and -5 <= s['pct_from_pivot_w'] <= 5
          and s.get('pivot_high_w') is not None]
    wp.sort(key=lambda s: abs(s.get('pct_from_pivot_w') or 0))

    out = {
        'ath_highs':             {'label': 'All-Time High Leaders',   'stocks': ath},
        'rs_trending':           {'label': 'RS Line Trending Up',     'stocks': rst},
        'rs_trending_minervini': {'label': 'RS Trend + Minervini',    'stocks': rst_min},
        'daily_near_pivot':      {'label': 'Near Pivot (Daily ±5%)',  'stocks': dp},
        'weekly_near_pivot':     {'label': 'Near Pivot (Weekly ±5%)', 'stocks': wp},
    }

    # ── Within 10% of All-Time High — full coverage ──────────────────────────
    # Unlike ath_highs (which mirrors the dashboard and can only judge stocks
    # the scanner priced), this one resolves an ATH for EVERY ticker in the
    # scan — including the Chartink-sourced symbols that arrive with no
    # metrics — by falling back to the local price cache. Without that
    # fallback roughly a quarter of the universe is silently unjudgeable.
    if cache_dir:
        wide, resolved, unresolved = [], 0, 0
        for s in allst:
            v = _num(s.get('pct_from_ath'))
            e = dict(s)
            if v is None:
                v, price = _ath_from_cache(s.get('ticker', ''), cache_dir)
                if v is None:
                    unresolved += 1
                    continue
                e['pct_from_ath'] = round(v, 1)
                e['ath_source']   = 'cache'
                if price is not None and not e.get('price'):
                    e['price'] = round(price, 2)
                resolved += 1
            else:
                e['ath_source'] = 'scanner'
            if v >= -10:
                wide.append(e)
        # Closest to the high first — this screen is about proximity, and
        # many of these stocks have no RS rank to sort on.
        wide.sort(key=lambda s: -(s.get('pct_from_ath') or -999))
        out['ath_within_10'] = {'label': 'Within 10% of All-Time High', 'stocks': wide}
        print(f"  ATH 10% screen: {len(wide)} stocks "
              f"({resolved} ATH recovered from cache, {unresolved} still unpriced)")

    return {k: v for k, v in out.items() if v['stocks']}


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
                # Full mode: always regenerate
                out_path = generate_chart(t, s, cache_dir, out_dir,
                                          is_us=is_us, bars=bars, weekly=use_weekly)
                if out_path:
                    generated.append(entry)
                else:
                    # no cached data — keep entry so it still appears in the screen
                    png = os.path.join(out_dir, t + suffix + '.png')
                    if os.path.exists(png):
                        generated.append(entry)

        if new_count:
            print(f"    {new_count} charts regenerated", flush=True)
        result[key] = {'label': defn['label'], 'stocks': generated}
    return result


# ── ANCHORED VWAP FROM WEEKLY HIGHER LOWS ─────────────────────────────────────
# Weekly pivot low = unique lowest low of AVWAP_PIVOT_LR weeks either side —
# the same 10/10 standard find_pivot_lows() uses for the chart's pivot marks.
# (3/3 was tried first; it anchored on minor pullbacks, e.g. LLOYDSME Sep '26
# at 1741 instead of the Feb '26 base low at 1042.9 that traders anchor on.)
# HL1 = the latest pivot low, only if above the pivot before it; HL2 = the
# pivot before that, only if it too is a higher low. AVWAP is anchored at
# each, price = (O+H+L+C)/4.
AVWAP_PIVOT_LR = 10
AVWAP_NEAR_PCT = 3.0
_ETF_RE = re.compile(r'BEES|ETF|LIQID|LIQUID|ADD$|^(NEXT50|MIDSMALL|BSLNIFTY|HDFCNIFTY|MON100|MAFANG|ALPHA|'
                     r'MOM\d+|MOSMALL\d*|LOWVOL\d*|QUAL30|MIDCAP|SMALLCAP|BANKNIFTY|NIFTY\w*|SENSEX\w*|'
                     r'GOLD\w*|SILVER\w*|LIQUID\w*|METAL|PSUBANK|ITBEES|PHARMABEES|AUTOBEES)$')


AVWAP_LONG_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache_daily_long')
AVWAP_LONG_MAX_AGE = 7 * 86400   # older bars never change; refresh weekly for new listings/splits


def _long_daily_path(ticker, is_us):
    return os.path.join(AVWAP_LONG_DIR, ('US_' if is_us else 'NS_') + ticker.replace('.', '_') + '.pkl')


def ensure_long_daily(tickers, is_us=False, batch=100):
    """
    The scanner caches ~2 years of daily bars, which with 10/10 weekly pivots
    rarely holds two higher lows. Keep 5 years of daily bars per ticker so
    HL2 can be found and its AVWAP computed from daily bars. Batched yfinance
    download, only for missing/stale files.
    """
    import time
    import yfinance as yf
    os.makedirs(AVWAP_LONG_DIR, exist_ok=True)
    now   = time.time()
    stale = [t for t in tickers
             if not os.path.exists(_long_daily_path(t, is_us))
             or now - os.path.getmtime(_long_daily_path(t, is_us)) > AVWAP_LONG_MAX_AGE]
    if not stale:
        return
    print(f"  Long daily history: fetching {len(stale)} ticker(s)…", flush=True)
    for i in range(0, len(stale), batch):
        chunk = stale[i:i + batch]
        syms  = {t: (t if is_us else t + '.NS') for t in chunk}
        try:
            raw = yf.download(list(syms.values()), period='6y', interval='1d', auto_adjust=False,
                              group_by='ticker', progress=False, threads=False, timeout=30)
        except Exception as e:
            print(f"    batch {i // batch + 1} failed: {e}")
            continue
        for t, sym in syms.items():
            try:
                df = raw[sym][['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except Exception:
                df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            # Written even when empty, so delisted symbols wait out the max age
            # instead of being re-requested on every run.
            df.to_pickle(_long_daily_path(t, is_us))
        time.sleep(1)


def avwap_daily_series(df_daily, ticker, is_us=False):
    """
    The scanner's daily bars, extended backwards with the long daily cache.
    The splice is skipped if the two sources disagree by >3% on overlapping
    days (e.g. a split adjusted in one source but not the other).
    """
    p = _long_daily_path(ticker, is_us)
    if not os.path.exists(p):
        return df_daily
    try:
        ld = pd.read_pickle(p)
        if not len(ld):
            return df_daily
        ld.index = pd.to_datetime(ld.index)
        if (ld.index.tz is None) != (df_daily.index.tz is None):
            ld.index = (ld.index.tz_localize(df_daily.index.tz) if ld.index.tz is None
                        else ld.index.tz_localize(None))
        overlap = df_daily.index.intersection(ld.index)
        if len(overlap) < 5:
            return df_daily
        chk = overlap[:20]
        ratio = (df_daily.loc[chk, 'Close'] / ld.loc[chk, 'Close']).median()
        if abs(ratio - 1) > 0.03:
            return df_daily
        older = ld[ld.index < df_daily.index[0]]
        return pd.concat([older, df_daily]) if len(older) else df_daily
    except Exception:
        return df_daily


def chart_daily_series(df_daily, ticker, is_us=False):
    """
    Daily bars for the 3-year chart view: the scanner cache extended backwards
    with the 5-year cache. Unlike avwap_daily_series, a level difference
    between the sources (US cache is dividend-adjusted, the long cache is not)
    is reconciled by scaling the older bars to meet the scanner series.
    """
    p = _long_daily_path(ticker, is_us)
    if not os.path.exists(p):
        return df_daily
    try:
        ld = pd.read_pickle(p)
        if not len(ld):
            return df_daily
        ld.index = pd.to_datetime(ld.index)
        if (ld.index.tz is None) != (df_daily.index.tz is None):
            ld.index = (ld.index.tz_localize(df_daily.index.tz) if ld.index.tz is None
                        else ld.index.tz_localize(None))
        overlap = df_daily.index.intersection(ld.index)
        if len(overlap) < 5:
            return df_daily
        chk = overlap[:10]
        ratio = float((df_daily.loc[chk, 'Close'] / ld.loc[chk, 'Close']).median())
        if not np.isfinite(ratio) or ratio <= 0:
            return df_daily
        older = ld[ld.index < df_daily.index[0]].copy()
        if not len(older):
            return df_daily
        if abs(ratio - 1) > 0.005:
            for col in ('Open', 'High', 'Low', 'Close'):
                if col in older:
                    older[col] = older[col] * ratio
        cols = [c for c in df_daily.columns if c in older.columns]
        return pd.concat([older[cols], df_daily])
    except Exception:
        return df_daily


def refresh_recent_bars(daily_by_t, is_us=False, batch=100):
    """
    The scanner cache's latest bar can be an intraday snapshot (e.g. SUNPHARMA
    on 4 Nov: cached 1794.9 on 355k shares vs final 1773.4 on 4.9M), which
    skews both the AVWAP and the distance to it. Re-download the last 5 daily
    bars for the given tickers and overwrite those dates. Mutates daily_by_t.
    """
    import yfinance as yf
    tickers = list(daily_by_t)
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        syms  = {t: (t if is_us else t + '.NS') for t in chunk}
        try:
            raw = yf.download(list(syms.values()), period='5d', interval='1d', auto_adjust=False,
                              group_by='ticker', progress=False, threads=False, timeout=30)
        except Exception:
            continue
        for t, sym in syms.items():
            try:
                fr = raw[sym][['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except Exception:
                continue
            if not len(fr):
                continue
            dl = daily_by_t[t]
            fr.index = pd.to_datetime(fr.index)
            if (fr.index.tz is None) != (dl.index.tz is None):
                fr.index = (fr.index.tz_localize(dl.index.tz) if fr.index.tz is None
                            else fr.index.tz_localize(None))
            daily_by_t[t] = pd.concat([dl[~dl.index.isin(fr.index)], fr]).sort_index()


def avwap_line(df_daily, anchor):
    """Daily anchored VWAP from `anchor` (a bar in df_daily) to the end, price = (O+H+L+C)/4."""
    x  = df_daily[df_daily.index >= anchor]
    tp = (x['Open'] + x['High'] + x['Low'] + x['Close']) / 4
    v  = x['Volume'].astype(float)
    return (tp * v).cumsum() / v.cumsum().replace(0, np.nan)


def weekly_hl_avwap(df_daily):
    """
    Pivots are found on weekly bars; the AVWAP is computed on daily bars from
    the day that made each pivot week's low — what an anchored-VWAP tool gives
    when anchored on that day. (Weekly-bar AVWAP anchored at the whole week
    averages in the days before the low; SHRIRAMFIN read 911.92 that way vs
    ~921 from the low day.)

    Returns {'avwap': [{n, date, low, value, dist}, ...], 'avwap_min_dist',
    'avwap_near', 'avwap_rising', 'avwap_above_hl1'} or None if the stock has
    no intact weekly higher-low structure.
    """
    w = resample_weekly(df_daily)
    if len(w) < 30:
        return None
    lo  = w['Low'].values.astype(float)
    lr  = AVWAP_PIVOT_LR
    pls = [i for i in range(lr, len(lo) - lr)
           if lo[i] == lo[i - lr:i + lr + 1].min()
           and (lo[i - lr:i + lr + 1] == lo[i]).sum() == 1]
    # The structure must be intact now: the latest pivot itself has to be a
    # higher low. Older higher lows behind a newer lower low do not count
    # (e.g. CARERATING Dec '25 1446.9 → Mar '26 1392.7 is a lower low).
    if len(pls) < 2 or lo[pls[-1]] <= lo[pls[-2]]:
        return None
    hls = [pls[-1]]
    if len(pls) >= 3 and lo[pls[-2]] > lo[pls[-3]]:
        hls.insert(0, pls[-2])

    close = float(df_daily['Close'].iloc[-1])
    anchors = []
    for n, p in enumerate(reversed(hls), start=1):
        wk_end = w.index[p]
        days   = df_daily[(df_daily.index > wk_end - pd.Timedelta(days=7)) & (df_daily.index <= wk_end)]
        if not len(days):
            continue
        day  = days['Low'].idxmin()
        line = avwap_line(df_daily, day)
        val  = float(line.iloc[-1])
        if not np.isfinite(val):
            continue
        anchors.append({'n': n, 'date': str(day.date()), 'low': round(float(days['Low'].min()), 2),
                        'value': round(val, 2), 'dist': round((close / val - 1) * 100, 2)})
    if not anchors:
        return None
    near = [a for a in anchors if abs(a['dist']) <= AVWAP_NEAR_PCT]
    return {
        'avwap':           anchors,
        'avwap_min_dist':  min(abs(a['dist']) for a in anchors),
        'avwap_near':      '+'.join(f"HL{a['n']}" for a in near),
        'avwap_rising':    len(anchors) == 2 and anchors[0]['low'] > anchors[1]['low'],
        'avwap_above_hl1': close > anchors[0]['low'],
    }


def build_avwap_screens(results_json, cache_dir, out_dir, is_us=False):
    """
    Two screens, both drawn as weekly charts with the AVWAP lines (_AV.png):
      avwap_hl_near   — close within ±AVWAP_NEAR_PCT of either AVWAP
      avwap_hl_stage2 — same, plus price > MA50 > MA200, rising higher lows,
                        and price still above the latest higher low
    Charts are always redrawn: the AVWAP moves every bar, so the price-move
    manifest used by --daily would leave stale lines on unchanged stocks.
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
            if t and t not in seen and (is_us or not _ETF_RE.search(t)):
                seen[t] = s

    ensure_long_daily(list(seen), is_us)

    # Pass 1 on cached bars with a wide band, so a stock whose stale last bar
    # put it just outside ±AVWAP_NEAR_PCT is not lost before the refresh.
    daily_by_t = {}
    for t in seen:
        df = load_df(t, cache_dir, is_us)
        if df is None:
            continue
        try:
            dl   = avwap_daily_series(df, t, is_us)
            info = weekly_hl_avwap(dl)
        except Exception:
            info = None
        if info and info['avwap_min_dist'] <= AVWAP_NEAR_PCT * 2.5:
            daily_by_t[t] = dl

    # Pass 2 on refreshed recent bars, with the real band.
    refresh_recent_bars(daily_by_t, is_us)
    near = []
    for t, dl in daily_by_t.items():
        try:
            info = weekly_hl_avwap(dl)
        except Exception:
            info = None
        if not info or not info['avwap_near']:
            continue
        c = dl['Close'].values.astype(float)
        e = dict(seen[t])
        e.update(info)
        e['price'] = round(float(c[-1]), 2)
        e['avwap_stage2'] = bool(len(c) >= 200 and c[-1] > c[-50:].mean() > c[-200:].mean())
        e['img'] = t + '_AV.png'
        near.append(e)

    near.sort(key=lambda e: (-len(e['avwap_near']), e['avwap_min_dist']))
    drawn = []
    for e in near:
        if generate_chart(e['ticker'], e, cache_dir, out_dir, is_us=is_us, avwap=True,
                          avwap_daily=daily_by_t[e['ticker']]):
            drawn.append(e)
    stage2 = [e for e in drawn
              if e['avwap_stage2'] and e['avwap_rising'] and e['avwap_above_hl1']]
    print(f"  → avwap_hl_near: {len(drawn)} stocks · avwap_hl_stage2: {len(stage2)} stocks [weekly AVWAP]")
    return {
        'avwap_hl_stage2': {'label': f'AVWAP Weekly HL — Stage 2 (±{AVWAP_NEAR_PCT:g}%)', 'stocks': stage2},
        'avwap_hl_near':   {'label': f'AVWAP Weekly HL — All Near (±{AVWAP_NEAR_PCT:g}%)', 'stocks': drawn},
    }


def build_near_weekly_pivot(results_json, cache_dir, out_dir, bars,
                             is_us=False, manifest=None, existing_data=None,
                             pct_lo=-8.0, pct_hi=5.0):
    """
    Build the Near Weekly Pivot screen from ALL stocks in results.json
    where pct_from_pivot_w is within [pct_lo, pct_hi].
    Generates (or reuses) _W.png for each qualifying stock.
    Returns {'label':..., 'stocks':[...]} ready for india_data / us_data.
    """
    try:
        with open(results_json) as f:
            raw = json.load(f)
    except Exception as e:
        print(f"  ERROR loading {results_json}: {e}")
        return {'label': 'Near Weekly Pivot', 'stocks': []}

    # Deduplicate: one entry per ticker (keep first occurrence)
    seen = {}
    for scr in raw.get('screens', {}).values():
        for s in scr.get('stocks', []):
            t = s.get('ticker', '')
            if t and t not in seen:
                seen[t] = s

    # Filter by pct_from_pivot_w
    qualifying = []
    for t, s in seen.items():
        pct = s.get('pct_from_pivot_w')
        if pct is not None and pct_lo <= pct <= pct_hi:
            qualifying.append(s)

    # Sort by distance from pivot (closest first)
    qualifying.sort(key=lambda s: abs(s.get('pct_from_pivot_w') or 99))
    print(f"  Near Weekly Pivot: {len(qualifying)} stocks within {pct_lo}% to +{pct_hi}% of weekly pivot")

    # Restore previous AI results
    old_ai = {}
    if existing_data and 'near_weekly_pivot' in existing_data:
        old_ai = {e['ticker']: e.get('ai')
                  for e in existing_data['near_weekly_pivot'].get('stocks', []) if e.get('ai')}

    os.makedirs(out_dir, exist_ok=True)
    generated = []
    new_count  = 0

    for s in qualifying:
        t     = s.get('ticker', '')
        entry = dict(s)
        entry['img'] = t + '_W.png'
        if t in old_ai:
            entry['ai'] = old_ai[t]

        png = os.path.join(out_dir, t + '_W.png')

        if manifest is not None:
            # Daily mode
            if is_stale(t, s, manifest, out_dir, '_W'):
                out_path = generate_chart(t, s, cache_dir, out_dir,
                                          is_us=is_us, bars=bars, weekly=True)
                if out_path:
                    update_manifest(manifest, t, s, '_W')
                    generated.append(entry)
                    new_count += 1
                elif os.path.exists(png):
                    generated.append(entry)
            else:
                generated.append(entry)
        else:
            # Full mode: always regenerate
            out_path = generate_chart(t, s, cache_dir, out_dir,
                                      is_us=is_us, bars=bars, weekly=True)
            if out_path:
                generated.append(entry)
                new_count += 1
            elif os.path.exists(png):
                generated.append(entry)

    if new_count:
        print(f"    {new_count} weekly charts generated", flush=True)
    return {'label': 'Near Weekly Pivot', 'stocks': generated}


# ── CHART GENERATOR ───────────────────────────────────────────────────────────

def _load_live_prices():
    """Return (india_prices_dict, us_prices_dict) from live price JSON files."""
    india, us = {}, {}
    try:
        india = json.load(open(os.path.join(SCRIPT_DIR, 'live_prices.json'))).get('prices', {})
    except Exception:
        pass
    try:
        us = json.load(open(os.path.join(SCRIPT_DIR, 'live_prices_us.json'))).get('prices', {})
    except Exception:
        pass
    return india, us

# Loaded once at module level for all load_df calls in this run
_LIVE_INDIA, _LIVE_US = _load_live_prices()


def load_df(ticker, cache_dir, is_us=False):
    """Load cached OHLCV DataFrame for a ticker.

    Patches today's bar from live_prices.json when the yfinance daily close
    is NaN (Yahoo Finance lags ~1 trading day before publishing daily close).
    """
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

        # ── Patch today's bar if yfinance daily close is lagging ────────────
        live_map   = _LIVE_US if is_us else _LIVE_INDIA
        live_entry = live_map.get(ticker)
        if live_entry:
            from datetime import timezone as _tz, timedelta as _td
            tz_info = _tz(_td(hours=-4)) if is_us else _tz(_td(hours=5, minutes=30))
            today   = datetime.datetime.now(tz_info).date()
            last_date = df.index[-1]
            last_date_d = last_date.date() if hasattr(last_date, 'date') else last_date
            if last_date_d < today:
                live_p  = float(live_entry['price'])
                prev_c  = float(df['Close'].iloc[-1])
                # Build a synthetic today bar using live price
                idx_tz  = df.index.tz
                today_ts = pd.Timestamp(today).tz_localize(idx_tz) if idx_tz else pd.Timestamp(today)
                today_row = pd.DataFrame({
                    'Open':   [prev_c],
                    'High':   [max(prev_c, live_p)],
                    'Low':    [min(prev_c, live_p)],
                    'Close':  [live_p],
                    'Volume': [int(live_entry.get('volume', 0))],
                }, index=[today_ts])
                df = pd.concat([df, today_row])

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


def generate_chart(ticker, entry, cache_dir, out_dir, is_us=False, bars=250, weekly=False,
                   avwap=False, avwap_daily=None):
    """
    Generate annotated candlestick chart PNG.
    weekly=True → resample to weekly bars, use _W.png suffix.
    avwap=True  → weekly chart with anchored VWAPs from the anchors stored in
                  entry['avwap'] (see weekly_hl_avwap), saved as _AV.png so it
                  does not overwrite the plain weekly chart other screens use.
                  avwap_daily is the (longer) daily series the AVWAP is
                  computed on; it also supplies the weekly bars drawn.
    Weekly charts show WEEKLY_BARS (5 years) for the long-term trend view;
    daily charts show `bars` (default 1 year).
    Returns output path or None if failed.
    """
    df_daily = load_df(ticker, cache_dir, is_us)
    if df_daily is None:
        return None

    if avwap:
        weekly = True
        if avwap_daily is not None:
            df_daily = avwap_daily
    if weekly and not avwap:
        # scanner cache holds ~2 years; splice in the long cache for the 5-year view
        df_daily = chart_daily_series(df_daily, ticker, is_us)
    if weekly:
        df = resample_weekly(df_daily)
        suffix = '_AV' if avwap else '_W'
        bars_w = WEEKLY_BARS
        df = df.tail(bars_w).copy()
    else:
        df = df_daily
        suffix = ''
        df = df.tail(bars).copy()
    return _render_clean(ticker, entry, df, df_daily, out_dir, suffix,
                         is_us=is_us, weekly=weekly, avwap=avwap)


# ── CLEAN CHART RENDERER ──────────────────────────────────────────────────────
# TradingView-style: candles, MAs, 10/10 pivot labels, the screen's pivot line,
# AVWAP (AVWAP screens only) and volume. No patterns, boxes or trend lines.
CC_UP      = '#089981'
CC_DN      = '#F23645'
CC_GRID    = '#F0F3FA'
CC_AXIS    = '#B2B5BE'
CC_TEXT    = '#131722'
CC_MUTED   = '#787B86'
CC_MA_S    = '#2962FF'   # daily EMA21
CC_MA_M    = '#FF9800'   # SMA50 / 10W
CC_MA_L    = '#7E57C2'   # SMA200 / 40W
CC_AVWAP   = ('#1565C0', '#C2185B')
CC_PIVOT   = '#5D606B'


def _nice_log_ticks(lo, hi, n=9):
    """Round-number ticks (1, 1.5, 2, 3, 4, 5, 6, 8 × 10^k) roughly evenly spaced on a log axis."""
    if lo <= 0 or hi <= lo:
        return None
    cands = sorted(m * 10 ** k for k in range(int(np.floor(np.log10(lo))) - 1,
                                              int(np.ceil(np.log10(hi))) + 1)
                   for m in (1, 1.5, 2, 3, 4, 5, 6, 8))
    cands = [c for c in cands if lo <= c <= hi]
    if len(cands) <= n:
        return cands
    min_ratio = (hi / lo) ** (1.0 / (n + 1))
    out = [cands[0]]
    for c in cands[1:]:
        if c / out[-1] >= min_ratio:
            out.append(c)
    return out


def _fmt_px(v):
    return f'{v:,.0f}' if v >= 10000 else f'{v:,.2f}'


def _render_clean(ticker, entry, df, df_daily, out_dir, suffix,
                  is_us=False, weekly=False, avwap=False):
    n   = len(df)
    if n < 5:
        return None
    xs  = np.arange(n)
    op  = df['Open'].values.astype(float)
    hi  = df['High'].values.astype(float)
    lo  = df['Low'].values.astype(float)
    cl  = df['Close'].values.astype(float)
    vol = np.nan_to_num(df['Volume'].values.astype(float))

    # MAs on the full history, then sliced to the window so the left edge is right
    full_src = resample_weekly(df_daily) if weekly else df_daily
    full_c   = pd.Series(full_src['Close'].values.astype(float))
    if weekly:
        mas = [('SMA 10W', full_c.rolling(10).mean().values, CC_MA_M, 1.5),
               ('SMA 40W', full_c.rolling(40).mean().values, CC_MA_L, 1.8)]
    else:
        mas = [('EMA 21',  full_c.ewm(span=21, adjust=False).mean().values, CC_MA_S, 1.1),
               ('SMA 50',  full_c.rolling(50).mean().values,  CC_MA_M, 1.5),
               ('SMA 200', full_c.rolling(200).mean().values, CC_MA_L, 1.8)]
    mas = [(lbl, v[-n:], col, lw) for lbl, v, col, lw in mas]
    offset_full = len(full_src) - n

    fig = plt.figure(figsize=(CHART_W, CHART_H), facecolor='#FFFFFF')
    gs  = fig.add_gridspec(2, 1, height_ratios=[5, 1], left=0.008, right=0.93,
                           top=0.885, bottom=0.06, hspace=0.0)
    ax  = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)

    # price scale: log when the window spans a large move (keeps 5Y trends readable)
    y_lo, y_hi = float(np.nanmin(lo)), float(np.nanmax(hi))
    use_log = weekly and y_lo > 0 and y_hi / y_lo > 2.5
    if use_log:
        ax.set_yscale('log')
        pad = (y_hi / y_lo) ** 0.07
        ax.set_ylim(y_lo / pad, y_hi * pad)
    else:
        pad = (y_hi - y_lo) * 0.08
        ax.set_ylim(y_lo - pad, y_hi + pad)

    # ── candles ──
    up = cl >= op
    wick_w = 0.9 if n > 180 else 1.0
    body_w = 0.62 if n > 180 else 0.68
    for mask, col in ((up, CC_UP), (~up, CC_DN)):
        ax.vlines(xs[mask], lo[mask], hi[mask], color=col, linewidth=wick_w, zorder=3)
        b0 = np.minimum(op, cl)[mask]
        bh = np.abs(cl - op)[mask]
        bh = np.maximum(bh, (hi - lo)[mask] * 0.004 + 1e-9)
        ax.bar(xs[mask], bh, bottom=b0, width=body_w, color=col, edgecolor=col,
               linewidth=0.3, zorder=4)

    # ── moving averages ──
    for lbl, v, col, lw in mas:
        ok = np.isfinite(v)
        if ok.any():
            ax.plot(xs[ok], v[ok], color=col, linewidth=lw, zorder=5, alpha=0.95)

    # ── anchored VWAPs (AVWAP screens only) ──
    av_legend = []
    if avwap:
        for a in (entry.get('avwap') or []):
            day = pd.Timestamp(a['date'])
            if df_daily.index.tz is not None:
                day = day.tz_localize(df_daily.index.tz)
            ai = full_src.index.searchsorted(day)
            if ai >= len(full_src):
                continue
            line_d = avwap_line(df_daily, day)
            line   = line_d.groupby(full_src.index[full_src.index.searchsorted(line_d.index)]).last()
            line   = line.reindex(full_src.index[ai:]).values.astype(float)
            start  = ai - offset_full
            seg_x  = np.arange(start, start + len(line))
            keep   = (seg_x >= 0) & np.isfinite(line)
            col    = CC_AVWAP[a['n'] - 1]
            ax.plot(seg_x[keep], line[keep], color=col, linewidth=1.9, zorder=6)
            if 0 <= start < n:
                ax.annotate(f"AV{a['n']}", xy=(start, lo[start]), xytext=(0, -20),
                            textcoords='offset points', ha='center', va='top',
                            fontsize=8, fontweight='bold', color=col, zorder=8,
                            arrowprops=dict(arrowstyle='-', color=col, lw=0.8))
            av_legend.append((f"AVWAP{a['n']} {_fmt_px(a['value'])} ({a['dist']:+.1f}%)", col))

    # ── 10/10 pivot labels (price only, like TradingView Pivot Points High Low) ──
    full_hi = full_src['High'].values.astype(float)
    full_lo = full_src['Low'].values.astype(float)
    ph = find_pivot_highs(full_hi, left=10, right=10, lookback=n + 20)
    pl = find_pivot_lows(full_lo,  left=10, right=10, lookback=n + 20)
    for i, v in ph:
        x = i - offset_full
        if 2 <= x < n:
            ax.annotate(_fmt_px(v), xy=(x, v), xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8.5, color=CC_TEXT, zorder=7)
    for i, v in pl:
        x = i - offset_full
        if 2 <= x < n:
            ax.annotate(_fmt_px(v), xy=(x, v), xytext=(0, -5), textcoords='offset points',
                        ha='center', va='top', fontsize=8.5, color=CC_TEXT, zorder=7)

    # ── the screen's pivot (buy point): dashed from the pivot bar to the right edge ──
    piv     = entry.get('pivot_high_w' if weekly else 'pivot_high')
    piv_ago = entry.get('pivot_bars_ago_w' if weekly else 'pivot_bars_ago')
    tags = []
    if piv and piv > 0:
        px0 = max(0, n - 1 - int(piv_ago)) if piv_ago is not None else 0
        ax.hlines(piv, px0, n + 3, color=CC_PIVOT, linewidth=1.0,
                  linestyle=(0, (5, 4)), zorder=5)
        tags.append((piv, f'Pivot {_fmt_px(piv)}', CC_PIVOT))

    # ── last price line + tag ──
    last = float(cl[-1])
    lcol = CC_UP if n < 2 or cl[-1] >= cl[-2] else CC_DN
    ax.hlines(last, 0, n + 3, color=lcol, linewidth=0.8, linestyle=(0, (1, 3)), zorder=5)
    tags.append((last, _fmt_px(last), lcol))
    for yv, txt, col in tags:
        ax.annotate(txt, xy=(1.0, yv), xycoords=('axes fraction', 'data'),
                    xytext=(4, 0), textcoords='offset points', ha='left', va='center',
                    fontsize=8.5, fontweight='bold', color='#FFFFFF', zorder=9,
                    annotation_clip=False,
                    bbox=dict(boxstyle='square,pad=0.28', fc=col, ec=col))

    # ── volume ──
    vcol = np.where(up, CC_UP, CC_DN)
    axv.bar(xs, vol, width=body_w, color=vcol, alpha=0.45, linewidth=0, zorder=2)
    vma = pd.Series(vol).rolling(20 if not weekly else 10).mean().values
    axv.plot(xs, vma, color=CC_MUTED, linewidth=1.0, zorder=3)
    # cap the scale so one huge spike doesn't flatten every other bar
    pos = vol[vol > 0]
    vmax = float(np.nanmax(vol)) if len(pos) else 1.0
    if len(pos) > 20:
        vmax = min(vmax, float(np.percentile(pos, 95)) * 2.5)
    axv.set_ylim(0, vmax * 1.1)
    axv.set_yticks([])

    # ── axes ──
    ax.set_xlim(-1, n + 3)
    for a_ in (ax, axv):
        a_.set_facecolor('#FFFFFF')
        a_.grid(True, color=CC_GRID, linewidth=0.8, zorder=0)
        a_.set_axisbelow(True)
        for sp in ('top', 'left'):
            a_.spines[sp].set_visible(False)
        for sp in ('right', 'bottom'):
            a_.spines[sp].set_color(CC_AXIS)
            a_.spines[sp].set_linewidth(0.6)
        a_.tick_params(left=False, labelleft=False, right=True, labelright=True,
                       colors=CC_MUTED, labelsize=8.5, length=3, width=0.5)
    ax.tick_params(bottom=False, labelbottom=False)
    if use_log:
        tk = _nice_log_ticks(*ax.get_ylim())
        if tk:
            ax.yaxis.set_major_locator(mticker.FixedLocator(tk))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(7))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt_px(v)))

    # date ticks: weekly → Jan / Jul of each year; daily → each month (year on Jan)
    idx = df.index
    ticks, labels, prev = [], [], None
    for i, d in enumerate(idx):
        key = (d.year, d.month)
        if key == prev:
            continue
        prev = key
        if weekly and d.month in (1, 7):
            ticks.append(i); labels.append(str(d.year) if d.month == 1 else 'Jul')
        elif not weekly:
            ticks.append(i); labels.append(str(d.year) if d.month == 1 else d.strftime('%b'))
    ticks_ok = [(t, l) for t, l in zip(ticks, labels) if t > 2]
    if ticks_ok:
        axv.set_xticks([t for t, _ in ticks_ok])
        axv.set_xticklabels([l for _, l in ticks_ok], color=CC_MUTED, fontsize=8.5)
    axv.tick_params(axis='x', length=3, colors=CC_MUTED)

    # ── header (TradingView-style legend) ──
    exch = 'US' if is_us else 'NSE'
    tf   = '1W' if weekly else '1D'
    span = f"{round(n / 52)}Y" if weekly else f"{round(n / 21)}M" if n < 240 else '1Y'
    fig.text(0.010, 0.975, ticker, fontsize=15, fontweight='bold', color=CC_TEXT, va='top')
    fig.text(0.010 + 0.0118 * len(ticker) + 0.008, 0.968, f'·  {tf}  ·  {exch}  ·  {span}',
             fontsize=10, color=CC_MUTED, va='top')
    chg = (cl[-1] / cl[-2] - 1) * 100 if n > 1 and cl[-2] else 0.0
    ohlc = (f"O {_fmt_px(op[-1])}   H {_fmt_px(hi[-1])}   L {_fmt_px(lo[-1])}   "
            f"C {_fmt_px(cl[-1])}   {cl[-1] - cl[-2]:+,.2f} ({chg:+.2f}%)") if n > 1 else ''
    fig.text(0.40, 0.968, ohlc, fontsize=9.5, color=lcol, va='top', fontfamily='monospace')
    rs = entry.get('rs_rank')
    meta = '   '.join(x for x in [f'RS {rs}' if rs is not None else '',
                                   f"{entry['passed']}/8" if entry.get('passed') is not None else '']
                      if x)
    if meta:
        fig.text(0.93, 0.968, meta, fontsize=9.5, color=CC_MUTED, va='top', ha='right')
    lx = 0.010
    for lbl, v, col, lw in mas:
        cur_v = v[-1] if np.isfinite(v[-1]) else None
        txt = f"{lbl} {_fmt_px(cur_v)}" if cur_v else lbl
        fig.text(lx, 0.928, txt, fontsize=9, color=col, va='top', fontweight='bold')
        lx += 0.0068 * len(txt) + 0.018
    for txt, col in av_legend:
        fig.text(lx, 0.928, txt, fontsize=9, color=col, va='top', fontweight='bold')
        lx += 0.0068 * len(txt) + 0.018

    out_path = os.path.join(out_dir, f'{ticker}{suffix}.png')
    plt.savefig(out_path, dpi=DPI, facecolor='#FFFFFF', edgecolor='none')
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

def is_stale(ticker, entry, manifest, out_dir, suffix='', price_threshold=0.005):
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
    if m.get('date') != str(datetime.date.today()):
        return True                      # new day: always redraw
    # Drawn earlier today. Without this check a chart drawn mid-session stayed
    # frozen until tomorrow, so the closing candle never appeared.
    last_price = float(m.get('price', 0) or 0)
    cur_price  = float(entry.get('price', entry.get('close', 0)) or 0)
    return last_price <= 0 or abs(cur_price - last_price) / last_price >= price_threshold

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
            src      = src.tail(min(bars, 90)).copy()
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
                img_path=img_path,
                entry=s,               # ← stock entry for RAG context
            )
            if result:
                result['date'] = today
                s['ai'] = result
                vis_tag = '👁' if result.get('vision') else '📊'
                rag_tag = '📚' if result.get('rag') else ''
                print(f"    {vis_tag}{rag_tag} {ticker}: {result.get('pattern')} "
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
        # Keys prefixed with "_" are metadata, not screens (e.g. _sidebar,
        # which holds the sidebar layout as a list). Skip them.
        if key.startswith('_') or not isinstance(scr, dict):
            continue
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
    _rows = [(k, v) for k, v in screens_data.items()
             if not k.startswith('_') and isinstance(v, dict)]
    for ri, (key, scr) in enumerate(_rows, 4):
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
    no_chart  = 0
    for s in stocks:
        ticker = s.get('ticker', '')
        entry  = dict(s)
        entry['img'] = ticker + suffix + '.png'
        if is_stale(ticker, s, manifest, out_dir, suffix):
            out = generate_chart(ticker, s, cache_dir, out_dir,
                                 is_us=is_us, bars=bars, weekly=weekly)
            if out:
                update_manifest(manifest, ticker, s, suffix)
                new_count += 1
                print(f"  ↺ {ticker}{suffix}", end='  ', flush=True)
            else:
                # No cached price data — typically a Chartink-only ticker that
                # is outside the local scan universe. The stock still belongs
                # to this screen, so it is kept and the card falls back to the
                # "No chart data" placeholder. Dropping it here is what used
                # to make the viewer's counts disagree with the scanner's.
                no_chart += 1
        generated.append(entry)
    if new_count or no_chart:
        note = f" ({new_count} regenerated" + (f", {no_chart} without chart data" if no_chart else "") + ")"
        print(f"\n  → {label}: {len(generated)} stocks{note}")
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
    parser.add_argument('--bars',        type=int, default=250,
                        help='Candles to display per chart')
    parser.add_argument('--daily',       action='store_true',
                        help='Incremental mode: only regenerate changed charts (fast, for daily use)')
    parser.add_argument('--new-only',    action='store_true',
                        help='Skip base screens, only run extra pattern screens')
    parser.add_argument('--ai-analysis', action='store_true',
                        help='Run Ollama AI analysis (llama3.2, free, offline)')
    parser.add_argument('--ai-top',      type=int, default=25,
                        help='AI analysis: top N stocks per priority screen (default 25)')
    parser.add_argument('--build-rag',   metavar='PDF',
                        help='Build Minervini RAG index from PDF (one-time setup)')
    args = parser.parse_args()

    # ── RAG index build (one-time) ────────────────────────────────────────────
    if args.build_rag:
        if _mrag is None:
            print("ERROR: minervini_rag.py not found next to this script.")
            sys.exit(1)
        _mrag.build_index(args.build_rag)
        global _RAG_AVAILABLE
        _RAG_AVAILABLE = True
        print("\n✓ RAG index built. AI analysis will now use Minervini book context.")
        if not args.ai_analysis:
            sys.exit(0)

    # ── AI setup ─────────────────────────────────────────────────────────────
    if args.ai_analysis:
        if init_ai_client():
            rag_status = '+ 📚 Minervini RAG' if _RAG_AVAILABLE else '(RAG index not built — run --build-rag)'
            print(f"✓ Ollama AI ready ({_OLLAMA_MODEL}) — top {args.ai_top} per priority screen  {rag_status}")
        else:
            print("⚠  Ollama not reachable. Start with: ollama serve")

    # Mirror the scanner exactly: every screen it produced, in its own order.
    # Only the two Higher Low Pivot screens are computed locally.
    INDIA_SCREENS = mirrored_screen_keys(INDIA_JSON)
    US_SCREENS    = mirrored_screen_keys(US_JSON)

    # A ticker appears in many screens (RELIANCE might be in 20 of the 47), but
    # its PNG is identical whichever screen asked for it. Render each
    # ticker+timeframe once per run and reuse it — without this, mirroring all
    # screens means ~10,000 redundant renders for India alone.
    # Screens that carry their own meaningful order and must NOT be re-sorted
    # by RS below. ath_within_10 is ranked by distance from the high, and many
    # of its members are Chartink-sourced with no RS rank at all — sorting
    # those by RS drops every one of them to the bottom, including stocks
    # sitting exactly at their all-time high.
    KEEP_ORDER = {'ath_within_10'}

    rendered = set()

    def render_once(ticker, s, cache_dir, out_dir, is_us, weekly):
        """Draw a chart unless this ticker+timeframe was already drawn."""
        kk = (ticker, weekly, is_us)
        if kk in rendered:
            return True
        ok = generate_chart(ticker, s, cache_dir, out_dir,
                            is_us=is_us, bars=args.bars, weekly=weekly)
        if ok:
            rendered.add(kk)
        return ok

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
        india_screens.update(build_derived_screens(INDIA_JSON, INDIA_CACHE))
        for key, scr in india_screens.items():
            weekly = key in WEEKLY_MIRROR_SCREENS
            stocks = (scr['stocks'] if key in KEEP_ORDER
                      else sorted(scr['stocks'], key=lambda s: -(s.get('rs_rank') or 0)))[:args.max]
            if args.daily:
                generated = _regen_screen(stocks, INDIA_CACHE, OUT_INDIA, False,
                                          args.bars, manifest, weekly=weekly, label=scr['label'])
            else:
                suffix = '_W' if weekly else ''
                generated = []
                drawn = 0
                for s in stocks:
                    ticker = s.get('ticker', '')
                    entry  = dict(s)
                    entry['img'] = ticker + suffix + '.png'
                    if render_once(ticker, s, INDIA_CACHE, OUT_INDIA, False, weekly):
                        drawn += 1
                    # Kept whether or not a chart rendered — the screen must
                    # match the scanner stock-for-stock.
                    generated.append(entry)
                print(f"  → {key}: {len(generated)} stocks ({drawn} with charts)", flush=True)
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
        us_screens.update(build_derived_screens(US_JSON, US_CACHE))
        for key, scr in us_screens.items():
            weekly = key in WEEKLY_MIRROR_SCREENS
            stocks = (scr['stocks'] if key in KEEP_ORDER
                      else sorted(scr['stocks'], key=lambda s: -(s.get('rs_rank') or 0)))[:args.max]
            if args.daily:
                generated = _regen_screen(stocks, US_CACHE, OUT_US, True,
                                          args.bars, manifest, weekly=weekly, label=scr['label'])
            else:
                suffix = '_W' if weekly else ''
                generated = []
                drawn = 0
                for s in stocks:
                    ticker = s.get('ticker', '')
                    entry  = dict(s)
                    entry['img'] = ticker + suffix + '.png'
                    if render_once(ticker, s, US_CACHE, OUT_US, True, weekly):
                        drawn += 1
                    generated.append(entry)
                print(f"  → {key}: {len(generated)} stocks ({drawn} with charts)", flush=True)
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

    # ── Anchored VWAP from weekly higher lows ────────────────────────────────
    if args.market in ('india', 'both'):
        print("\n=== INDIA — Anchored VWAP (weekly HL) ===")
        india_data.update(build_avwap_screens(INDIA_JSON, INDIA_CACHE, OUT_INDIA, is_us=False))
    if args.market in ('us', 'both'):
        print("\n=== US — Anchored VWAP (weekly HL) ===")
        us_data.update(build_avwap_screens(US_JSON, US_CACHE, OUT_US, is_us=True))

    # ── Trend charts: every ticker gets a 5Y weekly (_W) and 1Y daily chart ──
    # so the viewer's "5Y Weekly / 1Y Daily" toggle works on every screen.
    def _ensure_both(data, cache_dir, out_dir, is_us):
        seen = {}
        for scr in data.values():
            for e in (scr.get('stocks') or []) if isinstance(scr, dict) else []:
                t = e.get('ticker')
                if t and t not in seen:
                    seen[t] = e
        made = 0
        for t, e in seen.items():
            for weekly, suffix in ((True, '_W'), (False, '')):
                if (t, weekly, is_us) in rendered:
                    continue
                if args.daily and not is_stale(t, e, manifest, out_dir, suffix):
                    continue
                if generate_chart(t, e, cache_dir, out_dir, is_us=is_us,
                                  bars=args.bars, weekly=weekly):
                    rendered.add((t, weekly, is_us))
                    if args.daily:
                        update_manifest(manifest, t, e, suffix)
                    made += 1
        print(f"  {len(seen)} tickers · {made} trend charts drawn", flush=True)

    if args.market in ('india', 'both'):
        print("\n=== INDIA — 5Y weekly / 1Y daily trend charts ===")
        _ensure_both(india_data, INDIA_CACHE, OUT_INDIA, False)
    if args.market in ('us', 'both'):
        print("\n=== US — 5Y weekly / 1Y daily trend charts ===")
        _ensure_both(us_data, US_CACHE, OUT_US, True)

    # ── Sector / industry / theme rotation (Sectors tab) ──────────────────────
    try:
        import sector_rotation
        sector_rotation.build([m for m in ('india', 'us') if args.market in (m, 'both')])
    except Exception as e:
        print(f"  sector rotation failed: {e}")

    # ── Near Weekly Pivot — retired ──────────────────────────────────────────
    # This was a viewer-only screen with its own pivot window, which meant it
    # disagreed with the dashboard's own weekly-pivot screen (187 vs 109 stocks
    # for India). The mirrored 'weekly_near_pivot' — a direct port of
    # dashboard.html's _buildWeeklyPivotData() — is now the single definition.
    # build_near_weekly_pivot() is left in place should it ever be wanted back.

    # ── AI batch analysis on priority screens ─────────────────────────────────
    if args.ai_analysis and _AI_ENABLED and args.ai_top > 0:
        print(f"\n=== AI ANALYSIS (top {args.ai_top} per priority screen) ===")
        today_str = str(datetime.date.today())
        # Global dedup: share ai result across screens so each ticker is only analysed once
        _ai_cache = {}  # ticker -> ai dict

        def _run_ai_deduped(scr, cache_dir, out_dir, is_us, bars, ai_top, weekly):
            stocks = scr.get('stocks', [])
            candidates = [s for s in stocks
                          if not (isinstance(s.get('ai'), dict) and s['ai'].get('date') == today_str)]
            candidates = candidates[:ai_top]
            if not candidates:
                print(f"    (all {len(stocks)} already analysed today)", flush=True)
                return
            # Reuse already-computed results from earlier screens
            fresh = []
            for s in candidates:
                t = s.get('ticker', '')
                if t in _ai_cache:
                    s['ai'] = _ai_cache[t]
                else:
                    fresh.append(s)
            if fresh:
                run_ai_on_screen({'stocks': fresh}, cache_dir, out_dir, is_us, bars, len(fresh), weekly)
                for s in fresh:
                    if s.get('ai'):
                        _ai_cache[s['ticker']] = s['ai']

        for key, scr in india_data.items():
            if key in AI_PRIORITY_SCREENS:
                weekly = EXTRA_SCREENS_DEF.get(key, {}).get('weekly', False)
                print(f"  {key} ({scr['label']})…")
                _run_ai_deduped(scr, INDIA_CACHE, OUT_INDIA, False, args.bars, args.ai_top, weekly)
        for key, scr in us_data.items():
            if key in AI_PRIORITY_SCREENS:
                weekly = EXTRA_SCREENS_DEF.get(key, {}).get('weekly', False)
                print(f"  {key} US ({scr['label']})…")
                _run_ai_deduped(scr, US_CACHE, OUT_US, True, args.bars, args.ai_top, weekly)

    # ── Save manifest ─────────────────────────────────────────────────────────
    if args.daily:
        save_manifest(manifest)

    # ── Build AI Picks screens (BUY-rated stocks, sorted by confidence) ──────
    def _build_ai_picks(data):
        picks = []
        seen  = set()
        for key, scr in data.items():
            if key.startswith('_') or not isinstance(scr, dict):
                continue
            for s in scr.get('stocks', []):
                ai = s.get('ai') or {}
                if ai.get('action') == 'BUY' and s.get('ticker') not in seen:
                    picks.append(s)
                    seen.add(s.get('ticker'))
        picks.sort(key=lambda s: (s.get('ai') or {}).get('conf') or 0, reverse=True)
        return {'label': 'AI Minervini Picks', 'stocks': picks}

    # ── Patch prices from live_prices files (scanner captures mid-session prices) ─
    def _patch_data_prices(data, live_prices_dict):
        """Replace scanner-embedded prices with end-of-session prices from live_prices.json."""
        for key, scr in data.items():
            # "_"-prefixed keys hold metadata (e.g. _sidebar), not screens.
            if key.startswith('_') or not isinstance(scr, dict):
                continue
            for s in scr.get('stocks', []):
                t = s.get('ticker', '')
                lp = live_prices_dict.get(t)
                if lp:
                    s['price'] = lp['price']

    _patch_data_prices(india_data, _LIVE_INDIA)
    _patch_data_prices(us_data,    _LIVE_US)

    # ── Write JSON + Excel — only for markets that were actually processed ──────
    # 'ai_picks' is only meaningful when --ai-analysis ran; otherwise it is an
    # empty screen the scanner has no counterpart for, so it is written only
    # when it actually holds something.
    if args.market in ('india', 'both'):
        _picks = _build_ai_picks(india_data)
        if _picks.get('stocks'):
            india_data['ai_picks'] = _picks
        else:
            india_data.pop('ai_picks', None)
        india_data['_sidebar'] = build_sidebar(india_data)
        with open(os.path.join(SCRIPT_DIR, 'india_data.json'), 'w') as f:
            json.dump(india_data, f, default=str)
        export_excel(india_data, os.path.join(SCRIPT_DIR, 'india_screens.xlsx'), market='India')

    if args.market in ('us', 'both'):
        _picks = _build_ai_picks(us_data)
        if _picks.get('stocks'):
            us_data['ai_picks'] = _picks
        else:
            us_data.pop('ai_picks', None)
        us_data['_sidebar'] = build_sidebar(us_data)
        with open(os.path.join(SCRIPT_DIR, 'us_data.json'), 'w') as f:
            json.dump(us_data, f, default=str)
        export_excel(us_data, os.path.join(SCRIPT_DIR, 'us_screens.xlsx'), market='US')

    elapsed = (datetime.datetime.now() - t0).seconds
    # "_"-prefixed keys are metadata (e.g. _sidebar), not screens.
    def _screens_only(d):
        return {k: v for k, v in d.items() if not k.startswith('_') and isinstance(v, dict)}
    _ind, _us = _screens_only(india_data), _screens_only(us_data)
    total = sum(len(v.get('stocks', [])) for v in _ind.values()) + \
            sum(len(v.get('stocks', [])) for v in _us.values())
    print(f"\n✓ Done in {elapsed//60}m {elapsed%60}s — "
          f"{total} stocks across {len(_ind)+len(_us)} screens")
    print(f"  Reload:  http://localhost:8765/chart_viewer.html")
    print(f"  Excel:   india_screens.xlsx  /  us_screens.xlsx")


if __name__ == '__main__':
    main()

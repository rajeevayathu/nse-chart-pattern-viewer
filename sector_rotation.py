"""
Sector / industry / theme rotation for the Gandiva dashboard.

For every stock the scanner covers (India: NSE universe, US: US universe) this
computes 3-day, 1-week, 1-month and 3-month returns, then ranks groups of
stocks against each other:

  sector    — NSE's sector classification (Nifty Total Market + Microcap 250
              lists); Yahoo's sector for stocks outside them and for the US
  industry  — Yahoo's finer industry (e.g. "Aerospace & Defense", "Auto Parts")
  theme     — membership of NSE sectoral / thematic indices (India only)

Each group gets the median return per period, its excess over the whole
universe, breadth (% of members up on the week, % above their 50-day MA), a
composite score and a rotation status (Leading / Improving / Weakening /
Lagging). A daily snapshot of ranks is kept so rank changes can be tracked.

Writes sector_data.json next to this file. Run directly, or it is called at
the end of generate_charts.py so it refreshes with every rescan.
"""
import os, re, io, json, time, glob, datetime, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SCANNER_DIR = os.path.join(SCRIPT_DIR, '..', 'minervini-trend-template-scanner')
CACHE_DIR   = os.path.join(SCRIPT_DIR, '.sector_cache')
OUT_JSON    = os.path.join(SCRIPT_DIR, 'sector_data.json')
HIST_JSON   = os.path.join(SCRIPT_DIR, 'sector_history.json')

MAP_MAX_AGE = 30 * 86400          # re-classify monthly
PERIODS     = [('r3d', 3), ('r1w', 5), ('r1m', 21), ('r3m', 63)]
WEIGHTS     = {'r3d': 0.15, 'r1w': 0.25, 'r1m': 0.30, 'r3m': 0.30}
MIN_MEMBERS = {'sector': 3, 'industry': 3, 'theme': 3}

NSE_ARCHIVE = 'https://archives.nseindia.com/content/indices/ind_{}.csv'
NSE_SECTOR_LISTS = ['niftytotalmarket_list', 'niftymicrocap250_list']
NSE_THEMES = {
    'niftyindiadefence_list':       'Defence',
    'niftypsubanklist':             'PSU Banks',
    'niftybanklist':                'Banks',
    'niftyfinancelist':             'Financial Services',
    'niftyitlist':                  'IT',
    'niftyindiadigital_list':       'Digital',
    'niftypharmalist':              'Pharma',
    'niftyhealthcarelist':          'Healthcare',
    'niftyfmcglist':                'FMCG',
    'niftyconsumerdurableslist':    'Consumer Durables',
    'niftyautolist':                'Auto',
    'niftymobility_list':           'Mobility',
    'niftymetallist':               'Metals',
    'niftycommoditieslist':         'Commodities',
    'niftyenergylist':              'Energy',
    'niftyoilgaslist':              'Oil & Gas',
    'niftyrealtylist':              'Realty',
    'niftycorehousing_list':        'Housing',
    'niftyinfralist':               'Infrastructure',
    'niftyindiamanufacturing_list': 'Manufacturing',
    'niftycpselist':                'CPSE',
    'niftypselist':                 'PSU (PSE)',
    'niftymedialist':               'Media',
    'niftyindiatourism_list':       'Tourism',
    'niftymnclist':                 'MNC',
}

ETF_RE = re.compile(r'BEES|ETF|LIQID|LIQUID|BETA$|CASE$|ADD$|^(NEXT50|MIDSMALL|MON100|MAFANG|'
                    r'ALPHA|MOM\d+|MOSMALL\d*|LOWVOL\d*|QUAL30|MIDCAP|SMALLCAP|NIFTY\w*|'
                    r'SENSEX\w*|GOLD\w*|SILVER\w*|METAL|PSUBANK|HDFC\w*NIFTY|\w*IETF)$')


# ── classification ────────────────────────────────────────────────────────────
def _fresh(path, max_age):
    return os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age


def _nse_csv(name):
    """NSE index constituent CSV, cached on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{name}.csv')
    if not _fresh(path, MAP_MAX_AGE):
        try:
            req = urllib.request.Request(NSE_ARCHIVE.format(name),
                                         headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(req, timeout=20).read()
            if data.count(b'\n') > 2:
                open(path, 'wb').write(data)
        except Exception as e:
            print(f'    NSE list {name}: {e}')
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# Yahoo sector → NSE sector, used only when an industry can't be mapped
YF_TO_NSE = {
    'Technology': 'Information Technology', 'Utilities': 'Power',
    'Real Estate': 'Realty', 'Energy': 'Oil Gas & Consumable Fuels',
    'Communication Services': 'Telecommunication',
    'Consumer Defensive': 'Fast Moving Consumer Goods', 'Industrials': 'Capital Goods',
    'Basic Materials': 'Chemicals', 'Consumer Cyclical': 'Consumer Durables',
    'Healthcare': 'Healthcare', 'Financial Services': 'Financial Services',
}


def _yf_info(sym):
    """Yahoo sector/industry, or None when Yahoo refuses (rate limit / crumb) so it is retried."""
    import yfinance as yf
    for attempt in range(4):
        try:
            i = yf.Ticker(sym).info or {}
            if i.get('sector') or i.get('industry') or i.get('quoteType'):
                return {'sector': i.get('sector') or '', 'industry': i.get('industry') or ''}
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def classify(tickers, is_us):
    """{ticker: {sector, industry, themes[]}} — cached, unknowns filled from Yahoo."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"map_{'us' if is_us else 'india'}.json")
    cache = {}
    if os.path.exists(path):
        try:
            cache = json.load(open(path))
        except Exception:
            cache = {}
    now = time.time()

    nse_sector, themes = {}, {}
    if not is_us:
        for name in NSE_SECTOR_LISTS:
            df = _nse_csv(name)
            if df is not None and 'Symbol' in df and 'Industry' in df:
                for sym, ind in zip(df['Symbol'], df['Industry']):
                    nse_sector[str(sym).strip()] = str(ind).strip()
        for name, label in NSE_THEMES.items():
            df = _nse_csv(name)
            if df is not None and 'Symbol' in df:
                for sym in df['Symbol']:
                    themes.setdefault(str(sym).strip(), []).append(label)

    todo = [t for t in tickers
            if t not in cache or now - cache[t].get('ts', 0) > MAP_MAX_AGE]
    if todo:
        print(f'    classifying {len(todo)} tickers via Yahoo…', flush=True)
        syms = [t.replace('_', '-') if is_us else t + '.NS' for t in todo]
        failed = 0
        with ThreadPoolExecutor(3) as ex:
            for t, info in zip(todo, ex.map(_yf_info, syms)):
                if info is None:
                    failed += 1          # not cached — retried on the next run
                    continue
                cache[t] = {**info, 'ts': now}
        json.dump(cache, open(path, 'w'))
        if failed:
            print(f'    {failed} lookups failed (will retry next run)', flush=True)

    # India: an industry's NSE sector is the one most of its NSE-listed members carry
    ind_to_nse = {}
    if not is_us:
        votes = {}
        for t, sec in nse_sector.items():
            ind = cache.get(t, {}).get('industry')
            if ind:
                votes.setdefault(ind, {}).setdefault(sec, 0)
                votes[ind][sec] += 1
        ind_to_nse = {ind: max(v, key=v.get) for ind, v in votes.items()}

    out = {}
    for t in tickers:
        c = cache.get(t, {})
        if is_us:
            sector = c.get('sector') or ''
        else:
            sector = (nse_sector.get(t) or ind_to_nse.get(c.get('industry'))
                      or YF_TO_NSE.get(c.get('sector'), c.get('sector')) or '')
        out[t] = {'sector': sector, 'industry': c.get('industry') or '',
                  'themes': themes.get(t, [])}
    return out


# ── prices ────────────────────────────────────────────────────────────────────
def _load_close(path):
    try:
        df = pd.read_pickle(path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c = df['Close'].astype(float).dropna()
        v = df['Volume'].astype(float) if 'Volume' in df else None
        return c, v
    except Exception:
        return None, None


def _live(is_us):
    path = os.path.join(SCRIPT_DIR, 'live_prices_us.json' if is_us else 'live_prices.json')
    try:
        d = json.load(open(path))
        day = pd.Timestamp(d['updated_at'][:10])
        return day, d.get('prices', {})
    except Exception:
        return None, {}


def stock_metrics(is_us):
    cache = os.path.join(SCANNER_DIR, '.cache_us' if is_us else '.cache')
    pat   = '*.pkl' if is_us else '*_NS.pkl'
    live_day, live = _live(is_us)
    newest = None
    rows = {}
    for f in glob.glob(os.path.join(cache, pat)):
        t = os.path.basename(f)[:-4]
        if not is_us:
            t = t[:-3]
        if not is_us and ETF_RE.search(t):
            continue
        c, v = _load_close(f)
        if c is None or len(c) < 70:
            continue
        c.index = pd.to_datetime(c.index).tz_localize(None)
        rows[t] = (c, v)
        newest = c.index[-1] if newest is None or c.index[-1] > newest else newest
    if newest is None:
        return {}, None

    out = {}
    for t, (c, v) in rows.items():
        if (newest - c.index[-1]).days > 10:          # stale / delisted
            continue
        # fold in the live price: replace today's bar, or append if the cache
        # hasn't got today yet (prev_close must match the cached close)
        lp = live.get(t)
        if lp and live_day is not None and lp.get('price'):
            p = float(lp['price'])
            if c.index[-1] == live_day:
                c = c.copy(); c.iloc[-1] = p
            elif c.index[-1] < live_day and lp.get('prev_close') and \
                    abs(float(lp['prev_close']) / c.iloc[-1] - 1) < 0.01:
                c = pd.concat([c, pd.Series([p], index=[live_day])])
        vals = c.values
        m = {'price': round(float(vals[-1]), 2)}
        for key, n in PERIODS:
            m[key] = round((vals[-1] / vals[-1 - n] - 1) * 100, 2) if len(vals) > n and vals[-1 - n] else None
        ma50 = vals[-50:].mean()
        m['above50'] = bool(vals[-1] > ma50)
        hi52 = vals[-252:].max()
        m['from_high'] = round((vals[-1] / hi52 - 1) * 100, 1)
        if v is not None and len(v) >= 20:
            m['turnover'] = float((c.iloc[-20:] * v.reindex(c.index).iloc[-20:].fillna(0)).mean())
        out[t] = m
    return out, max(c.index[-1] for c, _ in rows.values())


# ── dashboard context ─────────────────────────────────────────────────────────
def dashboard_context(is_us):
    """rs_rank and the bullish screens each stock appears in on the dashboard."""
    path = os.path.join(SCANNER_DIR, 'results_us.json' if is_us else 'results.json')
    ctx = {}
    try:
        res = json.load(open(path))
    except Exception:
        return ctx
    for key, scr in res.get('screens', {}).items():
        label = scr.get('label', key)
        for s in scr.get('stocks', []):
            t = str(s.get('ticker', '')).replace('.NS', '')
            c = ctx.setdefault(t, {'rs': s.get('rs_rank'), 'screens': []})
            if c['rs'] is None:
                c['rs'] = s.get('rs_rank')
            if len(c['screens']) < 40:
                c['screens'].append(re.sub(r'[^\w &/+%()≥<>.-]', '', label).strip() or key)
    return ctx


# ── aggregation ───────────────────────────────────────────────────────────────
def _status(short, long_):
    if short is None or long_ is None:
        return ''
    if short >= 0 and long_ >= 0:
        return 'Leading'
    if short >= 0:
        return 'Improving'
    if long_ >= 0:
        return 'Weakening'
    return 'Lagging'


def aggregate(metrics, cls, ctx):
    univ = {k: float(np.nanmedian([m[k] for m in metrics.values() if m.get(k) is not None]))
            for k, _ in PERIODS}
    groupings = {'sector': {}, 'industry': {}, 'theme': {}}
    for t, m in metrics.items():
        c = cls.get(t, {})
        if c.get('sector'):
            groupings['sector'].setdefault(c['sector'], []).append(t)
        if c.get('industry'):
            groupings['industry'].setdefault(c['industry'], []).append(t)
        for th in c.get('themes', []):
            groupings['theme'].setdefault(th, []).append(t)

    result = {}
    for gname, groups in groupings.items():
        rows = []
        for g, members in groups.items():
            if len(members) < MIN_MEMBERS[gname]:
                continue
            row = {'name': g, 'n': len(members)}
            for k, _ in PERIODS:
                vals = [metrics[t][k] for t in members if metrics[t].get(k) is not None]
                row[k] = round(float(np.median(vals)), 2) if vals else None
                row[k + '_x'] = round(row[k] - univ[k], 2) if row[k] is not None else None
            row['up_1w']   = round(100 * np.mean([(metrics[t].get('r1w') or 0) > 0 for t in members]))
            row['above50'] = round(100 * np.mean([metrics[t]['above50'] for t in members]))
            row['in_screens'] = sum(1 for t in members if t in ctx)
            short = np.mean([row['r3d_x'] or 0, row['r1w_x'] or 0])
            long_ = np.mean([row['r1m_x'] or 0, row['r3m_x'] or 0])
            row['status'] = _status(short, long_)
            top = sorted(members, key=lambda t: -(metrics[t].get('r1m') or -999))
            row['members'] = [
                {'t': t, **{k: metrics[t].get(k) for k, _ in PERIODS},
                 'price': metrics[t]['price'], 'from_high': metrics[t]['from_high'],
                 'above50': metrics[t]['above50'],
                 'rs': (ctx.get(t) or {}).get('rs'),
                 'screens': (ctx.get(t) or {}).get('screens', [])[:12],
                 'industry': cls.get(t, {}).get('industry', ''),
                 'sector': cls.get(t, {}).get('sector', '')}
                for t in top]
            rows.append(row)
        # composite: weighted percentile rank across the four periods
        if rows:
            for k, _ in PERIODS:
                ordered = sorted(rows, key=lambda r: (r[k] is None, r[k] if r[k] is not None else 0))
                for i, r in enumerate(ordered):
                    r[k + '_pct'] = 100 * i / max(1, len(rows) - 1) if r[k] is not None else 0
            for r in rows:
                r['score'] = round(sum(WEIGHTS[k] * r[k + '_pct'] for k, _ in PERIODS), 1)
                for k, _ in PERIODS:
                    del r[k + '_pct']
            rows.sort(key=lambda r: -r['score'])
            for i, r in enumerate(rows):
                r['rank'] = i + 1
        result[gname] = rows
    return result, {k: round(v, 2) for k, v in univ.items()}


# ── rank history ──────────────────────────────────────────────────────────────
def update_history(market, as_of, groups):
    try:
        hist = json.load(open(HIST_JSON))
    except Exception:
        hist = {}
    h = hist.setdefault(market, {})
    h[as_of] = {g: {r['name']: r['rank'] for r in rows} for g, rows in groups.items()}
    for d in sorted(h)[:-120]:            # keep ~6 months of daily snapshots
        del h[d]
    json.dump(hist, open(HIST_JSON, 'w'))

    dates = sorted(h)
    def rank_on(back_days):
        target = (pd.Timestamp(as_of) - pd.Timedelta(days=back_days)).strftime('%Y-%m-%d')
        prior = [d for d in dates if d <= target]
        return h[prior[-1]] if prior else None
    ref_1w, ref_1m = rank_on(7), rank_on(30)
    for g, rows in groups.items():
        for r in rows:
            for key, ref in (('chg_1w', ref_1w), ('chg_1m', ref_1m)):
                old = (ref or {}).get(g, {}).get(r['name'])
                r[key] = (old - r['rank']) if old else None
            r['hist'] = [h[d].get(g, {}).get(r['name']) for d in dates[-30:]]
    return dates[-30:]


# ── build ─────────────────────────────────────────────────────────────────────
def build_market(is_us):
    market = 'us' if is_us else 'india'
    metrics, last_bar = stock_metrics(is_us)
    if not metrics:
        print(f'  {market}: no price data')
        return None
    cls = classify(sorted(metrics), is_us)
    ctx = dashboard_context(is_us)
    groups, univ = aggregate(metrics, cls, ctx)
    as_of = str(datetime.date.today())
    hist_dates = update_history(market, as_of, groups)
    unclassified = sum(1 for t in metrics if not cls[t].get('sector'))
    print(f"  {market}: {len(metrics)} stocks · {len(groups['sector'])} sectors · "
          f"{len(groups['industry'])} industries · {len(groups['theme'])} themes"
          f" · {unclassified} unclassified", flush=True)
    return {'as_of': as_of, 'last_bar': str(last_bar.date()) if last_bar is not None else None,
            'stocks': len(metrics), 'universe': univ, 'hist_dates': hist_dates,
            'groups': groups}


def build(markets=('india', 'us')):
    print('\n=== SECTOR / THEME ROTATION ===', flush=True)
    try:
        out = json.load(open(OUT_JSON))
    except Exception:
        out = {}
    for m in markets:
        try:
            d = build_market(m == 'us')
            if d:
                out[m] = d
        except Exception as e:
            print(f'  {m}: sector rotation failed: {e}')
    out['generated_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    tmp = OUT_JSON + '.tmp'
    json.dump(out, open(tmp, 'w'), default=str)
    os.replace(tmp, OUT_JSON)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--market', default='both', choices=['india', 'us', 'both'])
    a = ap.parse_args()
    build(('india', 'us') if a.market == 'both' else (a.market,))

# NSE India Chart Pattern Viewer

A Finviz-style, locally-hosted stock chart viewer for **NSE India** (and US markets) built around Mark Minervini's SEPA / VCP methodology. Generates annotated candlestick charts for every stock across 40+ custom screens and serves them in a fast, dark-mode browser UI — no cloud dependency, no API keys.

---

## Features

### Chart Generation (`generate_charts.py`)

| Feature | Detail |
|---|---|
| **Clean TradingView-style charts** | Candles, moving averages, pivots, volume — no trend lines, S/R boxes or pattern badges |
| **Moving Averages** | EMA21 / SMA50 / SMA200 (daily) · SMA10W / SMA40W (weekly) |
| **Pivots** | Minervini 10/10 pivot highs/lows labelled with price; dashed screen-pivot line with right-axis tag |
| **AVWAP** | Anchored VWAP lines (AV1/AV2) drawn only on the AVWAP screens |
| **Volume** | Up/down coloured bars with a volume MA; y-axis capped at 2.5× the 95th percentile so one spike can't flatten the rest |
| **Daily + Weekly charts** | Each ticker gets a 1Y daily `T.png` (250 bars) and a 5Y weekly `T_W.png` (260 bars, log scale when the range is wide) |
| **6-year history cache** | `.cache_daily_long/` (refreshed weekly) supplies 5Y weekly bars with a full 40-week MA; US dividend-adjusted caches are rescaled at the splice |
| **40+ screens** | Full Template, Near Breakout, VCP Setup, IPO Base, ATH Breakout, Pocket Pivot, RS Leaders, Sector leaders, and more |
| **Manifest-based incremental rebuild** | `--daily` mode skips unchanged stocks — only regenerates charts where price moved >2% |
| **AI visual analysis** | Sends chart PNGs to a local **Ollama** vision model (`llava:7b`); stores BUY/WATCH/AVOID + pattern + confidence per stock |
| **Excel export** | One `.xlsx` per market, one sheet per screen, color-coded AI action cells, summary tab |

### Browser Viewer (`chart_viewer.html`)

| Feature | Detail |
|---|---|
| **Left sidebar navigation** | 40+ screens grouped into Base / Pattern sections, with live search filter |
| **Filter chips** | Filter by Breakout, VCP, On 20MA, RS≥85, Full Template, AI: BUY, AI: WATCH |
| **AI badges** | BUY (green) / WATCH (amber) / AVOID (red) overlay on every card |
| **Sort** | By RS Rank, AI Confidence, or default |
| **Modal detail view** | Full chart image with pattern/score/AI summary; keyboard ←→ navigation |
| **India / US market toggle** | Separate datasets; Excel download link updates automatically |
| **Excel download button** | One-click download of the current market's screen export |
| **Dark mode** | Native dark theme; respects `prefers-color-scheme` |
| **5Y Weekly / 1Y Daily toggle** | View selector in the toolbar; choice remembered in `localStorage` |

### Sector Rotation (`sector_rotation.py` + `sectors.html`)

Tab 3 of the Gandiva portal. Ranks every dashboard stock's **sector, industry and theme** by relative performance over **3D / 1W / 1M / 3M**.

| Feature | Detail |
|---|---|
| **Classification** | NSE sector (from NSE index constituent CSVs), Yahoo Finance industry (cached 30 days), ~25 NSE thematic indices (Banks, PSU Banks, Defence, Tourism, …). US uses Yahoo sectors/industries |
| **Metrics** | Median return per period, excess vs universe, % up this week, % above 50-DMA, number of members in dashboard screens |
| **Score & status** | Weighted percentile score (3D 15% · 1W 25% · 1M 30% · 3M 30%); Leading / Improving / Weakening / Lagging from short- vs long-term excess |
| **Monitoring** | Rebuilt on every chart run (every 30 min in market hours, using live prices); `sector_history.json` keeps 120 snapshots for rank changes and sparklines |
| **Page** | Auto-generated analysis, rotation quadrant, sortable heat table, expandable members with chart modal; auto-reloads every 2 min |

```bash
python3.10 sector_rotation.py --market both   # also runs automatically at the end of generate_charts.py
```

---

## Setup

### Requirements

```
Python 3.10+
yfinance
pandas
matplotlib
mplfinance
openpyxl      # for Excel export
ollama        # optional — for AI visual analysis
```

Install dependencies:

```bash
pip install yfinance pandas matplotlib mplfinance openpyxl
```

Install Ollama (for AI analysis — optional):

```bash
# macOS
brew install ollama
ollama serve &
ollama pull llava:7b   # 4.7 GB vision model
```

---

## Usage

### First run — full build

```bash
# India (NSE) — generate all charts
python generate_charts.py --market india

# US — generate all charts
python generate_charts.py --market us
```

### Daily incremental update (recommended)

```bash
# Only regenerates charts where price moved >2% or ticker is new
python generate_charts.py --market india --daily

# With AI analysis on top-25 near-breakout stocks
python generate_charts.py --market india --daily --ai-analysis --ai-top 25
```

### Launch the viewer

```bash
# Python's built-in HTTP server (required — viewer uses fetch() for JSON)
python -m http.server 8765 --directory /path/to/chart-pattern-viewer
# Open: http://localhost:8765/chart_viewer.html
```

Or use the Live Server extension in VS Code.

---

## AI Visual Analysis

When `--ai-analysis` is passed, the script sends each chart's PNG to a local Ollama vision model. No API key or internet connection required.

**Model compatibility:**
- `llava:7b` — recommended, 4.7 GB, works with Ollama ≥ 0.32
- `llava:13b` — higher quality, 8 GB
- `moondream` — tiny/fast, lower accuracy

The model is asked to identify:
- Chart pattern (VCP, Cup & Handle, Flat Base, etc.)
- Confidence score (0–100)
- Support and resistance levels
- Action signal: **BUY / WATCH / AVOID**
- One-line trading note

Results are stored per-stock in `india_data.json` / `us_data.json` and displayed as color badges in the viewer. Daily runs preserve AI results for unchanged stocks.

---

## Screens (40+)

| Category | Screens |
|---|---|
| **Base** | Full Template · Near Breakout · VCP Setup · High Above MA20 · RS Leaders · New Highs · Near Base Pivot |
| **Breakout** | ATH Breakout · Momentum Breakout · Power Play · Pocket Pivot · IPO Base |
| **Pullback** | First Pullback · On MA50 · On MA200 |
| **Trend** | Stage 2 Uptrend · Stage 4 Downtrend · RS Rating Surge |
| **CCI signals** | CCI Daily Cross · CCI Weekly Cross · CCI Daily Above · CCI Weekly Above |
| **Weekly** | Weekly Pivot · Weekly RS Leader |
| **Sector** | Auto-generated per NIFTY sector |

---

## File Structure

```
chart-pattern-viewer/
├── generate_charts.py   # chart generation + AI analysis + Excel export
├── chart_viewer.html    # browser-based viewer (single HTML file, no dependencies)
├── sector_rotation.py   # sector / industry / theme performance builder
├── sectors.html         # Sectors tab (rotation analysis page)
├── price_updater.py     # live prices every 60s + rescan/regenerate every 30 min
├── sector_data.json     # sector rotation output (auto-created, gitignored)
├── sector_history.json  # rank history snapshots (auto-created, gitignored)
├── .sector_cache/       # NSE constituent CSVs + Yahoo sector map (gitignored)
├── .cache_daily_long/   # 6-year daily history for 5Y weekly charts (gitignored)
├── charts_india/        # generated PNG files (gitignored, auto-created)
├── charts_us/           # generated PNG files (gitignored, auto-created)
├── india_data.json      # screen data for viewer (auto-created, gitignored)
├── us_data.json         # screen data for viewer (auto-created, gitignored)
├── chart_manifest.json  # incremental rebuild index (auto-created, gitignored)
├── india_screens.xlsx   # Excel export (auto-created, gitignored)
└── us_screens.xlsx      # Excel export (auto-created, gitignored)
```

---

## Configuration (inside `generate_charts.py`)

```python
# Chart appearance
--bars 250           # trading days shown in daily chart (CLI default)
WEEKLY_BARS  = 260   # weeks shown in weekly chart (5 years)
DPI          = 150

# Screening universe
NSE_CSV      = '/path/to/nse_symbols.csv'  # 1,312 NSE tickers
MTF_CSV      = '/path/to/mtf_list.csv'     # 1,488 MTF-enabled tickers

# AI
_OLLAMA_MODEL   = 'llava:7b'              # change to llava:13b for better quality
_AI_ENABLED     = False                   # set True or use --ai-analysis flag
AI_PRIORITY_SCREENS = {'near_breakout', 'vcp_setup', ...}  # screens to run AI on
```

---

## Strategy Context

Charts are annotated using Mark Minervini's **SEPA (Specific Entry Point Analysis)** methodology:
- Stage 2 uptrend (price above MA50 above MA200)
- Volatility Contraction Pattern (VCP) — 3–5 contractions in price and volume
- 10/10 pivot standard for S/R identification
- RS (Relative Strength) rating vs benchmark

The companion backtesting scripts (`backtest_canslim_sepa.py`, `backtest_v3_alpha.py`, etc.) achieved 31–35% CAGR on NSE India 2014–2026 using these signals.

---

## License

MIT — free to use, modify, and distribute.

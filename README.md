# NSE India Chart Pattern Viewer

A Finviz-style, locally-hosted stock chart viewer for **NSE India** (and US markets) built around Mark Minervini's SEPA / VCP methodology. Generates annotated candlestick charts for every stock across 40+ custom screens and serves them in a fast, dark-mode browser UI — no cloud dependency, no API keys.

---

## Features

### Chart Generation (`generate_charts.py`)

| Feature | Detail |
|---|---|
| **Candlestick + Volume** | OHLCV bars with color-coded volume |
| **Moving Averages** | MA20 / MA50 / MA200 (daily) · MA10 / MA26 / MA52 (weekly) |
| **Horizontal S/R Zones** | Cluster-based support/resistance from Minervini 10/10 pivot highs/lows (min 2 touches, 1.8% tolerance) |
| **Base Box** | Highlights the tightest consolidation range (≤20%) in last 50 bars |
| **Pattern Badge** | Auto-detects: Breakout · Cup & Handle · Double Bottom · Ascending Triangle · Bull Flag · Tight Base · Stage 2/4 |
| **Daily + Weekly charts** | Each ticker gets a `_D.png` (daily 200 bars) and `_W.png` (weekly 104 bars) |
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
BARS_DAILY   = 200   # trading days shown in daily chart
BARS_WEEKLY  = 104   # weeks shown in weekly chart

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

# Tech Pulse

Does online chatter about a company predict what its stock does next?

Tech Pulse collects public discussion from six sources going back to 2015, scores the
sentiment, tests whether that sentiment leads price movements, and backtests trading
strategies built on the resulting signals. Results are published as an interactive
Streamlit dashboard.

**Dashboard:** deployed from `stock_tracking/app.py`

---

## What the data says

Sentiment is measured across roughly 2,000 tickers with enough coverage to analyse.
The headline backtest result, starting from $10,000 in January 2015:

| Strategy | Description |
|----------|-------------|
| Sector Rotation | Weekly rotation into the sector with the strongest buzz |
| Position Trader | 21-day holds triggered by high-conviction sentiment |
| Buy & Hold | Equal weight across all tickers, never rebalanced |
| S&P 500 (SPY) | Passive benchmark |

Both sentiment-driven strategies have outperformed the passive benchmarks over the
full period. Figures are regenerated on every run, so consult the dashboard for
current numbers rather than trusting any hardcoded here.

---

## Data sources

| Source | What it captures | Weight |
|--------|------------------|--------|
| GDELT | Mainstream financial news (Reuters, FT, Bloomberg) | 0.9 |
| Hacker News | Technical audience, early signal on tech companies | 1.0 |
| Reddit (5 subreddits) | Retail investor sentiment | 0.7 – 1.1 |
| News sentiment API | Alpha Vantage and Polygon article sentiment | 1.1 |
| SEC EDGAR | 8-K material event filings | 1.5 |
| Stock prices | Daily OHLCV via yfinance | n/a |

Weights reflect how predictive each source has proven to be. Section 8 of
`B_correlation_engine.ipynb` recalculates them from measured hit rates rather than
intuition.

---

## Repository layout

```
.
├── Python files A-D/              Analysis pipeline, run in order
│   ├── A_sentiment_engine.ipynb   Collect, match, score, aggregate
│   ├── B_correlation_engine.ipynb Correlations, key moves, source weights
│   ├── C_strategy_engine.ipynb    Backtests, per-sector and portfolio-wide
│   └── D_charts.ipynb             Static charts (superseded by the dashboard)
│
├── Raw Data Pulled/               Source data, one folder per source
│   ├── hn_data/  reddit_data/  gdelt_data/
│   └── edgar_data/  stocktwits_data/
│
├── stock_tracking/                The Streamlit application
│   ├── app.py                     Entry point and navigation
│   ├── utils.py                   Data loading, colour system, sidebar
│   ├── pages/                     Seven dashboard pages
│   ├── stocks/                    Daily price history per ticker
│   ├── sentiment_outputs/         Quarterly signal files from notebook A
│   ├── correlation_outputs/       Correlation results from notebook B
│   └── strategy_outputs/          Equity curves and trade log from notebook C
│
├── incremental_updater.ipynb      Refresh all six sources in one pass
└── archive/                       Superseded collectors and experiments
```

---

## Running the pipeline

Set a GitHub token with write access in `.env`:

```
GITHUB_TOKEN=your_token_here
ALPHA_VANTAGE_KEY=your_key_here     # optional, for the news sentiment source
POLYGON_KEY=your_key_here           # optional
```

**Refreshing data.** Run `incremental_updater.ipynb`. Section 3 reports how stale each
source is before you commit to a full pass; sections 4 to 9 update each source in turn,
fetching only what is missing.

**Regenerating results.** Run the notebooks in order — each depends on the previous one:

1. `A_sentiment_engine.ipynb` — matches text to tickers, scores sentiment, writes
   quarterly signal files
2. `B_correlation_engine.ipynb` — measures whether sentiment leads price, identifies
   key moves, calibrates source weights
3. `C_strategy_engine.ipynb` — backtests each strategy and writes equity curves

The dashboard reads the saved outputs directly from GitHub, so it picks up new results
without redeployment.

---

## How companies are matched to text

The hardest problem here is deciding that an article mentioning "Nvidia" refers to NVDA
while an article containing the word "on" does not refer to ON Semiconductor. Matching
runs in three layers:

1. **Exact ticker symbols** — `$NVDA` or `NVDA` as a standalone word, minimum three
   characters, excluding a blocklist of common English words
2. **Manual aliases** — product and brand names that do not resemble the company name,
   such as Ozempic to NVO or Azure to MSFT
3. **Fuzzy company names** — `rapidfuzz` at a threshold of 90, restricted to company
   names of at least eight characters so short generic names cannot match

Section 9 of `B_correlation_engine.ipynb` audits the result and reports how many matched
tickers look like false positives.

---

## The dashboard

| Page | Question it answers |
|------|--------------------|
| Overview | How does sentiment track price for this company? |
| Signal Quality | When sentiment fires, how often is it right? |
| Correlation | Which companies have the most predictive sentiment? |
| Strategies | Would trading on these signals have beaten the market? |
| Key Moments | What was being said before a large price move? |
| Sources | Which feeds carry the most signal? |
| What If | What would a given investment have returned? |

Charts use an Okabe-Ito colour palette that remains readable under red-green colour
blindness. Sentiment direction is encoded blue for positive and orange for negative
rather than the conventional green and red, which are indistinguishable to roughly
one in twelve men.

---

## Known limitations

- Sentiment is measured from headlines and post titles rather than full article text
- VADER is a general-purpose sentiment model, not one tuned for financial language
- Backtests exclude slippage and assume trades fill at the closing price
- Coverage is uneven: heavily discussed companies produce far more signal than the rest
- The ticker universe still contains some false matches from text search

## Disclaimer

Built for research and coursework. Nothing here is investment advice.

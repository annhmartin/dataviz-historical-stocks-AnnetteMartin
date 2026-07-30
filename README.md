# Tech Pulse

**Does online chatter predict what a stock does next?**

Six sources of public discussion — mainstream news, Hacker News, five subreddits, investor
posts and SEC filings — scored for sentiment across eleven years and matched against daily
prices for roughly two thousand companies.

The short answer is yes, but modestly, and not in the direction you would expect.

🔗 **[Live dashboard](https://share.streamlit.io)** · Data Visualization · Final Individual
Project · Summer 2026

---

## What the analysis found

**Pessimism reads as a buy signal.** Positive chatter predicts gains at a rate modestly
above chance. Negative chatter predicts correctly *less* often than a coin flip, meaning
prices usually rose after it. Public negativity behaves closer to a contrarian indicator
than a warning — the most actionable finding here, and the reason the model treats the two
directions asymmetrically rather than as mirror images.

**The edge has faded.** Accuracy was measurably higher before 2021 than after. Commission-free
trading, a surge in retail participation, and funds mining social data directly all arrived
in that window, which is consistent with an edge being competed away as more people traded
on the same public signals.

**Volume and accuracy are unrelated.** The feeds that publish most are not the ones most
often right, and reliability barely improves as coverage grows. Weighting sources by volume —
the intuitive choice — would hand influence to whichever feed is noisiest.

**It survives execution, but not comfortably.** Sentiment-driven strategies finish ahead of
both passive benchmarks after transaction costs, while trailing the index across a
substantial share of rolling twelve-month windows. The gains arrive in bursts separated by
long stretches of underperformance.

Figures regenerate on every run, so consult the dashboard or the notebook for current
numbers rather than trusting any quoted in prose.

---

## Deliverables

| Artifact | File | What it is |
|----------|------|-----------|
| Analysis notebook | `Tech_Pulse_Analysis.ipynb` | Exploratory analysis plus ten analytical questions, each answered with one explanatory Plotly figure |
| Dashboard | `stock_tracking/` | Eight-page interactive Streamlit app, deployed on Community Cloud |
| Presentation | `presentation.pdf` | Slide deck summarising the findings |

---

## Data sources

| Source | Type | What it captures |
|--------|------|------------------|
| GDELT | Text, temporal | Mainstream financial news (Reuters, FT, Bloomberg) |
| Hacker News | Text, temporal | Technical audience, early signal on tech companies |
| Reddit — 5 subreddits | Text, categorical | Retail investor sentiment |
| Alpha Vantage / Polygon | Numerical | Scored news sentiment per ticker |
| SEC EDGAR | Categorical, temporal | 8-K material event filings |
| Daily prices | Numerical, temporal | Returns and realised volatility, via yfinance |

Attribute types span numerical, categorical, temporal and unstructured text. Source weights
are calibrated from measured directional hit rates in section 8 of the correlation notebook
rather than set by intuition.

---

## Repository layout

```
.
├── Tech_Pulse_Analysis.ipynb      Main analysis — ten questions, Plotly figures
│
├── Python files A-D/              Data pipeline, run in order
│   ├── A_sentiment_engine.ipynb   Collect, match to tickers, score, aggregate
│   ├── B_correlation_engine.ipynb Correlations, key moves, source weights, ticker audit
│   ├── C_strategy_engine.ipynb    Backtests, portfolio-wide and per sector
│   └── D_charts.ipynb             Earlier static charts, superseded by the notebook above
│
├── Raw Data Pulled/               Source data, one folder per source
│   ├── hn_data/  reddit_data/  gdelt_data/
│   └── edgar_data/  stocktwits_data/
│
├── stock_tracking/                Streamlit application
│   ├── app.py                     Entry point and navigation
│   ├── utils.py                   Data loading, colour system, sidebar filters
│   ├── pages/                     Eight dashboard pages
│   ├── .streamlit/config.toml     Theme matching the chart palette
│   ├── stocks/                    Daily price history per ticker
│   ├── sentiment_outputs/         Quarterly signal files from notebook A
│   ├── correlation_outputs/       Correlation results from notebook B
│   └── strategy_outputs/          Equity curves and trade log from notebook C
│
├── incremental_updater.ipynb      Refresh all six sources in one pass
└── archive/                       Superseded collectors and experiments
```

---

## The dashboard

Eight pages. The first leads with the findings; the rest are for exploration, filtered by
sector and date range from the sidebar.

| Page | Question it answers |
|------|--------------------|
| **Key Findings** | What did the analysis conclude? Headline metrics and the three main results |
| **Overview** | How does sentiment track price for one company? |
| **Signal Quality** | When sentiment fires, how often is it right? |
| **Correlation** | Which companies have the most predictive chatter? |
| **Strategies** | Would trading these signals have beaten the market? |
| **Key Moments** | What was being said before a large price move? |
| **Sources** | Which feeds carry the most, and the strongest, signal? |
| **What If** | What would a given investment have returned? |

---

## Design notes

Charts are built entirely in Plotly and follow a colour system defined once in
`stock_tracking/utils.py`.

Colour carries meaning in three separate roles, and the roles deliberately do not share
hues. **Sentiment** uses a diverging blue-to-orange scale. **Strategies** use a categorical
set where the two passive benchmarks are muted grey so the two signal-driven strategies
carry the saturated colour. **Signal quality quadrants** encode correctness by temperature —
cool for correct calls, warm for incorrect.

Values come from the Okabe-Ito and Paul Tol qualitative schemes, both constructed to stay
separable under red-green colour blindness. The conventional green/red pairing for
positive and negative is deliberately avoided: it collapses into a single mustard tone for
roughly one man in twelve, and sentiment direction is the most important encoding in the
project. Palettes were verified against dichromatic simulation rather than assumed safe.

Figures state their finding in the title rather than naming the variables, use muted grey
for context with one highlight colour for focus, and drop gridlines and chart borders that
carry no information.

---

## Running the pipeline

Set credentials in `.env`:

```
GITHUB_TOKEN=your_token_here
ALPHA_VANTAGE_KEY=your_key_here     # optional, for the news sentiment source
POLYGON_KEY=your_key_here           # optional
```

**Refreshing data.** Run `incremental_updater.ipynb`. Section 3 reports how stale each
source is before you commit to a full pass; sections 4 to 9 update each source, fetching
only what is missing.

**Regenerating results.** Run in order — each notebook depends on the previous one:

1. `A_sentiment_engine.ipynb` — matches text to tickers, scores sentiment, writes quarterly
   signal files plus a pre-aggregated bundle for the dashboard
2. `B_correlation_engine.ipynb` — measures whether sentiment leads price, identifies key
   moves, calibrates source weights, audits for false ticker matches
3. `C_strategy_engine.ipynb` — backtests each strategy portfolio-wide and per sector
4. `Tech_Pulse_Analysis.ipynb` — the ten analytical questions

The dashboard reads saved outputs directly from GitHub, so it picks up new results without
redeployment.

---

## How companies are matched to text

The hardest problem here is deciding that an article mentioning "Nvidia" refers to NVDA
while one containing the word "on" does not refer to ON Semiconductor. Matching runs in
three layers:

1. **Exact ticker symbols** — `$NVDA` or `NVDA` as a standalone word, minimum three
   characters, excluding a blocklist of common English words
2. **Manual aliases** — product and brand names that do not resemble the company name,
   such as Ozempic to NVO or Azure to MSFT
3. **Fuzzy company names** — `rapidfuzz` at a threshold of 90, restricted to company names
   of at least eight characters so short generic names cannot match

Section 9 of `B_correlation_engine.ipynb` audits the result, flagging tickers absent from
the SEC register, matching common words, showing near-zero sentiment variance, or drawing
almost all coverage from a single source.

---

## Limitations

- Sentiment is scored from headlines and post titles, not full article text
- VADER is a general-purpose model, not tuned for financial language where "beat
  expectations" and "missed estimates" carry meanings it does not know
- Backtests exclude slippage and assume fills at the closing price
- Coverage is heavily skewed toward a handful of large companies; the long tail is thin
- Matching company names in free text produces false positives that survive filtering
- Several results rest on modest samples once the data is split several ways, and the
  confidence bands on those figures are part of the finding rather than decoration
- This is one historical period in one market, and the post-2021 decay is itself evidence
  that these relationships do not hold still

## Disclaimer

Built for coursework and research. Nothing here is investment advice.

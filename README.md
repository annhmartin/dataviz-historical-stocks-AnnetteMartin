# Tech Pulse

**Does online chatter predict what a stock does next?**

Six sources of public discussion — mainstream news, Hacker News, five subreddits, investor
posts and SEC filings — scored for sentiment across eleven years and matched against daily
prices for roughly two thousand companies.

The short answer is yes, but modestly — and the edge has been shrinking.

🔗 **[Live dashboard](https://share.streamlit.io)** · Data Visualization · Final Individual
Project · Summer 2026

📓 **Main analysis notebook: [`Tech_Pulse_Analysis.ipynb`](Tech_Pulse_Analysis.ipynb)** — in
the repository root. This is the primary Python deliverable: exploratory analysis followed by
ten analytical questions, each answered with one Plotly figure. The notebooks in
`Python files A-D/` are the upstream data pipeline that produces the inputs it reads.

---

## What the analysis found

**Positive sentiment carries a small but consistent edge.** When public discussion turns
positive, prices rise over the following week modestly more often than chance would predict,
across roughly thirty thousand signals. The margin is small by design — an obvious edge
would already have been arbitraged away — but it is consistent enough to build a strategy on.

Negative signals in this dataset performed below chance, which would imply prices rose after
negative chatter. That contradicts a well-established body of research and rests on a sample
roughly a third the size, so it is treated as a limitation rather than a finding. The likely
causes are sample size, VADER's inability to read financial phrasing such as "missed
estimates", and scoring headlines rather than full articles. The analysis leans on positive
signals accordingly.

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

Six independent sources, all public. Nothing here is synthetic or a teaching set.

### 1. GDELT Global Knowledge Graph
- **Where:** [gdeltproject.org](https://www.gdeltproject.org/) · API endpoint
  [`api.gdeltproject.org/api/v2/doc/doc`](https://api.gdeltproject.org/api/v2/doc/doc)
- **Docs:** [GDELT 2.0 DOC API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
- **What:** Worldwide news coverage indexed continuously. Queried per company name for
  article headlines, publisher domain and timestamp.
- **In this repo:** `Raw Data Pulled/gdelt_data/gdelt_{TICKER}_{YEAR}.csv`
- **Note:** the `artlist` mode does not return usable tone values, so headlines are scored
  with the sentiment model rather than relying on GDELT's own tone field.

### 2. Hacker News
- **Where:** [Algolia HN Search API](https://hn.algolia.com/api) ·
  [`hn.algolia.com/api/v1/search_by_date`](https://hn.algolia.com/api/v1/search_by_date)
- **What:** Story titles, points and comment counts from a technically literate audience.
  Filtered to stories with at least 3 points.
- **In this repo:** `Raw Data Pulled/hn_data/hn_{YEAR}.csv`

### 3. Reddit
- **Where:** [Arctic Shift API](https://arctic-shift.photon-reddit.com/) ·
  [project on GitHub](https://github.com/ArthurHeitmann/arctic_shift)
- **What:** Historical post titles, scores and comment counts from r/wallstreetbets,
  r/stocks, r/investing, r/technology and r/SecurityAnalysis.
- **In this repo:** `Raw Data Pulled/reddit_data/reddit_{YEAR}.csv`
- **Note:** Arctic Shift is used because Reddit's own API no longer serves deep history.

### 4. News sentiment — Alpha Vantage and Polygon.io
- **Where:** [Alpha Vantage NEWS_SENTIMENT](https://www.alphavantage.co/documentation/#news-sentiment)
  · [Polygon.io Ticker News](https://polygon.io/docs/stocks/get_v2_reference_news)
- **Keys:** free tiers at [alphavantage.co](https://www.alphavantage.co/support/#api-key)
  and [polygon.io](https://polygon.io/)
- **What:** Article-level sentiment scores already tagged to a ticker, with relevance
  weighting.
- **In this repo:** `Raw Data Pulled/stocktwits_data/stocktwits_{TICKER}.csv`
- **Note:** the folder name is historical. StockTwits was the original source but moved
  behind Cloudflare bot protection, so these two APIs replaced it.

### 5. SEC EDGAR
- **Where:** [sec.gov/edgar](https://www.sec.gov/edgar/searchedgar/companysearch) ·
  submissions API [`data.sec.gov/submissions/CIK{cik}.json`](https://data.sec.gov/submissions/)
  · ticker register
  [`sec.gov/files/company_tickers.json`](https://www.sec.gov/files/company_tickers.json)
- **What:** Filing dates and types, chiefly 8-K material event notices. The ticker register
  also supplies the official company names used for text matching and in the dashboard.
- **In this repo:** `Raw Data Pulled/edgar_data/edgar_{TICKER}.csv`
- **Note:** the SEC requires a descriptive User-Agent header on every request.

### 6. Daily prices
- **Where:** [Yahoo Finance](https://finance.yahoo.com/) via
  [`yfinance`](https://github.com/ranaroussi/yfinance)
- **What:** Adjusted daily OHLCV, from which returns and 60-day realised volatility are
  derived.
- **In this repo:** `stock_tracking/stocks/prices_{TICKER}.csv` with an index at
  `stock_tracking/stocks/index.csv`

### Summary

| Source | Type | Location in repo |
|--------|------|------------------|
| GDELT | Text, temporal | `Raw Data Pulled/gdelt_data/` |
| Hacker News | Text, temporal | `Raw Data Pulled/hn_data/` |
| Reddit ×5 | Text, categorical | `Raw Data Pulled/reddit_data/` |
| Alpha Vantage / Polygon | Numerical | `Raw Data Pulled/stocktwits_data/` |
| SEC EDGAR | Categorical, temporal | `Raw Data Pulled/edgar_data/` |
| Daily prices | Numerical, temporal | `stock_tracking/stocks/` |

Attribute types span numerical, categorical, temporal and unstructured text. Source weights
are calibrated from measured directional hit rates in section 8 of the correlation notebook
rather than set by intuition.

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
| **Sentiment vs Price** | How does sentiment track price for one company? |
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

## Future work

**Replace VADER with FinBERT.** VADER is a rule-based model built for general English and
has no concept of finance: "beat expectations" scores positive because *beat* is a positive
word, while "missed estimates" barely registers. FinBERT is fine-tuned on analyst reports,
earnings calls and financial news, and reads context rather than matching a lexicon.

This matters most for the unresolved negative-signal result. If VADER is systematically
misreading bad news, the negative sample is not merely small but contaminated, and FinBERT
would establish whether the below-chance reading is a real effect or a measurement artifact.
Two changes pair naturally with it: scoring full article text rather than headlines, giving
the model the context it is built to use, and using FinBERT's three-way classification with
confidence scores so low-confidence readings can be excluded rather than averaged in.

The cost is compute. VADER scores half a million headlines in minutes locally; FinBERT needs
GPU inference or a paid API, making a full re-score of eleven years across six sources a
substantial job.

Other directions worth pursuing: intraday rather than daily resolution, since a signal that
works at T+5 may be stronger at T+1 with finer timing; explicit market-regime conditioning,
given the post-2021 decay; and extending beyond US equities to test whether the relationship
holds in markets with different retail participation.

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

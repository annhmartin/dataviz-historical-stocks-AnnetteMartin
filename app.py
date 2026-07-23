
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tech Pulse — Sentiment Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
GITHUB_REPO   = "annhmartin/dataviz-historical-stocks-AnnetteMartin"
GITHUB_TOKEN  = st.secrets.get("GITHUB_TOKEN", None)
OUTPUT_PREFIX = "sentiment_outputs"
CORR_PREFIX   = "correlation_outputs"
STRAT_PREFIX  = "strategy_outputs"
STOCKS_PREFIX = "stocks"
SENTIMENT_THRESHOLD = 0.05

SECTOR_MAP = {
    'AI Accelerators'      : ['NVDA','AMD'],
    'Semiconductor Supply' : ['TSM','INTC','QCOM'],
    'Big Tech'             : ['GOOGL','MSFT','AAPL','META'],
    'Cloud / SaaS'         : ['AMZN','SNOW','DDOG','CRM','NOW','MDB'],
    'Cybersecurity'        : ['CRWD','PANW','OKTA'],
    'Enterprise AI'        : ['PLTR'],
    'Macro Risk'           : ['COIN','TSLA'],
    'Portfolio'            : ['INCY','KGC','NVO','PM','WPM'],
    'Consumer Tech'        : ['NFLX','SPOT','PINS'],
    'Enterprise Fintech'   : ['PYPL'],
}

DEFAULT_TICKERS = ['NVDA','AAPL','MSFT','GOOGL','META',
                   'CRWD','PANW','PLTR','NVO','PM']

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#cccccc', 'axes.grid': True,
    'grid.color': '#e8e8e8', 'grid.linewidth': 0.7,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_csv(path, token=None):
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{path}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code == 404:
        return pd.DataFrame()
    resp.raise_for_status()
    content = resp.text.strip()
    if not content:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(content), low_memory=False)

@st.cache_data(ttl=3600)
def load_signals():
    frames = []
    for year in range(2015, datetime.now().year + 1):
        for q in [1, 2, 3, 4]:
            df = load_csv(f"{OUTPUT_PREFIX}/daily_signals_{year}_Q{q}.csv", GITHUB_TOKEN)
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=3600)
def load_price(ticker):
    df = load_csv(f"{STOCKS_PREFIX}/prices_{ticker}.csv", GITHUB_TOKEN)
    if df.empty:
        return pd.DataFrame()
    df['Date'] = pd.to_datetime(df['Date'])
    df['daily_return'] = df['Close'].pct_change(fill_method=None)
    df['pct_7d'] = df['Close'].pct_change(7, fill_method=None) * 100
    df['cumret'] = (1 + df['daily_return'].fillna(0)).cumprod() * 100 - 100
    return df.sort_values('Date')

# ── Load all data ─────────────────────────────────────────────────────────────
with st.spinner("Loading data from GitHub..."):
    daily_signals   = load_signals()
    best_per_ticker = load_csv(f"{CORR_PREFIX}/best_per_ticker.csv",  GITHUB_TOKEN)
    df_corr         = load_csv(f"{CORR_PREFIX}/corr_matrix.csv",      GITHUB_TOKEN)
    df_key_moves    = load_csv(f"{CORR_PREFIX}/key_moves.csv",        GITHUB_TOKEN)
    source_attr     = load_csv(f"{OUTPUT_PREFIX}/source_attribution.csv", GITHUB_TOKEN)
    df_spillover    = load_csv(f"{CORR_PREFIX}/spillover_pairs.csv",  GITHUB_TOKEN)
    df_equity       = load_csv(f"{STRAT_PREFIX}/equity_curves.csv",   GITHUB_TOKEN)
    df_trades       = load_csv(f"{STRAT_PREFIX}/trade_log.csv",       GITHUB_TOKEN)

if not df_equity.empty and 'date' in df_equity.columns:
    df_equity['date'] = pd.to_datetime(df_equity['date'])
if not df_trades.empty and 'entry_date' in df_trades.columns:
    df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
if not df_key_moves.empty:
    for col in ['move_date','sent_date']:
        if col in df_key_moves.columns:
            df_key_moves[col] = pd.to_datetime(df_key_moves[col])

all_tickers = sorted(daily_signals['ticker'].unique().tolist()) if not daily_signals.empty else DEFAULT_TICKERS
sig_col = 'adaptive_sentiment' if 'adaptive_sentiment' in daily_signals.columns else 'norm_sentiment'

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📡 Tech Pulse")
st.sidebar.markdown("---")

selected_tickers = st.sidebar.multiselect(
    "Select tickers",
    options=all_tickers,
    default=[t for t in DEFAULT_TICKERS if t in all_tickers],
    help="Choose companies to display across all charts"
)

sector_filter = st.sidebar.selectbox(
    "Or filter by sector",
    options=["All"] + list(SECTOR_MAP.keys()),
    index=0
)
if sector_filter != "All":
    sector_tickers = [t for t in SECTOR_MAP[sector_filter] if t in all_tickers]
    if sector_tickers:
        selected_tickers = sector_tickers

date_range = st.sidebar.date_input(
    "Date range",
    value=[pd.Timestamp("2018-01-01").date(), pd.Timestamp.today().date()],
    min_value=pd.Timestamp("2015-01-01").date(),
    max_value=pd.Timestamp.today().date()
)
start_date = pd.Timestamp(date_range[0])
end_date   = pd.Timestamp(date_range[1]) if len(date_range) > 1 else pd.Timestamp.today()

st.sidebar.markdown("---")
st.sidebar.caption(f"**Tickers selected:** {len(selected_tickers)}")
st.sidebar.caption(f"**Total tickers with signal:** {len(all_tickers):,}")
if not daily_signals.empty:
    st.sidebar.caption(f"**Signal days loaded:** {len(daily_signals):,}")

if not selected_tickers:
    st.warning("Please select at least one ticker in the sidebar.")
    st.stop()

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Sentiment vs Price",
    "🎯 Signal Quality",
    "🔗 Correlation",
    "🌊 Spillover",
    "💰 Portfolio Strategies",
    "🔍 Key Moments",
    "📰 Source Attribution"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: SENTIMENT VS PRICE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Sentiment vs Price")
    st.markdown(
        "**Bars** = daily sentiment score (green=positive, red=negative, gray=neutral) | "
        "**Dark line** = 30-day rolling average sentiment | "
        "**Blue line/shading** = 7-day % price change"
    )

    view_mode = st.radio("View mode", ["One at a time", "All stacked"], horizontal=True)

    if view_mode == "One at a time":
        ticker = st.selectbox("Select ticker", selected_tickers, key="t1_ticker")
        tickers_to_plot = [ticker]
    else:
        tickers_to_plot = selected_tickers

    rolling_window = st.slider("Sentiment rolling window (days)", 7, 90, 30, key="t1_roll")

    for ticker in tickers_to_plot:
        sig = daily_signals[
            (daily_signals['ticker']==ticker) &
            (daily_signals['date'] >= start_date) &
            (daily_signals['date'] <= end_date)
        ].copy()
        price = load_price(ticker)
        if price.empty:
            st.warning(f"{ticker}: no price data")
            continue
        price = price[(price['Date'] >= start_date) & (price['Date'] <= end_date)]

        has_signal = sig[sig_col].notna().sum()

        fig, ax1 = plt.subplots(figsize=(12, 4), facecolor='white')
        ax2 = ax1.twinx()

        # Price % change
        pct = price['pct_7d'].dropna()
        ax2.fill_between(price['Date'], 0, price['pct_7d'].fillna(0),
                         where=price['pct_7d'].fillna(0) >= 0, color='#27ae60', alpha=0.12)
        ax2.fill_between(price['Date'], 0, price['pct_7d'].fillna(0),
                         where=price['pct_7d'].fillna(0) < 0,  color='#e74c3c', alpha=0.12)
        ax2.plot(price['Date'], price['pct_7d'], color='#2980b9', linewidth=1.2, alpha=0.7)
        ax2.axhline(0, color='#aaaaaa', linewidth=0.5)
        ax2.set_ylabel('7-day % price change', fontsize=8, color='#2980b9')
        ax2.tick_params(axis='y', labelcolor='#2980b9')
        ax2.grid(False)

        # Sentiment
        if has_signal >= 5:
            sv = sig[sig_col].fillna(0)
            sent_smooth = sig[sig_col].rolling(rolling_window, min_periods=1).mean()
            colors_s = ['#27ae60' if v >= SENTIMENT_THRESHOLD
                         else ('#e74c3c' if v <= -SENTIMENT_THRESHOLD else '#bdc3c7')
                         for v in sv]
            ax1.bar(sig['date'], sv, color=colors_s, alpha=0.5, width=2)
            ax1.plot(sig['date'], sent_smooth, color='#2c3e50', linewidth=2, alpha=0.9,
                     label=f'{rolling_window}d rolling')
        else:
            ax1.text(0.5, 0.5, f'Only {has_signal} signal days',
                     ha='center', va='center', transform=ax1.transAxes, color='#888888')

        ax1.axhline(0, color='#aaaaaa', linewidth=0.5)
        ax1.set_ylabel('Sentiment', fontsize=8)
        ax1.set_facecolor('white')
        ax1.set_title(f'{ticker} — Sentiment (bars) + 7-day % price change (blue) | {has_signal:,} signal days',
                      fontsize=11, fontweight='bold')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SIGNAL QUALITY (FOUR QUADRANT)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Signal Quality — Four Quadrant")
    st.markdown("""
    Each quadrant shows what happened after the sentiment signal fired:
    - **True Positive** (top-right, green): positive buzz → stock went UP ✓
    - **False Positive** (bottom-right, red): positive buzz → stock went DOWN ✗
    - **True Negative** (bottom-left, blue): negative buzz → stock went DOWN ✓
    - **False Negative** (top-left, orange): negative buzz → stock went UP ✗

    **Dot size** = number of stories that day. **Orange line** = overall trend.
    Above 50% accuracy means the signal beats random chance.
    """)

    hold_days = st.slider("Hold period (trading days)", 1, 21, 5, key="t2_hold")
    ticker_q  = st.selectbox("Select ticker", selected_tickers, key="t2_ticker")

    sig_q   = daily_signals[(daily_signals['ticker']==ticker_q) &
                              (daily_signals['date'] >= start_date) &
                              (daily_signals['date'] <= end_date)].copy()
    price_q = load_price(ticker_q)

    if not price_q.empty and not sig_q.empty:
        price_q = price_q.set_index('Date').sort_index()
        price_q['daily_return'] = price_q['Close'].pct_change(fill_method=None)
        rows = []
        for _, srow in sig_q.iterrows():
            sv = srow.get(sig_col, np.nan)
            if pd.isna(sv) or abs(sv) < SENTIMENT_THRESHOLD: continue
            future = price_q.index[price_q.index > srow['date']]
            if len(future) < hold_days: continue
            ep  = price_q.loc[future[0], 'Close']
            xp  = price_q.loc[future[hold_days-1], 'Close']
            ret = (xp - ep) / ep * 100
            if sv >= SENTIMENT_THRESHOLD and ret > 0:   q,c = 'True Positive',  '#27ae60'
            elif sv >= SENTIMENT_THRESHOLD and ret <= 0: q,c = 'False Positive', '#e74c3c'
            elif sv <= -SENTIMENT_THRESHOLD and ret < 0: q,c = 'True Negative',  '#2980b9'
            else:                                         q,c = 'False Negative', '#e67e22'
            rows.append({'sent':sv,'ret':ret,'q':q,'c':c,'stories':srow.get('story_count',1)})

        if rows:
            df_plot = pd.DataFrame(rows)
            counts  = df_plot['q'].value_counts()
            total   = len(df_plot)
            tp = counts.get('True Positive',0)
            fp = counts.get('False Positive',0)
            tn = counts.get('True Negative',0)
            fn = counts.get('False Negative',0)
            acc = (tp+tn)/total

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{acc:.1%}")
            col2.metric("True Positive", tp, f"{tp/total:.0%}")
            col3.metric("False Positive", fp, f"{fp/total:.0%}")
            col4.metric("True Negative", tn, f"{tn/total:.0%}")
            col5.metric("False Negative", fn, f"{fn/total:.0%}")

            fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
            ax.scatter(df_plot['sent'], df_plot['ret'], c=df_plot['c'], alpha=0.65,
                       s=df_plot['stories'].clip(1,50)*8+15,
                       edgecolors='white', linewidths=0.4, zorder=3)
            ax.axhline(0, color='#888888', linewidth=1)
            ax.axvline(0, color='#888888', linewidth=1)
            xmax = df_plot['sent'].abs().max()*1.1
            ymax = df_plot['ret'].abs().max()*1.1
            ax.set_xlim(-xmax,xmax); ax.set_ylim(-ymax,ymax)
            ax.text( xmax*.55,  ymax*.88, f'True Positive\n{tp} ({tp/total:.0%})',
                     fontsize=9, color='#27ae60', ha='center', fontweight='bold')
            ax.text(-xmax*.55,  ymax*.88, f'False Negative\n{fn} ({fn/total:.0%})',
                     fontsize=9, color='#e67e22', ha='center', fontweight='bold')
            ax.text( xmax*.55, -ymax*.88, f'False Positive\n{fp} ({fp/total:.0%})',
                     fontsize=9, color='#e74c3c', ha='center', fontweight='bold')
            ax.text(-xmax*.55, -ymax*.88, f'True Negative\n{tn} ({tn/total:.0%})',
                     fontsize=9, color='#2980b9', ha='center', fontweight='bold')
            if len(df_plot) > 5:
                z  = np.polyfit(df_plot['sent'], df_plot['ret'], 1)
                xs = np.linspace(-xmax, xmax, 100)
                ax.plot(xs, np.poly1d(z)(xs), color='#f39c12', linewidth=2, zorder=4, label='Trend')
            ax.set_xlabel('← Negative buzz  |  Positive buzz →', fontsize=10)
            ax.set_ylabel(f'Price change {hold_days}d later (%)', fontsize=10)
            ax.set_title(f'{ticker_q} — Signal Quality | Accuracy: {acc:.1%} | n={total} signals',
                         fontsize=12, fontweight='bold')
            ax.set_facecolor('white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough signal days to compute quadrant chart for this ticker + date range.")
    else:
        st.warning("No data available for selected ticker.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Sentiment-to-Price Correlation")
    st.markdown("""
    Shows which tickers have the strongest statistical relationship between sentiment and future price moves.
    - **Green bar** = positive buzz predicted price going UP
    - **Red bar** = positive buzz predicted price going DOWN (contrarian signal — negative buzz may be more useful)
    - **★** = statistically significant (p < 0.05)
    - **Label** = which signal type and how many days forward worked best
    """)

    if df_corr.empty or best_per_ticker.empty:
        st.warning("No correlation data. Run B_correlation_engine.ipynb first.")
    else:
        sig_only = st.checkbox("Show significant only (p < 0.05)", value=True)
        n_show   = st.slider("Number of tickers to show", 10, 50, 20, key="t3_n")
        min_n    = st.slider("Minimum data points (n)", 50, 500, 200, key="t3_minn")

        plot_df = best_per_ticker.copy()
        if 'n' in plot_df.columns:
            plot_df = plot_df[plot_df['n'] >= min_n]
        if sig_only and 'significant' in plot_df.columns:
            plot_df = plot_df[plot_df['significant']==True]
        plot_df = plot_df.sort_values('corr').head(n_show)

        if plot_df.empty:
            st.info("No tickers match the current filters. Try lowering the minimum n or unchecking significant only.")
        else:
            fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df)*0.4)), facecolor='white')
            colors = ['#e74c3c' if v < 0 else '#27ae60' for v in plot_df['corr']]
            labels = plot_df['ticker'] + ' (' + plot_df['signal'] + ' T+' + plot_df['horizon'].astype(str) + ')'
            bars   = ax.barh(labels, plot_df['corr'], color=colors, edgecolor='white')
            ax.axvline(0, color='#aaaaaa', linewidth=0.8)
            if 'significant' in plot_df.columns:
                for bar, (_, row) in zip(bars, plot_df.iterrows()):
                    if row.get('significant'):
                        ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                                '★', va='center', fontsize=9, color='gold')
            ax.set_xlabel('Pearson Correlation (sentiment → forward price return)', fontsize=10)
            ax.set_title(f'Best Sentiment Signal per Ticker | {len(plot_df)} shown', fontsize=12, fontweight='bold')
            ax.set_facecolor('white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            with st.expander("View raw correlation table"):
                st.dataframe(plot_df[['ticker','signal','horizon','corr','pval','n','significant']].reset_index(drop=True))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SPILLOVER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Cross-Company Sentiment Spillover")
    st.markdown("""
    When Company A gets talked about, does Company B's stock move?
    This shows confirmed spillover pairs — relationships that held consistently across 90-day rolling windows.

    **How to read:** Row ticker's SENTIMENT predicts Column ticker's PRICE MOVE.
    """)

    if df_spillover.empty:
        st.warning("No spillover data. Run B_correlation_engine.ipynb first.")
    else:
        min_consistency = st.slider("Min consistency (% of windows correlated)", 0.20, 0.70, 0.35, key="t4_cons")
        filtered_spill  = df_spillover[df_spillover['consistency'] >= min_consistency]

        st.metric("Confirmed spillover pairs", len(filtered_spill))

        if not filtered_spill.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top positive pairs")
                pos = filtered_spill[filtered_spill['direction']=='positive'].nlargest(10,'avg_corr')
                if not pos.empty:
                    st.dataframe(pos[['sent_ticker','price_ticker','avg_corr','consistency']].reset_index(drop=True))
            with col2:
                st.subheader("Top negative pairs")
                neg = filtered_spill[filtered_spill['direction']=='negative'].nsmallest(10,'avg_corr')
                if not neg.empty:
                    st.dataframe(neg[['sent_ticker','price_ticker','avg_corr','consistency']].reset_index(drop=True))

            # Heatmap of spillover
            try:
                pivot = filtered_spill.pivot_table(
                    index='sent_ticker', columns='price_ticker', values='avg_corr'
                )
                if not pivot.empty and len(pivot) <= 30:
                    fig, ax = plt.subplots(figsize=(min(16, len(pivot.columns)*0.8+4),
                                                    min(12, len(pivot)*0.6+3)), facecolor='white')
                    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                                center=0, linewidths=0.3, ax=ax,
                                cbar_kws={'label': 'Avg rolling correlation'}, annot_kws={'size': 7})
                    ax.set_title('Cross-Company Spillover\nRow sentiment → Column price',
                                 fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: PORTFOLIO STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Portfolio Strategy Comparison")

    if df_equity.empty:
        st.warning("No strategy data. Run C_strategy_engine.ipynb first.")
    else:
        strat_colors = {
            'SP_500_SPY_'         : '#f39c12',
            'Buy__Hold'           : '#1a1a2e',
            'Sector_Rotation'     : '#e74c3c',
            'Position_Trader'     : '#8e44ad',
            'Opt_Stop_Global_'    : '#27ae60',
            'Opt_Stop_Per_Sector_': '#00b4d8',
        }
        strat_labels = {
            'SP_500_SPY_'         : 'S&P 500 (SPY)',
            'Buy__Hold'           : 'Buy & Hold',
            'Sector_Rotation'     : 'Sector Rotation',
            'Position_Trader'     : 'Position Trader',
            'Opt_Stop_Global_'    : 'Optimal Stop (Global)',
            'Opt_Stop_Per_Sector_': 'Optimal Stop (Sector)',
        }

        # Filter equity by date
        eq_filtered = df_equity[(df_equity['date'] >= start_date) & (df_equity['date'] <= end_date)]

        # Summary metrics
        metric_cols = st.columns(len([c for c in df_equity.columns if c != 'date']))
        for col_name, mcol in zip([c for c in df_equity.columns if c != 'date'], metric_cols):
            series = eq_filtered.set_index('date')[col_name].dropna()
            if not series.empty:
                final  = series.iloc[-1]
                start  = series.iloc[0]
                ret    = (final/start - 1) * 100
                label  = strat_labels.get(col_name, col_name.replace('_',' ').strip())
                mcol.metric(label, f"${final:,.0f}", f"{ret:+.1f}%")

        st.markdown("---")

        # Main equity chart
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), facecolor='white',
                                  gridspec_kw={'height_ratios':[3,1]})
        ax = axes[0]
        spy_series = None
        for col_name in df_equity.columns:
            if col_name == 'date': continue
            series = eq_filtered.set_index('date')[col_name].dropna()
            if series.empty: continue
            color = strat_colors.get(col_name, '#888888')
            label = strat_labels.get(col_name, col_name.replace('_',' ').strip())
            ls    = '--' if 'SPY' in col_name else '-'
            lw    = 2.5 if col_name in ('Sector_Rotation','Buy__Hold','SP_500_SPY_') else 1.8
            ax.plot(series.index, series.values, color=color, linestyle=ls,
                    linewidth=lw, label=f'{label}: ${series.iloc[-1]:,.0f}', alpha=0.9)
            if 'SPY' in col_name: spy_series = series

        rot_col = [c for c in df_equity.columns if 'Rotation' in c]
        if rot_col and spy_series is not None:
            rot_s = eq_filtered.set_index('date')[rot_col[0]].dropna()
            spy_a = spy_series.reindex(rot_s.index).interpolate('time')
            ax.fill_between(rot_s.index, spy_a, rot_s,
                            where=rot_s >= spy_a, color='#a9dfbf', alpha=0.2, label='Rotation > SPY')
            ax.fill_between(rot_s.index, spy_a, rot_s,
                            where=rot_s <  spy_a, color='#f5b7b1', alpha=0.2, label='SPY > Rotation')

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
        ax.set_title('Portfolio Growth: All Strategies vs S&P 500\n(green = rotation beating SPY, red = SPY winning)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, facecolor='white', loc='upper left', ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_facecolor('white')

        # Winning strategy bar
        ax2 = axes[1]
        monthly = eq_filtered.set_index('date').resample('ME').last().dropna(how='all')
        if not monthly.empty:
            winner_each = monthly.drop(columns=[c for c in monthly.columns if pd.isna(monthly[c]).all()]).idxmax(axis=1)
            for month, winner in winner_each.items():
                c = strat_colors.get(winner, '#cccccc')
                ax2.bar(month, 1, width=20, color=c, alpha=0.9)
            ax2.set_yticks([])
            ax2.set_ylabel('Leader', fontsize=8)
            ax2.set_title('Leading strategy each month (color = winner)', fontsize=10, fontweight='bold')
            legend_patches = [mpatches.Patch(color=v, label=strat_labels.get(k,k))
                              for k,v in strat_colors.items()]
            ax2.legend(handles=legend_patches, fontsize=7, loc='upper left', ncol=3, facecolor='white')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax2.set_facecolor('white')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Trade log
        if not df_trades.empty:
            st.markdown("---")
            st.subheader("Trade Log")
            strat_filter = st.multiselect(
                "Filter by strategy",
                options=df_trades['strategy'].unique().tolist() if 'strategy' in df_trades.columns else [],
                default=df_trades['strategy'].unique().tolist() if 'strategy' in df_trades.columns else []
            )
            trades_show = df_trades.copy()
            if strat_filter and 'strategy' in trades_show.columns:
                trades_show = trades_show[trades_show['strategy'].isin(strat_filter)]
            trades_show = trades_show[
                (trades_show['entry_date'] >= start_date) &
                (trades_show['entry_date'] <= end_date)
            ] if 'entry_date' in trades_show.columns else trades_show

            wins = (trades_show['return_pct'] > 0).sum() if 'return_pct' in trades_show.columns else 0
            total_t = len(trades_show)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total trades", total_t)
            c2.metric("Win rate", f"{wins/total_t:.1%}" if total_t>0 else "N/A")
            c3.metric("Avg return", f"{trades_show['return_pct'].mean():.2f}%" if 'return_pct' in trades_show.columns else "N/A")
            c4.metric("Best trade", f"{trades_show['return_pct'].max():.1f}%" if 'return_pct' in trades_show.columns else "N/A")

            st.dataframe(trades_show.sort_values('entry_date', ascending=False).head(100).reset_index(drop=True),
                         use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: KEY MOMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Key Moment Zoom")
    st.markdown("""
    Shows dates where a stock had a large abnormal price move (z-score > 2.0)
    and sentiment preceded it. 
    - **Orange line** = the day of the big price move
    - **Purple dashed** = when the sentiment signal fired before the move
    - **Top panel** = % price change from start of window
    - **Bottom panel** = sentiment bars in the same window
    """)

    if df_key_moves.empty:
        st.warning("No key moves data. Run B_correlation_engine.ipynb first.")
    else:
        show_predicted = st.checkbox("Show only correctly predicted moves", value=True)
        ticker_km = st.selectbox("Select ticker", selected_tickers, key="t6_ticker")

        km_filtered = df_key_moves[df_key_moves['ticker'] == ticker_km].copy()
        if show_predicted and 'predicted' in km_filtered.columns:
            km_filtered = km_filtered[km_filtered['predicted']==True]
        km_filtered = km_filtered.sort_values('zscore', key=abs, ascending=False).head(5)

        if km_filtered.empty:
            st.info(f"No {'predicted ' if show_predicted else ''}key moves found for {ticker_km}. Try unchecking 'predicted only' or choosing a different ticker.")
        else:
            move_options = [f"{row['move_date'].strftime('%b %Y')} ({row['return_pct']:+.1f}%)"
                            for _, row in km_filtered.iterrows()]
            selected_move = st.selectbox("Select event to zoom", move_options)
            move_idx = move_options.index(selected_move)
            move = km_filtered.iloc[move_idx]

            move_date = move['move_date']
            win_start = move_date - pd.Timedelta(days=30)
            win_end   = move_date + pd.Timedelta(days=15)

            sig_km = daily_signals[
                (daily_signals['ticker']==ticker_km) &
                (daily_signals['date'] >= win_start) &
                (daily_signals['date'] <= win_end)
            ].copy()
            price_km = load_price(ticker_km)
            price_km = price_km[(price_km['Date'] >= win_start) & (price_km['Date'] <= win_end)] if not price_km.empty else pd.DataFrame()

            if not price_km.empty:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), facecolor='white')

                ref = price_km['Close'].iloc[0]
                pct_w = (price_km['Close'] - ref) / ref * 100
                ax1.plot(price_km['Date'], pct_w, color='#2980b9', linewidth=2)
                ax1.fill_between(price_km['Date'], 0, pct_w, where=pct_w>=0, color='#a9dfbf', alpha=0.4)
                ax1.fill_between(price_km['Date'], 0, pct_w, where=pct_w<0,  color='#f5b7b1', alpha=0.4)
                ax1.axhline(0, color='#aaaaaa', linewidth=0.8)
                ax1.axvline(move_date, color='#f39c12', linewidth=3, zorder=5, label='Big move')
                ax1.axvline(move['sent_date'], color='#8e44ad', linewidth=2, linestyle='--', zorder=4,
                            label=f'Sentiment {move["days_before"]}d before')
                ax1.set_ylabel('% change from window start', fontsize=9)
                ax1.set_title(
                    f'{ticker_km} — {move_date.strftime("%B %Y")} | Move: {move["return_pct"]:+.1f}% (z={move["zscore"]:+.1f})\n'
                    f'Sentiment {move["days_before"]} day(s) before: {move["sentiment"]:+.3f} | Sources: {move.get("sources_active","unknown")}',
                    fontsize=11, fontweight='bold'
                )
                ax1.legend(fontsize=8, facecolor='white')
                ax1.set_facecolor('white')

                if not sig_km.empty:
                    sv = sig_km[sig_col].fillna(0)
                    bar_colors = ['#27ae60' if v >= SENTIMENT_THRESHOLD
                                   else ('#e74c3c' if v <= -SENTIMENT_THRESHOLD else '#bdc3c7')
                                   for v in sv]
                    ax2.bar(sig_km['date'], sv, color=bar_colors, alpha=0.8, width=1.5)
                    ax2.axhline(0, color='#aaaaaa', linewidth=0.7)
                    ax2.axvline(move_date, color='#f39c12', linewidth=3, zorder=5)
                    ax2.axvline(move['sent_date'], color='#8e44ad', linewidth=2, linestyle='--', zorder=4)
                ax2.set_ylabel('Sentiment score', fontsize=9)
                ax2.set_title('Sentiment in 30 days before and 15 days after the move', fontsize=10)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax2.set_facecolor('white')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: SOURCE ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Source Attribution")
    st.markdown("""
    Shows which data source (HN, Reddit, GDELT, StockTwits) contributes most articles
    for each ticker. This shows **volume**, not quality — a source with more articles
    doesn't necessarily have better sentiment signal.

    - **GDELT** = mainstream financial news (Reuters, Bloomberg, FT etc)
    - **HN** = Hacker News — tech expert audience
    - **Reddit** = retail investor sentiment (WSB, stocks, investing)
    - **StockTwits** = direct investor sentiment tagged by ticker
    """)

    if source_attr.empty:
        st.warning("No source attribution data. Run A_sentiment_engine.ipynb first.")
    else:
        n_src = st.slider("Number of tickers to show", 10, 50, 25, key="t7_n")
        top_by_items = (source_attr.groupby('ticker')['item_count'].sum()
                        .sort_values(ascending=False).head(n_src).index.tolist())
        pivot = source_attr[source_attr['ticker'].isin(top_by_items)].pivot_table(
            index='ticker', columns='source', values='item_count', fill_value=0
        )
        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
        pivot = pivot.sort_values(pivot.columns[0], ascending=False)

        src_colors = {
            'hn':'#e67e22',
            'reddit_wallstreetbets':'#e74c3c',
            'reddit_stocks':'#c0392b',
            'reddit_investing':'#922b21',
            'reddit_technology':'#7b241c',
            'reddit_SecurityAnalysis':'#641e16',
            'gdelt':'#2980b9',
            'stocktwits':'#27ae60',
            'edgar_8k':'#8e44ad',
        }

        fig, ax = plt.subplots(figsize=(13, max(7, len(pivot)*0.4)), facecolor='white')
        bottom = np.zeros(len(pivot))
        for col in pivot.columns:
            color = src_colors.get(col, '#888888')
            ax.bar(pivot.index, pivot[col], bottom=bottom, label=col,
                   color=color, edgecolor='white', alpha=0.85)
            bottom += pivot[col].values
        ax.set_ylabel('Share of signal articles (%)', fontsize=10)
        ax.set_title('Source Contribution per Ticker (% of articles — volume, not quality)',
                     fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend(fontsize=8, facecolor='white', bbox_to_anchor=(1.01,1), loc='upper left')
        ax.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("View dominant source per ticker"):
            dominant = (source_attr.sort_values('item_count', ascending=False)
                        .drop_duplicates(subset='ticker')
                        [[ 'ticker','source','item_count','avg_sentiment']]
                        .rename(columns={'source':'dominant_source','item_count':'articles'})
                        .sort_values('articles', ascending=False))
            st.dataframe(dominant.reset_index(drop=True), use_container_width=True)

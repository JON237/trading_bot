import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from src.fetcher import DataFetcher
    from src.indicators import add_indicators_1h
    from src.strategy import MultiTimeframeStrategy
    from src.backtest import calculate_position_size
except ImportError as e:
    st.error(f"Failed to load dependencies: {e}")
    st.stop()

# ----------- CONFIG -----------
st.set_page_config(page_title="1H Crypto Bot Monitor", layout="wide", page_icon="📈")

# ----------- STATE LOADING -----------
@st.cache_resource
def load_model():
    model_path = os.path.join(script_dir, "models", "best_model_1h.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

trades_file = os.path.join(script_dir, "logs", "trades_1h.csv")
if os.path.exists(trades_file):
    trades = pd.read_csv(trades_file)
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
else:
    trades = pd.DataFrame(columns=["timestamp", "action", "price", "pnl", "reason"])

initial_capital = 10000.0
portfolio = initial_capital
wins = 0
losses = 0

equity_list = []
if len(trades) > 0:
    equity_list.append({"timestamp": trades['timestamp'].iloc[0], "equity": portfolio})
else:
    equity_list.append({"timestamp": datetime.now(), "equity": portfolio})

position_open = False
entry_price = 0.0

# Simulate portfolio values based on 30% flat capital sizing for generic rendering
for _, row in trades.iterrows():
    if row['action'] == 'BUY':
        position_open = True
        entry_price = row['price']
    elif row['action'] == 'SELL':
        position_open = False
        pnl_pct = row['pnl']
        if pd.notna(pnl_pct):
            if pnl_pct > 0: wins += 1
            else: losses += 1
            # Mock estimation of portfolio compound
            trade_usd = portfolio * 0.30
            portfolio += trade_usd * (pnl_pct / 100)
            equity_list.append({"timestamp": row['timestamp'], "equity": portfolio})

equity_df = pd.DataFrame(equity_list)

total_trades = wins + losses
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

if len(trades) > 0:
    start_time = trades['timestamp'].iloc[0]
    uptime_hrs = (datetime.now() - start_time).total_seconds() / 3600
else:
    uptime_hrs = 0.0

# ----------- LIVE DATA PROCESSING -----------
fetcher = DataFetcher(exchange_id='binance')

try:
    df_1h = fetcher.fetch_ohlcv("BTC/USDT", timeframe="1h", since_days=5)
    df_1h = df_1h.tail(200).reset_index(drop=True)
    
    df_4h = fetcher.fetch_ohlcv("BTC/USDT", timeframe="4h", since_days=10)
    df_4h = df_4h.tail(50).reset_index(drop=True)
    
    # Process indicators
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        df_1h_features = add_indicators_1h(df_1h.copy())
    
    # Strategy evaluation
    strategy = MultiTimeframeStrategy()
    if model:
        signal, conf, trend, score = strategy.generate_signals(df_1h_features, df_4h, model)
    else:
        signal, conf, trend, score = 0, 0.0, "UNKNOWN", 0
        
    latest_row = df_1h_features.iloc[-1]
    
except Exception as e:
    st.error(f"Live data extraction failed: {e}")
    st.stop()


# ----------- UI LAYOUT -----------
st.title("⚡ 1H Trading Bot Live Monitor")

# Add manual refresh
st.sidebar.markdown("### Controls")
if st.sidebar.button("Force Refresh Data 🔄"):
    st.rerun()

# --- ROW 1: METRICS ---
c1, c2, c3, c4 = st.columns(4)
diff_usd = portfolio - initial_capital
pnl_total_pct = (diff_usd / initial_capital) * 100

c1.metric("Portfolio Value", f"${portfolio:,.2f}", f"{diff_usd:+.2f} USD")
c2.metric("Total PnL", f"{pnl_total_pct:+.2f}%", f"Total Trades: {int(total_trades)}")
c3.metric("Win Rate", f"{win_rate:.1f}%", f"{wins} W / {losses} L")
c4.metric("Active Since", f"{uptime_hrs:.1f} hrs", f"Signals Online: {'Yes' if model else 'No'}")


# --- ROW 2: CHART ---
st.markdown("### 📊 Live 1H Chart & Tracking")
df_chart = df_1h_features.tail(48).copy()

fig = go.Figure(data=[go.Candlestick(
    x=df_chart['timestamp'],
    open=df_chart['open'],
    high=df_chart['high'],
    low=df_chart['low'],
    close=df_chart['close'],
    name="BTC/USDT"
)])

if 'EMA_9' in df_chart.columns:
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['EMA_9'], mode='lines', name='EMA 9', line=dict(color='rgba(0,191,255,0.7)')))
if 'EMA_21' in df_chart.columns:
    fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['EMA_21'], mode='lines', name='EMA 21', line=dict(color='rgba(255,165,0,0.7)')))

# Add trade markers
recent_trades = trades.tail(20)
for _, t in recent_trades.iterrows():
    if t['timestamp'] >= df_chart['timestamp'].min():
        color = "#00ff00" if t['action'] == 'BUY' else "#ff0000"
        symbol = "triangle-up" if t['action'] == 'BUY' else "triangle-down"
        
        # Determine exact matched timestamp or closest
        fig.add_trace(go.Scatter(
            x=[t['timestamp']], y=[t['price']],
            mode='markers', marker=dict(color=color, size=16, symbol=symbol, line=dict(width=2, color='black')),
            name=t['action'] + " Entry",
            hoverinfo="name+y"
        ))

# Add Stop / TP lines if active
if position_open:
    atr_val = latest_row.get('ATR_14', entry_price * 0.05)
    _, sl, tp = calculate_position_size(portfolio, entry_price, atr_val, 0.02)
    fig.add_hline(y=sl, line_dash="dash", line_color="#ff4757", annotation_text="🛡️ Stop-Loss")
    fig.add_hline(y=tp, line_dash="dash", line_color="#2ed573", annotation_text="🎯 Take-Profit")
    fig.add_hline(y=entry_price, line_dash="solid", line_color="#ffffff", annotation_text="💵 Entry", opacity=0.3)

fig.update_layout(
    template="plotly_dark",
    height=550,
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)


# --- ROW 3: PANELS ---
st.markdown("---")
c_left, c_right = st.columns([1, 2])

with c_left:
    st.markdown("### 🤖 ML Signal Engine")
    
    # Visual Badge
    status_color = "#2ed573" if signal == 1 else ("#ff4757" if signal == -1 else "#747d8c")
    status_text = "BUY" if signal == 1 else ("SELL" if signal == -1 else "HOLD")
    st.markdown(f"<h2 style='text-align: center; color: {status_color};'>{status_text}</h2>", unsafe_allow_html=True)
    
    st.markdown("**Confidence Match**")
    st.progress(conf)
    st.markdown(f"<div style='text-align: right;'>{conf*100:.1f}%</div>", unsafe_allow_html=True)
    
    st.markdown(f"**4H Global Trend:** `{trend}`")
    st.markdown(f"**Algorithmic Score:** `{score}/10`")
    
    st.markdown("---")
    st.markdown("#### Key Drivers")
    r1, r2 = st.columns(2)
    r1.metric("Fast RSI (7)", f"{latest_row.get('RSI_7', 0):.1f}")
    r2.metric("Slow RSI (14)", f"{latest_row.get('RSI_14', 0):.1f}")
    
    macd_val = latest_row.get('MACD_hist_12_26_9', 0)
    r1.metric("MACD Hist", f"{macd_val:.2f}", "Bullish" if macd_val > 0 else "Bearish")
    bb_pos = latest_row.get('BB_position', 0)
    r2.metric("BB Position", f"{bb_pos:.2f}", "Oversold" if bb_pos < 0.2 else ("Overbought" if bb_pos > 0.8 else "Neutral"))

with c_right:
    st.markdown("### 📝 Last 10 MTF Trades")
    if len(trades) > 0:
        display_df = trades.tail(10).iloc[::-1].copy()
        
        # Clean up column formatting for visuals
        if 'price' in display_df.columns:
            display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}")
        if 'pnl' in display_df.columns:
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) and x != 0 else "-")
            
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=360
        )
    else:
        st.info("No trades executed yet.")


# --- ROW 4: EQUITY CURVE ---
st.markdown("---")
st.markdown("### 📈 Estimated Equity Curve (vs Market)")

try:
    df_bnh = fetcher.fetch_ohlcv("BTC/USDT", timeframe="1d", since_days=30)
    df_bnh['bnh_equity'] = (initial_capital / df_bnh['close'].iloc[0]) * df_bnh['close']
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=equity_df['timestamp'], y=equity_df['equity'], name='Bot Equity Estimate', line=dict(color='#00ff00', width=3)))
    fig2.add_trace(go.Scatter(x=df_bnh['timestamp'], y=df_bnh['bnh_equity'], name='Market Reality (Buy & Hold)', line=dict(color='#00a8ff', dash='dot')))
    
    fig2.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.warning("Could not calculate background market curve.")

# --- AUTO RELOAD ---
st.sidebar.markdown("---")
st.sidebar.markdown("*UI will auto-refresh roughly every 60s.*")

# Magic autoreload delay (placed at end of script so it doesn't block DOM rendering)
time.sleep(60)
st.rerun()

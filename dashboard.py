"""
dashboard.py
A live monitoring dashboard for the trading bot using Streamlit.
Run using: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import time
import os
import sys
import matplotlib.pyplot as plt

# Ensure src/ is in the path to import project modules securely
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")

# Page title
st.title("Trading Bot Dashboard")

def load_trades():
    logs_file = os.path.join("logs", "trades.csv")
    if os.path.exists(logs_file):
        try:
            df = pd.read_csv(logs_file)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        except pd.errors.EmptyDataError:
            pass
    return pd.DataFrame()

trades_df = load_trades()

# Section 1 — Key Metrics
st.header("Section 1 — Key Metrics")
col1, col2, col3, col4 = st.columns(4)

total_pnl = 0.0
num_trades = 0
win_rate = 0.0
current_position = "NONE"

if not trades_df.empty:
    sells = trades_df[trades_df['action'] == 'SELL']
    
    total_pnl = sells['pnl'].sum()
    num_trades = len(sells)
    
    if num_trades > 0:
        winning_trades = len(sells[sells['pnl'] > 0])
        win_rate = (winning_trades / num_trades) * 100
        
    last_action = trades_df.iloc[-1]['action']
    if last_action == 'BUY':
        current_position = "LONG"

col1.metric("Total PnL (%)", f"{total_pnl:.2f}%")
col2.metric("Number of Trades", num_trades)
col3.metric("Win Rate (%)", f"{win_rate:.2f}%")
col4.metric("Current Position", current_position)

st.divider()

# Section 2 — Equity Curve
st.header("Section 2 — Equity Curve")
if not trades_df.empty:
    # Build timeline equity 
    plot_df = trades_df.copy()
    plot_df['cumulative_pnl'] = plot_df['pnl'].cumsum()
    plot_df['equity'] = 100 + plot_df['cumulative_pnl']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    plt.style.use('dark_background')
    
    # Plot equity line
    ax.plot(plot_df['timestamp'], plot_df['equity'], color='#00a8ff', linewidth=2, label='Equity %')
    
    # Plot Buy/Sell markers
    buys = plot_df[plot_df['action'] == 'BUY']
    sells = plot_df[plot_df['action'] == 'SELL']
    
    ax.scatter(buys['timestamp'], buys['equity'], color='lime', marker='^', s=150, zorder=5, label='BUY')
    ax.scatter(sells['timestamp'], sells['equity'], color='red', marker='v', s=150, zorder=5, label='SELL')
    
    ax.set_ylabel("Virtual Portfolio Equity (%)")
    ax.grid(color='#2f3640', linestyle='--', linewidth=0.5)
    ax.legend()
    fig.autofmt_xdate()
    
    st.pyplot(fig)
else:
    st.info("No trades logged yet. Equity curve cannot be generated.")

st.divider()

# Section 3 — Recent Trades
st.header("Section 3 — Recent Trades")
if not trades_df.empty:
    recent_trades = trades_df.tail(10).sort_values(by='timestamp', ascending=False)
    # Format dataframe for display
    display_df = recent_trades[['timestamp', 'action', 'price', 'pnl', 'reason']].copy()
    st.dataframe(display_df, use_container_width=True)
else:
    st.info("No trades executed yet.")

st.divider()

# Section 4 — Last Signal
st.header("Section 4 — Last Signal")
try:
    from fetcher import DataFetcher
    from indicators import add_indicators
    from strategy import MLStrategy
    import warnings
    warnings.filterwarnings('ignore')
    
    st.text("Fetching live market data...")
    
    fetcher = DataFetcher('binance')
    df = fetcher.fetch_ohlcv("BTC/USDT", timeframe="1h", since_days=5)
    
    if not df.empty:
        df = df.tail(100).reset_index(drop=True)
        df = add_indicators(df)
        
        strategy = MLStrategy(model_path="models/rf_model.pkl")
        df = strategy.generate_signals(df)
        
        if not df.empty:
            latest = df.iloc[-1]
            
            c1, c2, c3, c4 = st.columns(4)
            rsi_val = latest.get('RSI_14', 0.0)
            c1.metric("RSI (14)", f"{rsi_val:.2f}")
            
            # Find MACD dynamically due to potential pandas-ta suffix variations
            macd_col = [c for c in df.columns if c.startswith('MACDh')]
            macd_val = latest[macd_col[0]] if macd_col else 0.0
            c2.metric("MACD Hist", f"{macd_val:.2f}")
            
            sma_dist = latest.get('dist_sma20', 0.0) * 100
            c3.metric("SMA 20 Position", f"{sma_dist:.2f}%")
            
            conf = latest.get('confidence', 0.0) * 100
            sig = latest.get('signal', 0)
            action = "BUY" if sig == 1 else "SELL" if sig == -1 else "HOLD"
            c4.metric(f"Model Signal: {action}", f"Confidence: {conf:.1f}%")
            
        else:
            st.warning("Insufficient data generated after indicator application.")
    else:
        st.warning("Failed to fetch recent data from the exchange.")
        
except ImportError as e:
    st.error(f"Missing core module files: {e}")
except Exception as e:
    st.error(f"Error fetching live signal metrics: {e}")

# Auto-refresh mechanism using modern Streamlit controls
time.sleep(60)
st.rerun()

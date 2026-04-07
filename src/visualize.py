"""
visualize.py
Generates overview charts for historical data or backtesting results.
Creates a 3-panel plot using matplotlib showing price action, volume, and RSI.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_chart(df: pd.DataFrame, symbol: str = "BTC/USDT"):
    """
    Generate a 3-panel chart representing price, volume, and RSI.
    Saves the output as a `.png` file.
    """
    if df.empty:
        print("⚠️ No data to plot.")
        return
        
    print(f"📊 Plotting {symbol} chart...")
    
    # Enable dark background style
    plt.style.use('dark_background')
    
    # Create the 3-panel layout: Height ratios 3:1:1
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)
    
    # Ensure timestamp column is used for X-axis
    if 'timestamp' in df.columns:
        x_dates = pd.to_datetime(df['timestamp'])
    else:
        x_dates = pd.to_datetime(df.index)
        
    # Get date range for title
    date_start = x_dates.iloc[0].strftime('%Y-%m-%d')
    date_end = x_dates.iloc[-1].strftime('%Y-%m-%d')
    fig.suptitle(f"{symbol} Overview ({date_start} to {date_end})", fontsize=16, fontweight='bold', color='white')

    # Data separation for green/red candles
    up = df['close'] >= df['open']
    down = df['close'] < df['open']
    
    color_up = '#00ff00'  # Lime green
    color_down = '#ff0000' # Bright red
    
    # ==========================
    # Panel 1: Price Action
    # ==========================
    # Draw candlestick wicks
    ax1.vlines(x_dates[up], df['low'][up], df['high'][up], color=color_up, linewidth=1)
    ax1.vlines(x_dates[down], df['low'][down], df['high'][down], color=color_down, linewidth=1)
    
    # Draw candlestick bodies
    ax1.vlines(x_dates[up], df['open'][up], df['close'][up], color=color_up, linewidth=4)
    ax1.vlines(x_dates[down], df['open'][down], df['close'][down], color=color_down, linewidth=4)
    
    # Plot Indicators: SMA 20 & 50
    if 'SMA_20' in df.columns:
        ax1.plot(x_dates, df['SMA_20'], color='#fbc531', linewidth=1.5, label='SMA 20')
    if 'SMA_50' in df.columns:
        ax1.plot(x_dates, df['SMA_50'], color='#00a8ff', linewidth=1.5, label='SMA 50')
        
    # Plot Bollinger Bands
    bbl_cols = [col for col in df.columns if col.startswith('BBL')]
    bbu_cols = [col for col in df.columns if col.startswith('BBU')]
    
    if bbl_cols and bbu_cols:
        bbl = df[bbl_cols[0]]
        bbu = df[bbu_cols[0]]
        ax1.plot(x_dates, bbu, color='mediumpurple', linewidth=1, linestyle='--', alpha=0.7)
        ax1.plot(x_dates, bbl, color='mediumpurple', linewidth=1, linestyle='--', alpha=0.7)
        ax1.fill_between(x_dates, bbl, bbu, color='mediumpurple', alpha=0.1, label='Bollinger Bands')
        
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(color='#2f3640', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left', frameon=True, fontsize='small', framealpha=0.6)
    
    # ==========================
    # Panel 2: Volume
    # ==========================
    ax2.bar(x_dates[up], df['volume'][up], color=color_up, alpha=0.8)
    ax2.bar(x_dates[down], df['volume'][down], color=color_down, alpha=0.8)
    ax2.set_ylabel('Volume')
    ax2.grid(color='#2f3640', linestyle='--', linewidth=0.5)
    
    # ==========================
    # Panel 3: RSI
    # ==========================
    rsi_cols = [col for col in df.columns if col.startswith('RSI')]
    if rsi_cols:
        rsi = df[rsi_cols[0]]
        ax3.plot(x_dates, rsi, color='cyan', linewidth=1.5)
        # horizontal lines for overbought/oversold boundaries
        ax3.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(30, color='green', linestyle='--', alpha=0.5)
        # Shade regions
        ax3.fill_between(x_dates, rsi, 70, where=(rsi >= 70), color='red', alpha=0.3)
        ax3.fill_between(x_dates, rsi, 30, where=(rsi <= 30), color='green', alpha=0.3)
        
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.grid(color='#2f3640', linestyle='--', linewidth=0.5)
    
    # Formatting X-axis dates
    plt.xlabel('Date')
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save chart to the charts/ directory
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    charts_dir = os.path.join(script_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    filename = f"{symbol.replace('/', '_')}_overview.png"
    filepath = os.path.join(charts_dir, filename)
    
    # Save the figure
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Successfully saved overview chart to: charts/{filename}")

def plot_1h_chart(df: pd.DataFrame, last_n_candles: int = 200, symbol: str = "BTC/USDT"):
    """
    Generate a 4-panel 1H timeframe chart (Price, Volume, RSI, MACD).
    Saves the output as a `.png` file and displays it.
    """
    if df.empty:
        print("⚠️ No data to plot.")
        return
        
    df = df.tail(last_n_candles).copy()
        
    print(f"📊 Plotting 1H {symbol} chart...")
    
    # Enable dark background style
    plt.style.use('dark_background')
    
    # Create the 4-panel layout: Height ratios 10:4:3:3
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [10, 4, 3, 3]}, sharex=True)
    
    if 'timestamp' in df.columns:
        x_dates = pd.to_datetime(df['timestamp'])
    else:
        x_dates = pd.to_datetime(df.index)
        
    # Get date range for title
    date_start = x_dates.iloc[0].strftime('%Y-%m-%d')
    date_end = x_dates.iloc[-1].strftime('%Y-%m-%d')
    fig.suptitle(f"{symbol} 1H Overview ({date_start} to {date_end})", fontsize=16, fontweight='bold', color='white')

    up = df['close'] >= df['open']
    down = df['close'] < df['open']
    
    color_up = '#00ff00'
    color_down = '#ff0000'
    
    # ==========================
    # Panel 1: Price Action
    # ==========================
    ax1.vlines(x_dates[up], df['low'][up], df['high'][up], color=color_up, linewidth=1)
    ax1.vlines(x_dates[down], df['low'][down], df['high'][down], color=color_down, linewidth=1)
    
    ax1.vlines(x_dates[up], df['open'][up], df['close'][up], color=color_up, linewidth=4)
    ax1.vlines(x_dates[down], df['open'][down], df['close'][down], color=color_down, linewidth=4)
    
    if 'EMA_9' in df.columns:
        ax1.plot(x_dates, df['EMA_9'], color='dodgerblue', linewidth=1.2, label='EMA 9')
    if 'EMA_21' in df.columns:
        ax1.plot(x_dates, df['EMA_21'], color='darkorange', linewidth=1.2, label='EMA 21')
    if 'SMA_50' in df.columns:
        ax1.plot(x_dates, df['SMA_50'], color='gray', linewidth=1.5, linestyle='--', label='SMA 50')
        
    bbl_cols = [col for col in df.columns if col.startswith('BBL')]
    bbu_cols = [col for col in df.columns if col.startswith('BBU')]
    
    if bbl_cols and bbu_cols:
        bbl = df[bbl_cols[0]]
        bbu = df[bbu_cols[0]]
        ax1.plot(x_dates, bbu, color='lightblue', linewidth=0.5, alpha=0.5)
        ax1.plot(x_dates, bbl, color='lightblue', linewidth=0.5, alpha=0.5)
        ax1.fill_between(x_dates, bbl, bbu, color='mediumpurple', alpha=0.15, label='Bollinger Bands')
        
    ax1.set_ylabel('Price')
    ax1.grid(color='#2f3640', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left', frameon=True, fontsize='small', framealpha=0.6)
    
    # ==========================
    # Panel 2: Volume
    # ==========================
    ax2.bar(x_dates[up], df['volume'][up], color=color_up, alpha=0.8)
    ax2.bar(x_dates[down], df['volume'][down], color=color_down, alpha=0.8)
    
    if 'Volume_SMA_20' in df.columns:
        ax2.plot(x_dates, df['Volume_SMA_20'], color='yellow', linewidth=1.5, label='Vol SMA 20')
        ax2.legend(loc='upper left', fontsize='small')
    
    ax2.set_ylabel('Volume')
    ax2.grid(color='#2f3640', linestyle='--', linewidth=0.5)
    
    # ==========================
    # Panel 3: RSI
    # ==========================
    if 'RSI_14' in df.columns:
        ax3.plot(x_dates, df['RSI_14'], color='cyan', linewidth=1.5, label='RSI 14')
    if 'RSI_7' in df.columns:
        ax3.plot(x_dates, df['RSI_7'], color='magenta', linewidth=1.2, linestyle='--', label='RSI 7')
        
    ax3.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax3.fill_between(x_dates, 40, 60, color='gray', alpha=0.2, label='Neutral Zone')
    
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.grid(color='#2f3640', linestyle='--', linewidth=0.5)
    ax3.legend(loc='upper left', fontsize='small')
    
    # ==========================
    # Panel 4: MACD
    # ==========================
    macd_diff_cols = [c for c in df.columns if c.startswith('MACDh')]
    macd_line_cols = [c for c in df.columns if c.startswith('MACD_') and not c.startswith('MACDh')]
    macd_sig_cols = [c for c in df.columns if c.startswith('MACDs')]
    
    if macd_diff_cols and macd_line_cols and macd_sig_cols:
        macd_diff = df[macd_diff_cols[0]]
        macd_line = df[macd_line_cols[0]]
        macd_sig = df[macd_sig_cols[0]]
        
        macdup = macd_diff >= 0
        macddown = macd_diff < 0
        
        ax4.bar(x_dates[macdup], macd_diff[macdup], color='green', alpha=0.7)
        ax4.bar(x_dates[macddown], macd_diff[macddown], color='red', alpha=0.7)
        
        ax4.plot(x_dates, macd_line, color='white', linewidth=1.2, label='MACD Line')
        ax4.plot(x_dates, macd_sig, color='orange', linewidth=1.2, linestyle='--', label='Signal')
        
    ax4.set_ylabel('MACD')
    ax4.grid(color='#2f3640', linestyle='--', linewidth=0.5)
    ax4.legend(loc='upper left', fontsize='small')
    
    # Formatting X-axis dates
    plt.xlabel('Date')
    fig.autofmt_xdate()
    plt.tight_layout()
    
    # Save chart
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    charts_dir = os.path.join(script_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    filename = f"{symbol.replace('/', '_')}_1h_overview.png"
    filepath = os.path.join(charts_dir, filename)
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Successfully saved overview chart to: charts/{filename}")
    plt.show()  # Display it
    plt.close(fig)

if __name__ == "__main__":
    from indicators import add_indicators
    # Test chart generation if running directly
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(script_dir, "data", "BTC_USDT_1d.csv")
    
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file)
        # Add indicators 
        test_df = add_indicators(test_df)
        # Plot it
        plot_chart(test_df)
    else:
        print("⚠️ Test file not found!")

"""
backtest.py
The backtesting engine. Simulates the trading strategy on historical data
to evaluate performance, calculating metrics like win rate, drawdowns, and PnL.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_position_size(capital, entry_price, atr, risk_pct=0.02):
    """
    Calculates position size based on ATR dynamic stop loss.
    Never risks more than risk_pct of total capital.
    Max position size is capped at 30% of total capital.
    """
    # Safe fallback if ATR is somehow missing
    if pd.isna(atr) or atr <= 0:
        atr = entry_price * 0.05
        
    risk_amount = capital * risk_pct
    stop_distance = 1.5 * atr
    
    # How many units we can buy with that risk tolerance
    position_size = risk_amount / stop_distance
    
    # Cap position limit to 30% of capital
    max_position = (capital * 0.30) / entry_price
    position_size = min(position_size, max_position)
    
    stop_price = entry_price - stop_distance
    take_profit_price = entry_price + (2 * stop_distance)
    
    return position_size, stop_price, take_profit_price

def run_backtest(df: pd.DataFrame, initial_capital: float = 10000.0, fee: float = 0.001, risk_pct: float = 0.02):
    """
    Simulates trading based on the 'signal' column using ATR-based dynamic sizing.
    Calculates portfolio metrics, risk management, and plots the equity curve.
    """
    print(f"⏳ Running backtest (Dynamic ATR Sizing - Risk {risk_pct*100:.1f}%)...")
    
    if 'signal' not in df.columns:
        print("❌ 'signal' column missing. Please run the strategy logic first.")
        return df
        
    capital = initial_capital
    position = 0.0  # Asset holding
    
    equity_curve = []
    trades = []
    trade_cost = 0.0
    
    entry_price = 0.0
    entry_time = None
    
    current_stop_price = 0.0
    current_tp_price = 0.0
    
    stop_loss_count = 0
    take_profit_count = 0
    trade_durations = []
    
    position_sizes_pct = []
    rrs = []
    
    # Pre-parse timestamps to datetime to ensure .days calculations work smoothly
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    for i, row in df.iterrows():
        signal = row.get('signal', 0)
        close_price = row['close']
        current_time = row.get('timestamp', None)
        
        sold = False
        
        # Risk Management & Sell Logic
        if position > 0:
            reason = None
            if close_price <= current_stop_price:
                reason = "stop_loss"
            elif close_price >= current_tp_price:
                reason = "take_profit"
            elif signal == -1:
                reason = "signal_sell"
                
            if reason:
                # Sell entire position
                gross_return = position * close_price
                fee_amount = gross_return * fee
                net_return = gross_return - fee_amount
                
                capital += net_return
                
                profit_pct = (net_return - trade_cost) / trade_cost
                trades.append(profit_pct)
                
                if reason == "stop_loss":
                    stop_loss_count += 1
                elif reason == "take_profit":
                    take_profit_count += 1
                    
                if current_time is not None and entry_time is not None:
                    # In 1H data, .days might be 0 for quick trades, so calculate hours if possible
                    duration_hrs = (current_time - entry_time).total_seconds() / 3600
                    trade_durations.append(duration_hrs)
                
                position = 0.0
                sold = True
                
        # Buy Logic
        if signal == 1 and position == 0 and not sold:
            atr_val = row.get('ATR_14', 0)
            pos_size, current_stop_price, current_tp_price = calculate_position_size(capital, close_price, atr_val, risk_pct)
            
            invest_amount = pos_size * close_price
            fee_amount = invest_amount * fee
            
            trade_cost = invest_amount + fee_amount
            capital -= trade_cost
            position = pos_size
            
            entry_price = close_price
            entry_time = current_time
            
            # Record tracking metrics
            pos_pct = invest_amount / (capital + trade_cost) * 100
            position_sizes_pct.append(pos_pct)
            
            if (entry_price - current_stop_price) > 0:
                 rrs.append((current_tp_price - entry_price) / (entry_price - current_stop_price))
            else:
                 rrs.append(0)
                 
            print(f"📈 TRADE EXECUTED | Entry: {entry_price:.2f} | Stop: {current_stop_price:.2f} | TP: {current_tp_price:.2f}")
            
        current_equity = capital + (position * close_price)
        equity_curve.append(current_equity)

    df['equity'] = equity_curve
    
    # Buy and Hold baseline
    first_price = df.iloc[0]['close']
    bnh_asset = (initial_capital * (1 - fee)) / first_price
    df['bnh_equity'] = bnh_asset * df['close']
    
    # Calculate Metrics
    final_equity = df['equity'].iloc[-1]
    bnh_final_equity = df['bnh_equity'].iloc[-1]
    
    total_return_pct = ((final_equity / initial_capital) - 1) * 100
    bnh_return_pct = ((bnh_final_equity / initial_capital) - 1) * 100
    
    num_trades = len(trades)
    win_rate = (sum(1 for t in trades if t > 0) / num_trades * 100) if num_trades > 0 else 0.0
    avg_duration = np.mean(trade_durations) if trade_durations else 0.0
    
    avg_pos_pct = np.mean(position_sizes_pct) if position_sizes_pct else 0.0
    avg_rr = np.mean(rrs) if rrs else 0.0
    
    # Max Drawdown
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
    max_drawdown = df['drawdown'].min() * 100
    
    # Sharpe Ratio
    df['daily_return'] = df['equity'].pct_change()
    mean_return = df['daily_return'].mean()
    std_return = df['daily_return'].std()
    
    if pd.isna(std_return) or std_return == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (mean_return / std_return) * np.sqrt(365) # Approx scaled to daily equivalents for generic data
        
    print(f"\n--- Backtest Metrics ---")
    print(f"Total Return:       {total_return_pct:.2f}%")
    print(f"Buy & Hold Return:  {bnh_return_pct:.2f}%")
    print(f"Number of Trades:   {num_trades}")
    print(f"Win Rate:           {win_rate:.2f}%")
    print(f"Max Drawdown:       {max_drawdown:.2f}%")
    print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
    print(f"Average Risk/Reward ratio: {avg_rr:.2f}")
    print(f"Average position size as % of capital: {avg_pos_pct:.2f}%")
    print(f"Stop-Loss Hits:     {stop_loss_count}")
    print(f"Take-Profit Hits:   {take_profit_count}")
    print(f"Avg Trade Duration: {avg_duration:.1f} hours")
    print("------------------------\n")
    
    # Plotting
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    
    x_dates = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else df.index
    
    plt.plot(x_dates, df['equity'], label=f'Strategy Equity (Dynamic ATR)', color='#00ff00', linewidth=2)
    plt.plot(x_dates, df['bnh_equity'], label='Buy & Hold Equity', color='#00a8ff', linewidth=1.5, linestyle='--')
    
    plt.title(f'Backtest Result: Equity vs Buy & Hold', fontsize=14, fontweight='bold', color='white')
    plt.ylabel('Portfolio Value (USDT)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(color='#2f3640', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Save Plot
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    charts_dir = os.path.join(script_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    filename = "backtest_result_dynamic_atr.png"
    plot_path = os.path.join(charts_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved chart to: charts/{filename}")
    
    return df

def run_backtest_1h(df_1h: pd.DataFrame, df_4h: pd.DataFrame, model, initial_capital=10000.0, test_start_idx=None):
    """
    Comprehensive backtest for MultiTimeframeStrategy on 1H data.
    Simulates row by row, accounts for entry/exit slippage, ATR stops, and prints 
    a detailed markdown metrics table. Focuses on the last 20% of data (test set).
    """
    import sys, os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        
    from strategy import MultiTimeframeStrategy
    
    strategy = MultiTimeframeStrategy()
    
    if test_start_idx is None:
        test_size = int(len(df_1h) * 0.2)
        start_idx = len(df_1h) - test_size
    else:
        start_idx = test_start_idx
        test_size = len(df_1h) - start_idx
        
    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    entry_time = None
    
    current_stop_price = 0.0
    current_tp_price = 0.0
    
    fee_rate = 0.001       # 0.1% fee
    slippage_rate = 0.0005 # 0.05% slippage
    
    equity_curve = []
    trade_durations = []
    trades_pnl = []
    timestamps = []
    
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    if 'timestamp' in df_4h.columns:
        df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    
    print(f"⏳ Starting 1H Comprehensive Backtest on {test_size} test candles...")
    
    print_step = test_size // 10 if test_size >= 10 else 1
    
    for count, i in enumerate(range(start_idx, len(df_1h))):
        if count % print_step == 0:
            print(f"   Progress: {count}/{test_size} candles simulated...")
            
        current_time = df_1h.iloc[i]['timestamp']
        timestamps.append(current_time)
        close_price = df_1h.iloc[i]['close']
        
        history_start = max(0, i - 168)
        df_1h_slice = df_1h.iloc[history_start:i+1].copy()
        
        if 'timestamp' in df_4h.columns:
             df_4h_slice = df_4h[df_4h['timestamp'] <= current_time].copy()
        else:
             idx_4h = int(i / 4)
             df_4h_slice = df_4h.iloc[:idx_4h+1].copy()
             
        if len(df_4h_slice) == 0:
             equity_curve.append(capital + position * close_price)
             continue
             
        sold = False
        
        if position > 0:
            reason = None
            exit_price = 0.0
            
            if close_price <= current_stop_price:
                reason = "Stop Loss"
                exit_price = current_stop_price * (1 - slippage_rate)
            elif close_price >= current_tp_price:
                reason = "Take Profit"
                exit_price = current_tp_price * (1 - slippage_rate)
            
            if reason:
                gross_return = position * exit_price
                fee_amount = gross_return * fee_rate
                net_return = gross_return - fee_amount
                
                entry_cost = position * entry_price
                trade_pnl = net_return - entry_cost
                trades_pnl.append(trade_pnl)
                
                capital += net_return
                
                if entry_time is not None:
                    duration_hrs = (pd.to_datetime(current_time) - pd.to_datetime(entry_time)).total_seconds() / 3600
                    trade_durations.append(duration_hrs)
                
                position = 0.0
                sold = True
                
        if position == 0 and not sold:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                signal, conf, trend, score = strategy.generate_signals(df_1h_slice, df_4h_slice, model)
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
            
            if signal == 1:
                atr_val = df_1h_slice.iloc[-1].get('ATR_14', close_price * 0.05)
                
                actual_entry = close_price * (1 + slippage_rate)
                pos_size, current_stop_price, current_tp_price = calculate_position_size(capital, actual_entry, atr_val, risk_pct=0.02)
                
                invest_amount = pos_size * actual_entry
                fee_amount = invest_amount * fee_rate
                trade_cost = invest_amount + fee_amount
                
                if capital >= trade_cost:
                    capital -= trade_cost
                    position = pos_size
                    entry_price = actual_entry
                    entry_time = current_time
                    
        current_equity = capital + (position * close_price)
        equity_curve.append(current_equity)

    print("🏁 Backtest complete. Calculating metrics...\n")
    
    res = pd.DataFrame({'timestamp': timestamps, 'equity': equity_curve, 'close': df_1h.iloc[start_idx:]['close'].values})
    
    first_price = res.iloc[0]['close']
    bnh_asset = (initial_capital * (1 - (fee_rate + slippage_rate))) / first_price
    res['bnh_equity'] = bnh_asset * res['close']
    
    final_eq = res['equity'].iloc[-1]
    bnh_eq = res['bnh_equity'].iloc[-1]
    
    ret_strat = ((final_eq / initial_capital) - 1) * 100
    ret_bnh = ((bnh_eq / initial_capital) - 1) * 100
    
    days_passed = (res['timestamp'].iloc[-1] - res['timestamp'].iloc[0]).days
    days_passed = max(1, days_passed)
    ann_strat = ((final_eq / initial_capital) ** (365 / days_passed) - 1) * 100 if final_eq > 0 else -100
    ann_bnh = ((bnh_eq / initial_capital) ** (365 / days_passed) - 1) * 100 if bnh_eq > 0 else -100
    
    def get_max_dd(s):
        peak = s.cummax()
        return ((s - peak) / peak).min() * 100
        
    dd_strat = get_max_dd(res['equity'])
    dd_bnh = get_max_dd(res['bnh_equity'])
    
    def get_sharpe(s):
        dr = s.pct_change().dropna()
        if dr.std() == 0: return 0.0
        return (dr.mean() / dr.std()) * np.sqrt(365 * 24)
        
    sharpe_strat = get_sharpe(res['equity'])
    sharpe_bnh = get_sharpe(res['bnh_equity'])
    
    wins = [x for x in trades_pnl if x > 0]
    losses = [abs(x) for x in trades_pnl if x <= 0]
    num_trades = len(trades_pnl)
    win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0.0
    pf = (sum(wins) / sum(losses)) if sum(losses) > 0 else (99.9 if sum(wins) > 0 else 0.0)
    avg_dur = np.mean(trade_durations) if trade_durations else 0.0
    
    print("="*50)
    print("         1H COMPREHENSIVE BACKTEST RESULTS         ")
    print("="*50)
    print(f"| Metric              | ML Strategy | Buy & Hold |")
    print(f"|---------------------|-------------|------------|")
    print(f"| Total Return        | {ret_strat:>10.2f}% | {ret_bnh:>9.2f}% |")
    print(f"| Annualized Return   | {ann_strat:>10.2f}% | {ann_bnh:>9.2f}% |")
    print(f"| Max Drawdown        | {dd_strat:>10.2f}% | {dd_bnh:>9.2f}% |")
    print(f"| Sharpe Ratio        | {sharpe_strat:>11.2f} | {sharpe_bnh:>10.2f} |")
    print(f"| Win Rate            | {win_rate:>10.2f}% |        --- |")
    print(f"| Avg Trade Duration  | {avg_dur:>7.1f} hrs |        --- |")
    print(f"| Total Trades        | {num_trades:>11d} |        --- |")
    print(f"| Profit Factor       | {pf:>11.2f} |        --- |")
    print("="*50 + "\n")
    
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    plt.plot(res['timestamp'], res['equity'], label='ML Strategy (1H Dynamic ATR)', color='#00ff00', linewidth=2)
    plt.plot(res['timestamp'], res['bnh_equity'], label='Buy & Hold', color='#00a8ff', linewidth=1.5, linestyle='--')
    plt.title('MultiTimeframe Strategy 1H Backtest', fontsize=14, fontweight='bold', color='white')
    plt.ylabel('Portfolio Value (USDT)')
    plt.legend()
    plt.grid(color='#2f3640', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    charts_dir = os.path.join(script_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    plot_path = os.path.join(charts_dir, "backtest_1h.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved chart to: {plot_path}")
    
    metrics = {
        'return': ret_strat,
        'sharpe': sharpe_strat,
        'max_dd': dd_strat,
        'win_rate': win_rate
    }
    return res, metrics

def walkforward_test(df_1h, df_4h, n_splits=5):
    """
    Run walk-forward validation by partitioning the data into expanding periods.
    """
    import sys, os
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        
    try:
        from ml_features import prepare_features_1h
    except ImportError:
        print("⚠️ Failed to import prepare_features_1h for Walk-Forward Test.")
        return
        
    # We silently extract features over entire history to get X, y indices
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        df_feat, X, y = prepare_features_1h(df_1h.copy())
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
        
    if len(X) < n_splits * 10:
        print(f"⚠️ Not enough data points ({len(X)}) to do {n_splits} split walk-forward. Need more history!")
        return
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    print(f"\n🚀 Starting Walk-Forward Validation ({n_splits} Windows)...")
    print("| Period | Return   | Sharpe | MaxDD   | WinRate |")
    print("|--------|----------|--------|---------|---------|")
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        period = i + 1
        
        # 1. Retrain the ML model on training portion
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        
        # We spawn a standard lightweight RF for consistency
        model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_split=20, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Calculate exactly which rows in df_1h map to test_idx to pass to run_backtest_1h
        actual_test_idx_in_df = X.index[test_idx]
        test_start_pos = df_1h.index.get_loc(actual_test_idx_in_df[0])
        test_end_pos = df_1h.index.get_loc(actual_test_idx_in_df[-1])
        
        df_1h_slice = df_1h.iloc[:test_end_pos+1].copy()
        
        # Silence standard backtest prints so only the final table prints neatly
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
             res, metrics = run_backtest_1h(df_1h_slice, df_4h, model, test_start_idx=test_start_pos)
        finally:
             sys.stdout.close()
             sys.stdout = old_stdout
             
        ret = metrics['return']
        sharpe = metrics['sharpe']
        max_dd = metrics['max_dd']
        win_rate = metrics['win_rate']
        
        print(f"| {period:<6} | {ret:>7.2f}% | {sharpe:>6.2f} | {max_dd:>6.2f}% | {win_rate:>6.2f}% |")
        
        results.append(metrics)
        
    avg_ret = np.mean([r['return'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    avg_max_dd = np.mean([r['max_dd'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results])
    
    print("|--------|----------|--------|---------|---------|")
    print(f"| Average| {avg_ret:>7.2f}% | {avg_sharpe:>6.2f} | {avg_max_dd:>6.2f}% | {avg_win_rate:>6.2f}% |")
    print("")
    
    positive_periods = sum(1 for r in results if r['return'] > 0)
    positivity_rate = positive_periods / n_splits
    
    if avg_sharpe > 0.5 and positivity_rate > 0.6:
        print("✅ Strategy is ROBUST")
    else:
        print("⚠️ WARNING: Possible overfitting")

def stress_test(model, initial_capital=10000.0):
    """
    Downloads historical data for severely difficult market regimes
    and runs full backtest on each to print a comparative survivability table.
    """
    import ccxt
    from datetime import datetime, timezone
    import time
    import sys, os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        
    try:
        from indicators import add_indicators_1h
    except ImportError:
        print("⚠️ Could not import add_indicators_1h for stress testing.")
        return
        
    def fetch_period(start_date, end_date):
        exchange = ccxt.binance({'enableRateLimit': True})
        since = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        all_ohlcv = []
        while since < end_ts:
            try:
                ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', since, limit=1000)
                if not ohlcv:
                    break
                filtered = [x for x in ohlcv if x[0] <= end_ts]
                if not filtered:
                    break
                all_ohlcv.extend(filtered)
                since = filtered[-1][0] + 3600000  # advance by 1 hour
                time.sleep(0.1) # Respect Binance rate limit for unauthenticated
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        if not all_ohlcv:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    periods = [
        {"name": "Bull run", "start": "2021-01-01", "end": "2021-04-30"},
        {"name": "Crash", "start": "2021-05-01", "end": "2021-05-31"},
        {"name": "Bear market", "start": "2021-11-01", "end": "2022-06-30"},
        {"name": "Recovery", "start": "2023-01-01", "end": "2023-12-31"}
    ]
    
    print("\n" + "🔥"*20)
    print("      STRATEGY STRESS TEST      ")
    print("🔥"*20 + "\n")
    
    performs_well = []
    struggles = []
    
    for p in periods:
        print(f"Downloading data for {p['name']} ({p['start']} to {p['end']})...")
        df_1h = fetch_period(p['start'], p['end'])
        
        if len(df_1h) < 200:
            print(f"⚠️ Not enough data fetched for {p['name']}.")
            continue
            
        # Mute indicator calculation prints
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            df_1h = add_indicators_1h(df_1h)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            
        # Create 4H timeframe natively via pandas resample
        df_1h_temp = df_1h.copy()
        df_1h_temp.set_index('timestamp', inplace=True)
        # Using '4h' correctly for modern pandas versions
        df_4h = df_1h_temp.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        
        print(f"Running backtest simulation...")
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            # Offset start_idx to 168 to give time for 7-day ATR averages
            res, metrics = run_backtest_1h(df_1h, df_4h, model, initial_capital=initial_capital, test_start_idx=168)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            
        bnh_return = ((res['bnh_equity'].iloc[-1] / initial_capital) - 1) * 100
        bot_return = metrics['return']
        max_dd = metrics['max_dd']
        
        # Outperformance heuristic
        outperformed = "YES" if bot_return > bnh_return else "NO"
        
        # Logic to decide if it performed "well" or "struggled"
        if outperformed == "YES" and bot_return > -15.0:
            performs_well.append(p['name'])
        else:
            struggles.append(p['name'])
            
        print(f"--- {p['name']} ({p['start']} to {p['end']}) ---")
        print(f"Market Return: {bnh_return:.2f}%")
        print(f"Bot Return:    {bot_return:.2f}%")
        print(f"Max Drawdown:  {max_dd:.2f}%")
        print(f"Outperformed?: {outperformed}\n")
        
    print("="*40)
    print("FINAL VERDICT:")
    print(f"Bot performs well in: {', '.join(performs_well) if performs_well else 'None'}")
    print(f"Bot struggles in: {', '.join(struggles) if struggles else 'None'}")
    
    if len(struggles) >= 2:
        print("Recommendation: reduce position size and adjust stop-loss to preserve capital during extreme trends")
    else:
        print("Recommendation: Strategy is remarkably robust. Collect more data to keep algorithms aligned with modern liquidity regimes.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(script_dir, "data", "BTC_USDT_1d.csv")
    
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        
        if 'SMA_20' not in df.columns:
            df['SMA_20'] = df['close'].rolling(20).mean()
        if 'SMA_50' not in df.columns:
            df['SMA_50'] = df['close'].rolling(50).mean()
            
        if 'signal' not in df.columns:
            from strategy import MACrossoverStrategy
            strategy = MACrossoverStrategy()
            df = strategy.generate_signals(df)
            
        print("\n=== DYNAMIC ATR SIZING ===")
        df_rm = run_backtest(df.copy(), risk_pct=0.02)
    else:
        print("⚠️ Test file not found. Please run fetcher.py first!")

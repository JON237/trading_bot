"""
bot.py
The main entry point for the live trading loop.
Orchestrates downloading data, applying indicators, evaluating the strategy,
executing orders, and triggering telemetry notifications.
"""

import time
import logging
import os
import pandas as pd
from datetime import datetime

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def log_trade(action: str, price: float, pnl: float = 0.0, reason: str = ""):
    """Log trading actions to a CSV file."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, "trades.csv")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("timestamp,action,price,pnl,reason\n")
            
    with open(log_file, "a") as f:
        f.write(f"{timestamp},{action},{price:.2f},{pnl:.2f},{reason}\n")

def run_paper_trading(interval_minutes: int = 60, stop_loss_pct: float = 0.03, take_profit_pct: float = 0.06):
    """
    Main paper trading logic loop.
    Runs continuously, downloading data on the configured interval to track ML predictions.
    """
    logging.info(f"Starting paper trading bot (Interval: {interval_minutes}m)...")
    
    try:
        from fetcher import DataFetcher
        from indicators import add_indicators
        from strategy import MLStrategy
        from notifier import send_telegram
    except ImportError as e:
        logging.error(f"Missing modules. Ensure you run this from the project src directory: {e}")
        return

    # 1. Test Notification immediately at standard boot block!
    startup_msg = "Trading bot started. Paper trading mode. Strategy: ML"
    send_telegram(startup_msg)
    
    fetcher = DataFetcher(exchange_id='binance')
    strategy = MLStrategy()
    
    position = "NONE"
    entry_price = 0.0
    
    # 2. Advanced Daily Trackers
    last_day = datetime.now().date()
    daily_pnl_pct = 0.0
    trades_today = 0
    
    timeframe = "1h" if interval_minutes == 60 else f"{interval_minutes}m"
    if interval_minutes >= 1440:
        timeframe = "1d"
        
    try:
        while True:
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            current_date = now.date()
            
            # Check for midnight crossover to send daily summary
            if current_date > last_day:
                open_pos_str = "YES" if position == "LONG" else "NO"
                pnl_sign = "+" if daily_pnl_pct > 0 else ""
                summary = f"Daily PnL: {pnl_sign}{daily_pnl_pct:.2f}% | Trades today: {trades_today} | Open position: {open_pos_str}"
                send_telegram(summary)
                
                # Reset trackers
                last_day = current_date
                daily_pnl_pct = 0.0
                trades_today = 0
            
            try:
                # 1. Fetch latest data (fetching sufficient trailing bars so indicators can spin up)
                df = fetcher.fetch_ohlcv("BTC/USDT", timeframe=timeframe, since_days=5)
                
                if df.empty:
                    logging.warning("Datafeed empty. Retrying next cycle...")
                    time.sleep(60)
                    continue
                    
                # Explicitly slice the last 100 candles strictly per requirements before running indicator algorithms
                df = df.tail(100).reset_index(drop=True)
                
                # 2. Calculate Indicators
                import warnings
                warnings.filterwarnings('ignore')
                df = add_indicators(df)
                
                # 3. Generate ML signal via rf_model.pkl
                df = strategy.generate_signals(df)
                
                if df.empty:
                    logging.warning("Indicator requirements trimmed dataframe perfectly flat. Waiting...")
                    time.sleep(60)
                    continue
                    
                # Grab extreme bottom evaluation row
                latest_row = df.iloc[-1]
                current_price = latest_row['close']
                signal = latest_row.get('signal', 0)
                confidence = latest_row.get('confidence', 0.0) * 100
                
                pnl_pct = 0.0
                sold = False
                reason = ""
                
                # 5 & 6. Sell logic / Risk Management evaluating active position status
                if position == "LONG":
                    change_pct = (current_price - entry_price) / entry_price
                    pnl_pct = change_pct * 100
                    
                    if change_pct <= -stop_loss_pct:
                        reason = "stop-loss"
                    elif change_pct >= take_profit_pct:
                        reason = "take-profit"
                    elif signal == -1:
                        reason = "signal"
                        
                    if reason:
                        pnl_sign = "+" if pnl_pct > 0 else ""
                        msg = f"SELL BTC/USDT at ${current_price:.2f} — PnL: {pnl_sign}{pnl_pct:.2f}% — Reason: {reason}"
                        send_telegram(msg)
                        
                        log_trade("SELL", current_price, pnl_pct, reason)
                        position = "NONE"
                        entry_price = 0.0
                        sold = True
                        
                        trades_today += 1
                        daily_pnl_pct += pnl_pct
                
                # 4. Long Configuration Trigger
                if signal == 1 and position == "NONE" and not sold:
                    position = "LONG"
                    entry_price = current_price
                    
                    msg = f"BUY BTC/USDT at ${current_price:.2f} — confidence: {confidence:.0f}%"
                    send_telegram(msg)
                    
                    log_trade("BUY", current_price, 0.0, "signal_buy")
                
                # 8. Print terminal status interface
                print(f"[{current_time}] Position: {position} | Entry: ${entry_price:.2f} | Current: ${current_price:.2f} | PnL: {pnl_pct:.2f}%")
                
            except Exception as e:
                logging.error(f"Error evaluating market pipeline matrices: {e}")
                
            # Run continuously, blocking execution until next required tick
            time.sleep(interval_minutes * 60)
            
    except KeyboardInterrupt:
        # Halt execution gracefully
        print("\nBot stopped cleanly")

def run_paper_trading_15m(max_hours=None):
    """
    15M Paper trading loop. Waits for exactly the next 15-minute mark, fetches data,
    evaluates MTF strategy, applies ATR stops natively via backtest module,
    and logs actions without crashing.
    """
    import sys, os
    import traceback
    import joblib
    from datetime import timedelta
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    try:
        from fetcher import DataFetcher
        from indicators import add_indicators_1h
        from strategy import MultiTimeframeStrategy, RuleBasedStrategy, BollingerBounceStrategy
        from notifier import send_telegram, send_trade_open_alert, send_trade_closed_alert, send_daily_summary, send_risk_alert
        from backtest import calculate_position_size
        from executor import BinanceExecutor
    except ImportError as e:
        logging.error(f"Missing modules. Ensure running from src directory: {e}")
        return
        
    # Attempt to load config flag
    try:
        from config import USE_ML_MODEL
    except ImportError:
        USE_ML_MODEL = True

    model = None
    if USE_ML_MODEL:
        model_path = os.path.join(os.path.dirname(script_dir), "models", "best_model_1h.pkl")
        try:
            model = joblib.load(model_path)
            print(f"1H Trading Bot started. Model: best_model_1h.pkl. Strategy: MultiTimeframe MTF")
            strategy = MultiTimeframeStrategy()
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return
    else:
        print(f"1H Trading Bot started. Strategy: BollingerBounceStrategy (ML Disabled)")
        strategy = BollingerBounceStrategy()
        
    try:
        send_telegram("Bot started in paper trading mode")
    except:
        pass
        
    fetcher = DataFetcher(exchange_id='binance')
    live_executor = BinanceExecutor(fetcher.exchange)
    
    initial_portfolio = 10000.0
    portfolio = initial_portfolio
    position_size = 0.0
    position = "NONE"
    entry_price = 0.0
    entry_time = None
    stop_price = 0.0
    tp_price = 0.0
    
    hours_run = 0
    trades_executed = 0
    
    last_summary_date = datetime.utcnow().date()
    daily_trades = 0
    daily_wins = 0
    daily_losses = 0
    daily_pnl_usd = 0.0
    daily_best_pct = -100.0
    daily_worst_pct = 100.0
    
    def log_trade_1h(action, price, pnl, reason):
        logs_dir = os.path.join(os.path.dirname(script_dir), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, "trades_1h.csv")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("timestamp,action,price,pnl,reason\n")
        with open(log_file, "a") as f:
            f.write(f"{ts},{action},{price:.2f},{pnl:.2f},{reason}\n")
            
    while True:
        try:
            now = datetime.now()
            
            # Calculate next 15-minute mark
            minutes_to_add = 15 - (now.minute % 15)
            next_run = (now + timedelta(minutes=minutes_to_add)).replace(second=0, microsecond=0)
            sleep_secs = (next_run - now).total_seconds()
            
            now_utc = datetime.utcnow()
            if now_utc.hour == 20 and now_utc.date() > last_summary_date:
                total_pnl_usd = portfolio - initial_portfolio
                total_pnl_pct = (total_pnl_usd / initial_portfolio) * 100
                day_pnl_pct = (daily_pnl_usd / portfolio) * 100 if portfolio > 0 else 0.0
                
                send_daily_summary(
                    daily_trades, daily_wins, daily_losses,
                    daily_pnl_usd, day_pnl_pct, total_pnl_usd, total_pnl_pct,
                    daily_best_pct if daily_best_pct != -100.0 else 0.0,
                    daily_worst_pct if daily_worst_pct != 100.0 else 0.0,
                    position
                )
                
                last_summary_date = now_utc.date()
                daily_trades = 0
                daily_wins = 0
                daily_losses = 0
                daily_pnl_usd = 0.0
                daily_best_pct = -100.0
                daily_worst_pct = 100.0
            
            now_str = datetime.now().strftime("%H:%M")
            
            if not getattr(live_executor, 'first_run_done', False):
                live_executor.first_run_done = True
                print(f"[{now_str}] Cold Boot: Running immediate evaluation...")
            else:
                # Wait until next 15m mark
                print(f"[{now_str}] Sleeping for {sleep_secs} seconds until next 15-minute candle...")
                time.sleep(sleep_secs + 2) # small buffer
            
            df_15m = fetcher.fetch_ohlcv("BTC/USDT", timeframe="15m", since_days=25)
            if df_15m.empty:
                continue
            df_15m = df_15m.tail(2160).reset_index(drop=True)
            
            df_4h = fetcher.fetch_ohlcv("BTC/USDT", timeframe="4h", since_days=30)
            if df_4h.empty:
                continue
            df_4h = df_4h.tail(150).reset_index(drop=True)
            
            try:
                df_15m = add_indicators_1h(df_15m)
            except Exception as e:
                print(f"Error adding indicators: {e}")
                continue
                
            current_data = df_15m.iloc[:-1]
            last_candle_15m = df_15m.iloc[-1]
            
            if USE_ML_MODEL:
                signal, conf, score, reason_or_trend = strategy.generate_signals(current_data, df_4h)
            else:
                signal, conf, score, reason_or_trend = strategy.generate_signals(current_data)
                
            exec_price = float(last_candle_15m['close'])
            atr_val = float(current_data['ATR_14'].iloc[-1]) if 'ATR_14' in current_data.columns else exec_price * 0.02
            
            pnl_pct = 0.0
            if position == "LONG":
                pnl_pct = ((exec_price - entry_price) / entry_price) * 100
                
            sold = False
            
            # Check active position
            if position == "LONG":
                reason = None
                if not USE_ML_MODEL and signal == -1:
                    reason = reason_or_trend
                else:
                    if exec_price <= stop_price:
                        reason = "Stop Loss"
                    elif exec_price >= tp_price:
                        reason = "Take Profit"
                    elif signal == -1:
                        reason = "Signal Sell"
                    
                if reason:
                    # Execute sell & cancel defensive OCO
                    live_executor.cancel_all_orders()
                    live_executor.execute_sell(position_size)
                    
                    net_return = position_size * exec_price * 0.999 # 0.1% exchange fee
                    pnl_usd = net_return - (position_size * entry_price)
                    
                    portfolio += net_return
                    sold = True
                    position = "NONE"
                    log_trade_1h("SELL", latest_close, pnl_pct, reason)
                    
                    daily_trades += 1
                    daily_pnl_usd += pnl_usd
                    if pnl_pct > 0: daily_wins += 1
                    else: daily_losses += 1
                    if pnl_pct > daily_best_pct: daily_best_pct = pnl_pct
                    if pnl_pct < daily_worst_pct: daily_worst_pct = pnl_pct
                    
                    duration_hrs = (datetime.now() - entry_time).total_seconds() / 3600 if entry_time else 1.0
                    port_pct_total = ((portfolio - initial_portfolio) / initial_portfolio) * 100
                    
                    if reason == "Stop Loss":
                        try: send_risk_alert(pnl_usd, pnl_pct, "Price dropped below ATR stop")
                        except: pass
                        
                    try:
                        send_trade_closed_alert(
                            reason, entry_price, latest_close, pnl_usd, pnl_pct,
                            duration_hrs, portfolio, port_pct_total
                        )
                    except: pass
                    
                    entry_price = 0.0
                    
            # Check for new signals
            if position == "NONE" and signal == 1 and not sold:
                buy_res = live_executor.buy_market(latest_close)
                
                if buy_res:
                    # Fetch executed metrics (fallback to simulation)
                    pos_size = float(buy_res.get('filled', TRADE_SIZE_USDT / latest_close))
                    exec_price = float(buy_res.get('price', latest_close))
                    
                    # Calculate defensive logic based on strategy 
                    if not USE_ML_MODEL:
                        new_stop = exec_price * 0.98
                        new_tp = float(df_1h.iloc[-1].get('BBU_20_2.0', exec_price * 1.05))
                    else:
                         _, new_stop, new_tp = calculate_position_size(portfolio, exec_price, atr_val, risk_pct=0.02)
                    
                    live_executor.place_oco_sell(pos_size, exec_price, new_tp, new_stop)
                    
                    invest = pos_size * exec_price * 1.001 # approx fee
                    if portfolio >= invest:
                        portfolio -= invest
                        position = "LONG"
                        position_size = pos_size
                        entry_price = exec_price
                        entry_time = datetime.now()
                        stop_price = new_stop
                        tp_price = new_tp
                        log_trade_1h("BUY", exec_price, 0.0, "Signal Buy")
                        trades_executed += 1
                        
                        try:
                            send_trade_open_alert(exec_price, pos_size, invest, new_stop, new_tp, conf * 100, score, reason_or_trend)
                        except: pass
                    
            sig_str = "BUY" if signal == 1 else ("SELL" if signal == -1 else "HOLD")
            pos_str = f"LONG ${entry_price:,.0f}" if position == "LONG" else "NONE"
            pnl_str = f"+{pnl_pct:.1f}%" if pnl_pct > 0 else f"{pnl_pct:.1f}%"
            
            print(f"[{now_str}] Signal: {sig_str} | Confidence: {conf*100:.0f}% | Position: {pos_str} | PnL: {pnl_str} | Portfolio: ${portfolio:,.0f}")
            
            hours_run += 1
            if max_hours and hours_run >= max_hours:
                print("\n=== Session Summary ===")
                print(f"Hours Run: {hours_run}")
                print(f"Trades Executed: {trades_executed}")
                print(f"Final Portfolio: ${portfolio:,.0f}")
                break

                
        except KeyboardInterrupt:
            print("\nBot stopped cleanly via KeyboardInterrupt")
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    run_paper_trading_15m()

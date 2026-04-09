"""
strategy.py
Contains the trading logic and rules for generating buy/sell signals.
"""

import os
import joblib
import pandas as pd
import numpy as np

class MACrossoverStrategy:
    def __init__(self, fast_ma: str = 'SMA_20', slow_ma: str = 'SMA_50'):
        """Initialize the Moving Average Crossover strategy."""
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy (1), sell (-1), or hold (0) signals based on MA crossovers.
        Adds a 'signal' column to the DataFrame.
        """
        if self.fast_ma not in df.columns or self.slow_ma not in df.columns:
            print(f"⚠️ Missing columns {self.fast_ma} or {self.slow_ma}.")
            df['signal'] = 0
            return df
            
        diff = df[self.fast_ma] - df[self.slow_ma]
        prev_diff = diff.shift(1)
        
        buy_condition = (prev_diff <= 0) & (diff > 0)
        sell_condition = (prev_diff >= 0) & (diff < 0)
        
        df['signal'] = 0
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df


class MLStrategy:
    def __init__(self, model_path: str = "models/rf_model.pkl", confidence_threshold: float = 0.60):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
    def generate_signals(self, df: pd.DataFrame, model=None, confidence_threshold: float = None) -> pd.DataFrame:
        """
        Generate signals using the trained RandomForest model probabilities.
        """
        df['signal'] = 0
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        # Ensure all standard ML features are derived for inference dynamically
        df['ret_1d'] = df['close'].pct_change(1)
        df['ret_3d'] = df['close'].pct_change(3)
        df['ret_5d'] = df['close'].pct_change(5)
        df['ret_10d'] = df['close'].pct_change(10)
        
        rsi_cols = [c for c in df.columns if c.startswith('RSI')]
        if rsi_cols: df['feature_rsi'] = df[rsi_cols[0]]
             
        macd_hist_cols = [c for c in df.columns if c.startswith('MACDh')]
        if macd_hist_cols: df['feature_macd_hist'] = df[macd_hist_cols[0]]
             
        if 'SMA_20' in df.columns:
             df['dist_sma20'] = (df['close'] - df['SMA_20']) / df['SMA_20']
             
        if 'Volume_SMA_20' in df.columns:
             df['vol_ratio'] = df['volume'] / df['Volume_SMA_20']
             
        bbl_cols = [c for c in df.columns if c.startswith('BBL')]
        bbu_cols = [c for c in df.columns if c.startswith('BBU')]
        if bbl_cols and bbu_cols:
             lower_band = df[bbl_cols[0]]
             upper_band = df[bbu_cols[0]]
             band_width = upper_band - lower_band
             df['bb_position'] = np.where(band_width == 0, 0.5, (df['close'] - lower_band) / band_width)
        
        # Load Model
        if model is None:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_file = os.path.join(script_dir, self.model_path)
            if not os.path.exists(model_file):
                print(f"⚠️ Model file not found at {model_file}")
                return df
            try:
                clf = joblib.load(model_file)
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                return df
        else:
            clf = model
            
        # Dynamically Extract expected features from what the model was trained on
        if hasattr(clf, 'feature_names_in_'):
            features = list(clf.feature_names_in_)
        else:
            print("⚠️ Model does not provide feature_names_in_ attribute.")
            return df
            
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"⚠️ Missing required features from dataframe: {missing_features}.")
            return df
            
        # Predict on valid non-NaN slices
        valid_idx = df[features].dropna().index
        if len(valid_idx) == 0:
            print("⚠️ All data rows contain NaN for the required ML features.")
            return df
            
        X = df.loc[valid_idx, features]
        probs = clf.predict_proba(X)
        
        # Binary Classification mapping assumes Class 0=Down, 1=Up
        prob_down = probs[:, 0]
        prob_up = probs[:, 1]
        
        signals = np.zeros(len(valid_idx))
        signals[prob_up > threshold] = 1        # Buy Signal
        signals[prob_down > threshold] = -1     # Sell Signal
        
        df.loc[valid_idx, 'signal'] = signals
        df.loc[valid_idx, 'confidence'] = np.maximum(prob_up, prob_down)
        return df

class MultiTimeframeStrategy:
    def __init__(self):
        pass

    def is_good_trading_hour(self, timestamp_utc) -> bool:
        """
        Active trading sessions (UTC times when crypto volume is highest):
        - Asian session: 00:00 - 08:00 UTC
        - London open: 07:00 - 10:00 UTC  
        - US session: 13:00 - 21:00 UTC
        - Avoid: 22:00 - 23:59 UTC (low volume, high spread)
        """
        if pd.isna(timestamp_utc) or timestamp_utc is None:
            return True
            
        try:
            hour = pd.to_datetime(timestamp_utc).hour
        except Exception:
            return True
            
        if 22 <= hour <= 23:
            return False
            
        return True

    def generate_signals(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame, model, confidence_threshold: float = 0.62, position="NONE", entry_price=0.0):
        """
        Generates trading signals by combining 4H trend direction with 1H ML predictions.
        Focuses only on the latest available candle for live decisions.
        """
        latest_1h_row = df_1h.iloc[-1]
        latest_timestamp = latest_1h_row.get('timestamp', "Unknown Time")
        
        # --- Filter 1: Trading Session ---
        if not self.is_good_trading_hour(latest_timestamp):
            print(f"Signal filtered: BAD_TRADING_SESSION (22:00-23:59 UTC) at {latest_timestamp}")
            return 0, 0.0, "NEUTRAL", 0
            
        # --- Filter 2: Volatility (ATR) ---
        if 'ATR_14' in df_1h.columns:
            lookback = min(168, len(df_1h)) # 168 hours = 7 days
            avg_atr = df_1h['ATR_14'].iloc[-lookback:].mean()
            current_atr = latest_1h_row['ATR_14']
            
            if pd.notna(avg_atr) and pd.notna(current_atr) and avg_atr > 0:
                if current_atr < 0.5 * avg_atr:
                    print(f"Signal filtered: LOW_VOLATILITY (ATR {current_atr:.2f} < 0.5 * avg_atr {avg_atr:.2f}) at {latest_timestamp}")
                    return 0, 0.0, "NEUTRAL", 0
                if current_atr > 3.0 * avg_atr:
                    print(f"Signal filtered: EXTREME_VOLATILITY (ATR {current_atr:.2f} > 3.0 * avg_atr {avg_atr:.2f}) at {latest_timestamp}")
                    return 0, 0.0, "NEUTRAL", 0

        # Step 1: Get 4H Trend Direction
        if 'SMA_20' not in df_4h.columns:
            df_4h['SMA_20'] = df_4h['close'].rolling(20).mean()
        if 'SMA_50' not in df_4h.columns:
            df_4h['SMA_50'] = df_4h['close'].rolling(50).mean()
            
        latest_4h = df_4h.iloc[-1]
        trend_bullish = latest_4h['SMA_20'] > latest_4h['SMA_50']
        trend_bearish = latest_4h['SMA_20'] < latest_4h['SMA_50']
        
        trend_str = "BULLISH" if trend_bullish else ("BEARISH" if trend_bearish else "NEUTRAL")
        
        # Step 2: Extract ML Features natively for inference
        df_1h = df_1h.copy()
        df_1h['price_change_1h'] = df_1h['close'].pct_change(1)
        df_1h['price_change_3h'] = df_1h['close'].pct_change(3)
        df_1h['price_change_6h'] = df_1h['close'].pct_change(6)
        df_1h['price_change_24h'] = df_1h['close'].pct_change(24)
        
        df_1h['rsi_14'] = df_1h['RSI_14'] if 'RSI_14' in df_1h.columns else np.nan
        df_1h['rsi_7'] = df_1h['RSI_7'] if 'RSI_7' in df_1h.columns else np.nan
        
        macd_cols = [c for c in df_1h.columns if c.startswith('MACDh')]
        df_1h['macd_hist'] = df_1h[macd_cols[0]] if macd_cols else np.nan
        
        bbl_cols = [c for c in df_1h.columns if c.startswith('BBL')]
        bbu_cols = [c for c in df_1h.columns if c.startswith('BBU')]
        if bbl_cols and bbu_cols:
             lower_band = df_1h[bbl_cols[0]]
             upper_band = df_1h[bbu_cols[0]]
             band_width = upper_band - lower_band
             df_1h['bb_position'] = np.where(band_width == 0, 0.5, (df_1h['close'] - lower_band) / band_width)
        else:
             df_1h['bb_position'] = np.nan
             
        df_1h['volume_ratio'] = df_1h['Volume_ratio'] if 'Volume_ratio' in df_1h.columns else (df_1h['volume'] / df_1h['Volume_SMA_20'] if 'Volume_SMA_20' in df_1h.columns else np.nan)
        df_1h['atr_ratio'] = df_1h['ATR_14'] / df_1h['close'] if 'ATR_14' in df_1h.columns else np.nan
        
        stoch_cols = [c for c in df_1h.columns if c.startswith('STOCHk')]
        df_1h['stoch_k'] = df_1h[stoch_cols[0]] if stoch_cols else np.nan
        
        df_1h['ema_cross'] = df_1h['EMA_9'] - df_1h['EMA_21'] if ('EMA_9' in df_1h.columns and 'EMA_21' in df_1h.columns) else np.nan
        
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
        else:
            print("⚠️ Model does not have standard feature names.")
            return 0, 0.0, trend_str, 0
            
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            latest_1h = df_1h.iloc[[-1]][features].fillna(method='ffill').fillna(0)
        
        probs = model.predict_proba(latest_1h)
        ml_up_prob = probs[0, 1]
        
        # Step 3: Combined signal
        signal = 0
        if ml_up_prob > confidence_threshold and trend_bullish:
            signal = 1
            signal_str = "BUY"
        elif ml_up_prob < (1 - confidence_threshold) and trend_bearish:
            signal = -1
            signal_str = "SELL"
        else:
            signal_str = "HOLD"
            
        # Confidence parsing
        confidence = ml_up_prob if signal_str in ["BUY", "HOLD"] else (1 - ml_up_prob)
        if signal_str == "HOLD":
             confidence = max(ml_up_prob, 1 - ml_up_prob) # Show the strongest tilt regardless
            
        # Step 4: Logic for Score
        if signal == 1:
            score = round(ml_up_prob * 10)
        elif signal == -1:
            score = round((1 - ml_up_prob) * 10)
        else:
            score = 0
            
        print(f"Signal: {signal_str} | Confidence: {confidence*100:.0f}% | 4H Trend: {trend_str} | Score: {score}/10")
        
        return signal, confidence, trend_str, score


class RuleBasedStrategy:
    """
    Purely rule-based dual-timeframe strategy.
    Disables ML completely. Uses EMA_9/EMA_21 crossovers paired with 4H SMA_20/SMA_50 trend.
    """
    def __init__(self):
        pass

    def is_good_trading_hour(self, timestamp_utc) -> bool:
        if pd.isna(timestamp_utc) or timestamp_utc is None:
            return True
        try:
            hour = pd.to_datetime(timestamp_utc).hour
        except Exception:
            return True
        if 22 <= hour <= 23:
            return False
        return True

    def generate_signals(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame, model=None, confidence_threshold=0, position="NONE", entry_price=0.0):
        latest_1h_row = df_1h.iloc[-1]
        latest_timestamp = latest_1h_row.get('timestamp', "Unknown Time")
        
        # --- Filter 1: Trading Session ---
        if not self.is_good_trading_hour(latest_timestamp):
            print(f"Signal filtered: BAD_TRADING_SESSION (22:00-23:59 UTC) at {latest_timestamp}")
            return 0, 0.0, "NEUTRAL", 0
            
        # --- Filter 2: Volatility (ATR) ---
        if 'ATR_14' in df_1h.columns:
            lookback = min(168, len(df_1h))
            avg_atr = df_1h['ATR_14'].iloc[-lookback:].mean()
            current_atr = latest_1h_row['ATR_14']
            
            if pd.notna(avg_atr) and pd.notna(current_atr) and avg_atr > 0:
                if current_atr < 0.5 * avg_atr:
                    print(f"Signal filtered: LOW_VOLATILITY (ATR {current_atr:.2f} < 0.5 * avg_atr {avg_atr:.2f}) at {latest_timestamp}")
                    return 0, 0.0, "NEUTRAL", 0
                if current_atr > 3.0 * avg_atr:
                    print(f"Signal filtered: EXTREME_VOLATILITY (ATR {current_atr:.2f} > 3.0 * avg_atr {avg_atr:.2f}) at {latest_timestamp}")
                    return 0, 0.0, "NEUTRAL", 0

        # Step 1: Get 4H Trend Direction
        if 'SMA_20' not in df_4h.columns:
            df_4h['SMA_20'] = df_4h['close'].rolling(20).mean()
        if 'SMA_50' not in df_4h.columns:
            df_4h['SMA_50'] = df_4h['close'].rolling(50).mean()
            
        latest_4h = df_4h.iloc[-1]
        trend_bullish = latest_4h['SMA_20'] > latest_4h['SMA_50']
        trend_bearish = latest_4h['SMA_20'] < latest_4h['SMA_50']
        
        trend_str = "BULLISH" if trend_bullish else ("BEARISH" if trend_bearish else "NEUTRAL")

        # Step 2: Extract 1H Indicators
        if 'EMA_9' not in df_1h.columns:
            df_1h['EMA_9'] = df_1h['close'].ewm(span=9, adjust=False).mean()
        if 'EMA_21' not in df_1h.columns:
            df_1h['EMA_21'] = df_1h['close'].ewm(span=21, adjust=False).mean()
            
        # Check crossover state over the last 2 candles
        prev_1h = df_1h.iloc[-2]
        curr_1h = df_1h.iloc[-1]
        
        # BUY Trigger: Crosses above
        trigger_bullish = (prev_1h['EMA_9'] <= prev_1h['EMA_21']) and (curr_1h['EMA_9'] > curr_1h['EMA_21'])
        # SELL Trigger: Crosses below
        trigger_bearish = (prev_1h['EMA_9'] >= prev_1h['EMA_21']) and (curr_1h['EMA_9'] < curr_1h['EMA_21'])

        # Step 3: Combined signal
        signal = 0
        if trigger_bullish and trend_bullish:
            signal = 1
            signal_str = "BUY"
        elif trigger_bearish and trend_bearish:
            signal = -1
            signal_str = "SELL"
        else:
            signal_str = "HOLD"

        confidence = 1.0 if signal != 0 else 0.0
        score = 10 if signal != 0 else 0

        print(f"Rule-Based Signal: {signal_str} | 1H EMA: {'Crossed Up' if trigger_bullish else ('Crossed Down' if trigger_bearish else 'No Cross')} | 4H Trend: {trend_str}")
        return signal, confidence, trend_str, score

class BollingerBounceStrategy:
    """
    Purely rule-based BB mean-reversion strategy.
    Disables ML completely.
    """
    def __init__(self):
        pass

    def generate_signals(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame=None, model=None, confidence_threshold=0, position="NONE", entry_price=0.0):
        latest = df_1h.iloc[-1]
        time_str = latest.get('timestamp', "Unknown Time")
        
        if 'BBL_20_2.0' not in df_1h.columns:
            import pandas_ta as ta
            bb = ta.bbands(df_1h['close'], length=20, std=2.0)
            if bb is not None:
                df_1h = pd.concat([df_1h, bb], axis=1)
                latest = df_1h.iloc[-1]
                
        bbl_col = 'BBL_20_2.0'
        bbm_col = 'BBM_20_2.0'
        bbu_col = 'BBU_20_2.0'
        
        if bbl_col not in df_1h.columns or 'RSI_14' not in df_1h.columns:
            print("⚠️ Bollinger Bands (20) or RSI_14 missing. Generating neutral signal.")
            return 0, 0.0, "NEUTRAL", 0
            
        close_price = latest['close']
        bb_lower = latest[bbl_col]
        bb_middle = latest[bbm_col]
        bb_upper = latest[bbu_col]
        rsi = latest['RSI_14']
        
        signal = 0
        reason = "HOLD"
        
        if position == "LONG":
            if close_price >= bb_middle:
                signal = -1
                reason = "band_middle"
            elif close_price >= bb_upper * 0.999:
                signal = -1
                reason = "band_upper"
            elif rsi > 65:
                signal = -1
                reason = "rsi_overbought"
            elif close_price <= entry_price * 0.98:
                signal = -1
                reason = "stop_loss"
                
        if position == "NONE" and signal == 0:
            if close_price <= bb_lower * 1.001 and rsi < 35:
                signal = 1
                reason = "BUY"
                
        if signal == 1:
            print(f"[{time_str}] BUY — Price at lower band: ${close_price:.2f} | RSI: {rsi:.1f} | BB_lower: ${bb_lower:.2f}")
        elif signal == -1:
            pnl_pct = ((close_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
            print(f"[{time_str}] SELL — Reason: {reason} | PnL: {pnl_pct:.2f}%")
            
        return signal, 1.0, reason, 10


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(script_dir, "data", "BTC_USDT_1d.csv")
    
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        
        from backtest import run_backtest
        import warnings
        warnings.filterwarnings('ignore')
        
        # Add natively-calculated basic indicators for backtesting framework
        if 'SMA_20' not in df.columns:
            df['SMA_20'] = df['close'].rolling(20).mean()
        if 'SMA_50' not in df.columns:
            df['SMA_50'] = df['close'].rolling(50).mean()
            
        # --- MA CROSSOVER EVAL ---
        df_ma = df.copy()
        ma_strategy = MACrossoverStrategy()
        df_ma = ma_strategy.generate_signals(df_ma)
        
        print("\n--- Running MA Crossover Strategy ---")
        df_ma = run_backtest(df_ma, risk_pct=0.02)
        
        ma_ret = ((df_ma['equity'].iloc[-1] / 10000.0) - 1) * 100 if 'equity' in df_ma.columns else 0
        ma_max_dd = ((df_ma['equity'] - df_ma['equity'].cummax()) / df_ma['equity'].cummax()).min() * 100 if 'equity' in df_ma.columns else 0
        ma_dr = df_ma['equity'].pct_change() if 'equity' in df_ma.columns else pd.Series([0])
        ma_sharpe = (ma_dr.mean() / ma_dr.std()) * np.sqrt(365) if ma_dr.std() != 0 else 0
        ma_trades = len([s for s in df_ma['signal'] if s == 1])
        
        # --- ML STRATEGY EVAL ---
        df_ml = df.copy()
        
        print("\n--- Running ML Strategy ---")
        ml_strategy = MLStrategy()
        df_ml = ml_strategy.generate_signals(df_ml, confidence_threshold=0.55)
        
        df_ml = run_backtest(df_ml, risk_pct=0.02)
        
        ml_ret = ((df_ml['equity'].iloc[-1] / 10000.0) - 1) * 100 if 'equity' in df_ml.columns else 0
        ml_max_dd = ((df_ml['equity'] - df_ml['equity'].cummax()) / df_ml['equity'].cummax()).min() * 100 if 'equity' in df_ml.columns else 0
        ml_dr = df_ml['equity'].pct_change() if 'equity' in df_ml.columns else pd.Series([0])
        ml_sharpe = (ml_dr.mean() / ml_dr.std()) * np.sqrt(365) if ml_dr.std() != 0 else 0
        ml_trades = len([s for s in df_ml['signal'] if s == 1])
        
        print("\n=== Strategy Comparison ===")
        print(f"| Metric          | MA Crossover | ML Strategy |")
        print(f"|-----------------|--------------|-------------|")
        print(f"| Total Return    | {ma_ret:11.2f}% | {ml_ret:10.2f}% |")
        print(f"| Max Drawdown    | {ma_max_dd:11.2f}% | {ml_max_dd:10.2f}% |")
        print(f"| Sharpe Ratio    | {ma_sharpe:12.2f} | {ml_sharpe:11.2f} |")
        print(f"| Number Trades   | {ma_trades:12d} | {ml_trades:11d} |")
    else:
        print("⚠️ Test file not found. Please run fetcher.py first!")

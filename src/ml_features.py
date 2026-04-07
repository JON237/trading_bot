"""
ml_features.py
Prepares historical price data and technical indicators into a feature matrix
for Machine Learning models, along with forward-looking target variables.
"""

import os
import pandas as pd
import numpy as np

def prepare_features(df: pd.DataFrame):
    """
    Creates ML features from price and indicator data.
    Generates a 3-day forward-looking target variable.
    Returns train and test dataframes split chronologically (80/20).
    """
    print("🧠 Preparing Machine Learning features...")
    
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. Price Momentum Features (Past Returns)
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_3d'] = df['close'].pct_change(3)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    
    # 2. Indicator Relative Features
    # RSI
    rsi_cols = [c for c in df.columns if c.startswith('RSI')]
    if rsi_cols:
         df['feature_rsi'] = df[rsi_cols[0]]
         
    # MACD Histogram
    macd_hist_cols = [c for c in df.columns if c.startswith('MACDh')]
    if macd_hist_cols:
         df['feature_macd_hist'] = df[macd_hist_cols[0]]
         
    # SMA 20 Distance
    if 'SMA_20' in df.columns:
         df['dist_sma20'] = (df['close'] - df['SMA_20']) / df['SMA_20']
         
    # Volume Ratio
    if 'Volume_SMA_20' in df.columns:
         df['vol_ratio'] = df['volume'] / df['Volume_SMA_20']
         
    # Bollinger Band Position
    bbl_cols = [c for c in df.columns if c.startswith('BBL')]
    bbu_cols = [c for c in df.columns if c.startswith('BBU')]
    if bbl_cols and bbu_cols:
         lower_band = df[bbl_cols[0]]
         upper_band = df[bbu_cols[0]]
         # Calculate position (0 = at lower band, 1 = at upper band)
         band_width = upper_band - lower_band
         # Avoid division by zero
         df['bb_position'] = np.where(band_width == 0, 0.5, (df['close'] - lower_band) / band_width)
         
    # 3. Target Variable (Future Returns - Lookahead)
    # 1 if price is higher in 3 days, 0 if lower
    df['future_close_3d'] = df['close'].shift(-3)
    df['target'] = (df['future_close_3d'] > df['close']).astype(int)
    
    # Drop rows with NaN values created by shifts and rolling windows
    # Including the last 3 days which won't have a future_close_3d
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # List of feature columns to keep alongside target
    feature_cols = [
        'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d',
        'feature_rsi', 'feature_macd_hist', 'dist_sma20', 
        'vol_ratio', 'bb_position'
    ]
    
    # Filter to ensure we only keep features that were successfully created
    valid_features = [col for col in feature_cols if col in df.columns]
    
    print(f"📊 Matrix generated with {len(valid_features)} valid features and {len(df)} rows.")
    
    # 4. Train / Test Split (Time-based, 80/20)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"✂️ Split Data -> Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    # Print Class Balance
    train_1s = train_df['target'].sum()
    train_0s = len(train_df) - train_1s
    
    test_1s = test_df['target'].sum()
    test_0s = len(test_df) - test_1s
    
    print("\n⚖️ Class Balance (Target = 1 if Price Up in 3 Days):")
    print(f"Training Set : {train_1s} Positives (1), {train_0s} Negatives (0)")
    print(f"Testing Set  : {test_1s} Positives (1), {test_0s} Negatives (0)")
    
    return train_df, test_df, valid_features

def prepare_features_1h(df: pd.DataFrame):
    """
    Creates ML features specifically for the 1H timeframe.
    Target: 1 if close price 3 hours later is higher than current close.
    Splits chronologically (80% train, 20% test).
    """
    df = df.copy()
    
    # Prevent lookahead bias implicitly with shift() for past data
    df['price_change_1h'] = df['close'].pct_change(1)
    df['price_change_3h'] = df['close'].pct_change(3)
    df['price_change_6h'] = df['close'].pct_change(6)
    df['price_change_24h'] = df['close'].pct_change(24)
    
    # 1H specific indicator parsing
    df['rsi_14'] = df['RSI_14'] if 'RSI_14' in df.columns else np.nan
    df['rsi_7'] = df['RSI_7'] if 'RSI_7' in df.columns else np.nan
    
    macd_cols = [c for c in df.columns if c.startswith('MACDh')]
    df['macd_hist'] = df[macd_cols[0]] if macd_cols else np.nan
    
    bbl_cols = [c for c in df.columns if c.startswith('BBL')]
    bbu_cols = [c for c in df.columns if c.startswith('BBU')]
    if bbl_cols and bbu_cols:
         lower_band = df[bbl_cols[0]]
         upper_band = df[bbu_cols[0]]
         band_width = upper_band - lower_band
         df['bb_position'] = np.where(band_width == 0, 0.5, (df['close'] - lower_band) / band_width)
    else:
         df['bb_position'] = np.nan
         
    df['volume_ratio'] = df['Volume_ratio'] if 'Volume_ratio' in df.columns else (df['volume'] / df['Volume_SMA_20'] if 'Volume_SMA_20' in df.columns else np.nan)
    df['atr_ratio'] = df['ATR_14'] / df['close'] if 'ATR_14' in df.columns else np.nan
    
    stoch_cols = [c for c in df.columns if c.startswith('STOCHk')]
    df['stoch_k'] = df[stoch_cols[0]] if stoch_cols else np.nan
    
    df['ema_cross'] = df['EMA_9'] - df['EMA_21'] if ('EMA_9' in df.columns and 'EMA_21' in df.columns) else np.nan
    
    # Target Variable: close price 3 hours (periods) later
    df['future_close_3h'] = df['close'].shift(-3)
    df['target'] = (df['future_close_3h'] > df['close']).astype(int)
    
    # Ensure no NaN rows from shift calculations, dropping last 3 rows plus initial lookbacks
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    features = [
        'price_change_1h', 'price_change_3h', 'price_change_6h', 'price_change_24h',
        'rsi_14', 'rsi_7', 'macd_hist', 'bb_position', 'volume_ratio', 'atr_ratio',
        'stoch_k', 'ema_cross'
    ]
    
    X = df[features]
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Train / Test split (80% / 20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    total = len(df)
    ups = df['target'].sum()
    downs = total - ups
    print(f"Class balance: UP: {ups/total*100:.1f}% | DOWN: {downs/total*100:.1f}%")
    
    if 'timestamp' in df.columns:
        t_col = pd.to_datetime(train_df['timestamp'])
        print(f"Training Data Range: {t_col.iloc[0].strftime('%Y-%m-%d')} to {t_col.iloc[-1].strftime('%Y-%m-%d')}")
        
        test_col = pd.to_datetime(test_df['timestamp'])
        print(f"Testing Data Range : {test_col.iloc[0].strftime('%Y-%m-%d')} to {test_col.iloc[-1].strftime('%Y-%m-%d')}")
        
    return train_df, test_df, features

if __name__ == "__main__":
    # Test script locally
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(script_dir, "data", "BTC_USDT_1d.csv")
    
    if os.path.exists(test_file):
         df = pd.read_csv(test_file)
         
         # Dynamically add indicators to test the ML features fully
         from indicators import add_indicators
         print("Loading technical indicators via pandas-ta...")
         df = add_indicators(df)
         
         train_df, test_df, features = prepare_features(df)
         
         if not train_df.empty:
             print("\n🔍 Example Training Features (First row):")
             print(train_df[features + ['target']].iloc[0])
    else:
         print("⚠️ Test file not found. Run fetcher.py first!")

import pandas as pd
import numpy as np
import os

def check_data(filepath: str):
    """
    Validates the data quality of 1H candlestick data.
    """
    print("Data Quality Report\n")
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        print("Data ready: NO")
        return
        
    # 1. Load data
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 5. Missing values (NaN)
    nan_counts = df.isna().sum()
    nan_total = nan_counts.sum()
    if nan_total > 0:
        print("Missing values (NaN) per column:")
        print(nan_counts[nan_counts > 0])
        print()
        
    # 2. Duplicate timestamps
    dup_mask = df.duplicated(subset=['timestamp'], keep='first')
    dup_count = dup_mask.sum()
    if dup_count > 0:
        df = df[~dup_mask].copy()
    
    # Ensure data is sorted by timestamp after duplicates are removed
    df = df.sort_values('timestamp').reset_index(drop=True)
        
    # 1. Missing candles (gaps > 1 hour)
    time_diffs = df['timestamp'].diff()
    gaps_mask = time_diffs > pd.Timedelta(hours=1)
    gaps_count = gaps_mask.sum()
    
    if gaps_count > 0:
        print(f"Found {gaps_count} missing intervals (gaps > 1 hour). First few instances:")
        gaps_idx = time_diffs[gaps_mask].index
        for idx in list(gaps_idx)[:5]:
            print(f"  Gap between {df['timestamp'].iloc[idx-1]} and {df['timestamp'].iloc[idx]} (Duration: {time_diffs.loc[idx]})")
        print()
            
    # 3. Zero or negative prices
    price_cols = ['open', 'high', 'low', 'close']
    bad_price_mask = (df[price_cols] <= 0).any(axis=1)
    bad_price_count = bad_price_mask.sum()
    if bad_price_count > 0:
        df = df[~bad_price_mask].copy()
        
    # 4. Extreme outliers (change > 20% in 1 hour)
    df['pct_change'] = df['close'].pct_change().abs()
    outlier_mask = df['pct_change'] > 0.20
    outliers_count = outlier_mask.sum()
    if outliers_count > 0:
        print(f"Found {outliers_count} extreme outliers. First few instances:")
        print(df[outlier_mask][['timestamp', 'close', 'pct_change']].head())
        print()
    
    # Drop calculation columns
    df = df.drop(columns=['pct_change'])
    
    # Determine ready state
    is_ready = "YES" if (gaps_count < 100 and nan_total == 0 and bad_price_count == 0) else "NO"
    final_count = len(df)
    
    # Save cleaned data
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_filepath = os.path.join(script_dir, "data", "BTC_USDT_1h_clean.csv")
    os.makedirs(os.path.dirname(clean_filepath), exist_ok=True)
    df.to_csv(clean_filepath, index=False)
    
    print("--- Summary Report ---")
    print(f"Total candles: {final_count}")
    print(f"Missing gaps found: {gaps_count}")
    print(f"Duplicates removed: {dup_count}")
    print(f"Outliers flagged: {outliers_count}")
    print(f"Data ready: {is_ready}")
    print(f"\nSaved cleaned data to: {clean_filepath}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(script_dir, "data", "BTC_USDT_1h.csv")
    check_data(filepath)

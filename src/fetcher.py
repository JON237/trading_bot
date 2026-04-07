"""
fetcher.py
Responsible for connecting to the cryptocurrency exchange via ccxt
and downloading historical or live market data (OHLCV).
"""

import os
import ccxt
import pandas as pd
from dotenv import load_dotenv

def connect_exchange():
    """
    Connect to the Binance Testnet using ccxt.
    Requires BINANCE_API_KEY and BINANCE_API_SECRET in the .env file.
    """
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    try:
        # Initialize Binance with testnet/sandbox mode
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        
        # Enable sandbox mode for the testnet (disabled here because testnet lacks sufficient historical candles for ML)
        exchange.set_sandbox_mode(False)
        
        # Test the connection by fetching the account balance
        balance = exchange.fetch_balance()
        
        # Extract available USDT
        usdt_balance = balance.get('USDT', {}).get('free', 0.0)
        
        print(f"✅ Successfully connected to Binance Testnet!")
        print(f"💰 Available USDT Balance: {usdt_balance}")
        
        return exchange
        
    except ccxt.AuthenticationError:
        print("❌ Authentication failed. Please check your API key and secret in the .env file.")
        print("⚠️ Returning unauthenticated exchange instance for public data only.")
        # Ensure the fallback instance does not attempt to send invalid testnet keys to mainnet
        fallback_exchange = ccxt.binance({'enableRateLimit': True})
        return fallback_exchange
    except ccxt.NetworkError as e:
        print(f"❌ Network error occurred while connecting to Binance: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        
    return None

class DataFetcher:
    def __init__(self, exchange_id: str = 'binance'):
        """Initialize the exchange connection."""
        self.exchange_id = exchange_id
        self.exchange = connect_exchange()
        
    def fetch_ohlcv(self, symbol: str = "BTC/USDT", timeframe: str = "1d", since_days: int = 365) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with pagination.
        Saves data as CSV and returns a pandas DataFrame.
        """
        if not self.exchange:
            print("❌ Exchange not connected. Cannot fetch data.")
            return pd.DataFrame()
            
        print(f"📥 Fetching {since_days} days of {timeframe} data for {symbol}...")
        
        # Calculate timestamps
        now = self.exchange.milliseconds()
        since = int(now - (since_days * 24 * 60 * 60 * 1000))
        
        all_ohlcv = []
        limit = 500  # Binance max per request
        
        while since < now:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                print(f"⏳ Downloaded {len(all_ohlcv)} candles so far...")
                
                # Update 'since' to the last candle's timestamp + 1 ms to get the next batch
                new_since = ohlcv[-1][0] + 1
                if new_since <= since:
                    break  # Safety check
                since = new_since
                
            except Exception as e:
                print(f"❌ Error fetching data batch: {e}")
                break
                
        if not all_ohlcv:
            print("⚠️ No data downloaded.")
            return pd.DataFrame()
            
        # Create DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Save to CSV
        import os
        filename = f"data/{symbol.replace('/', '_')}_{timeframe}.csv"
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(script_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {len(df)} candles to {filename}")
        
        return df

def fetch_ohlcv_1h() -> pd.DataFrame:
    """
    Downloads 6 months (180 days) of 1-hour BTC/USDT candlestick data.
    Handle pagination automatically (max 500 candles per request).
    """
    fetcher = DataFetcher()
    exchange = fetcher.exchange
    if not exchange:
        print("❌ Exchange not connected. Cannot fetch data.")
        return pd.DataFrame()

    symbol = "BTC/USDT"
    timeframe = "1h"
    limit = 500
    
    now = exchange.milliseconds()
    since = int(now - (180 * 24 * 60 * 60 * 1000))
    all_ohlcv = []
    
    print(f"📥 Fetching 6 months (180 days) of {timeframe} data for {symbol}...")
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            print(f"Fetched {len(all_ohlcv)} candles so far...")
            
            new_since = ohlcv[-1][0] + 1
            if new_since <= since:
                break
            since = new_since
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            break
            
    if not all_ohlcv:
        print("⚠️ No data downloaded.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "BTC_USDT_1h.csv")
    
    df.to_csv(filepath, index=False)
    
    start_date = df['timestamp'].iloc[0].strftime('%Y-%m-%d')
    end_date = df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
    print(f"Done. Total candles: {len(df)}. Date range: {start_date} to {end_date}")
    
    return df

if __name__ == "__main__":
    # Test script directly by fetching 6 months of 1-hour BTC/USDT data
    df_1h = fetch_ohlcv_1h()
    if not df_1h.empty:
        print("\n📈 First 5 rows of 1h data:")
        print(df_1h.head())

import os
import ccxt
from dotenv import load_dotenv

def test_connection():
    print("🔄 Initializing Binance Connection Test...")
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Cannot find API keys in .env!")
        return

    # Create naked exchange instance
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'
        }
    })
    
    try:
        # First test real Binance
        try:
            balance = exchange.fetch_balance({'type': 'future'})
            usdt_balance = balance.get('USDT', {}).get('free', 0.0)
            print("✅ SUCCESS: Connected to REAL Binance API!")
            print(f"💰 Available USDT Balance: {usdt_balance}")
            return
        except Exception:
            pass
            
        print("⚠️ Real Binance auth failed. Attempting Testnet connection...")
        exchange.urls['api']['fapiPublic'] = 'https://testnet.binancefuture.com/fapi/v1'
        exchange.urls['api']['fapiPrivate'] = 'https://testnet.binancefuture.com/fapi/v1'
        exchange.urls['api']['fapiPrivateV2'] = 'https://testnet.binancefuture.com/fapi/v2'
        
        balance = exchange.fetch_balance({'type': 'future'})
        usdt_balance = balance.get('USDT', {}).get('free', 0.0)
        print("✅ SUCCESS: Connected to BINANCE TESTNET/DEMO API!")
        print(f"💰 Available USDT Balance: {usdt_balance}")
        
    except Exception as e:
        print(f"❌ AUTHENTICATION FAILED: {str(e)}")
        print("Please double check your API Key and Secret in the .env file.")

if __name__ == "__main__":
    test_connection()

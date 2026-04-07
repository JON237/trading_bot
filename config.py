"""
config.py
Configuration and settings for the trading bot.
Handles loading environment variables (API keys, secrets) via python-dotenv.
"""

import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if present)
load_dotenv()

# Exchange settings
EXCHANGE_ID = os.getenv('EXCHANGE_ID', 'binance')
API_KEY = os.getenv('API_KEY', 'your_api_key_here')
API_SECRET = os.getenv('API_SECRET', 'your_api_secret_here')

# Trading parameters
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'

"""
config.py
Central configuration file for the trading bot.
"""

# ML disabled — set USE_ML_MODEL = True in config.py to re-enable
USE_ML_MODEL = False

# LIVE TRADING DANGER ZONE: Set to True to allow real Binance API execution
LIVE_TRADING_ENABLED = False

# TESTNET DANGER ZONE: Switch between real money and Binance Testnet
USE_TESTNET = True

# Position size for real trades in USDT (e.g. 50 USDT per trade)
TRADE_SIZE_USDT = 50.0

# Futures Leverage Multiplier (1x to 125x)
FUTURES_LEVERAGE = 5

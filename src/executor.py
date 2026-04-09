"""
executor.py
Handles live execution of orders to Binance Futures via CCXT with advanced risk controls.
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import logging
try:
    from config import LIVE_TRADING_ENABLED, TRADE_SIZE_USDT, FUTURES_LEVERAGE
except ImportError:
    LIVE_TRADING_ENABLED = False
    TRADE_SIZE_USDT = 50.0
    FUTURES_LEVERAGE = 5

class BinanceExecutor:
    def __init__(self, exchange):
        self.exchange = exchange
        self.symbol = "BTC/USDT"
        
        if self.exchange:
            try:
                self.exchange.load_markets()
                if LIVE_TRADING_ENABLED:
                    # Initialize Futures Leverage
                    self.exchange.set_leverage(FUTURES_LEVERAGE, self.symbol)
                    # Initialize Margin Mode (Isolated prevents cross-collateral liquidation)
                    self.exchange.set_margin_mode('isolated', self.symbol)
            except Exception as e:
                logging.error(f"Could not initialize Futures market constraints: {e}")

    def check_active_position(self) -> bool:
        """
        Safety Feature 3: Schutz gegen doppelte Orders
        Queries Binance directly to ensure no open BTC position exists.
        """
        if not self.exchange:
            return False
            
        try:
            positions = self.exchange.fetch_positions(symbols=[self.symbol])
            for pos in positions:
                contracts = float(pos.get('contracts', 0) or pos.get('positionAmt', 0))
                if contracts > 0:
                    return True
            return False
        except Exception as e:
            logging.error(f"Failed to check positions: {e}")
            return True # Block trades safely if API fails

    def verify_liquidation_distance(self, current_price: float, sl_price: float) -> bool:
        """
        Safety Feature 4: Liquidationsabstand
        Ensures the Stop-Loss mathematically triggers prior to Binance margin liquidation.
        """
        drop_pct = (current_price - sl_price) / current_price
        max_drop = 1.0 / FUTURES_LEVERAGE
        
        # We need at least a 10% safety buffer before the exchange native liquidation point
        safety_threshold = max_drop * 0.90
        
        if drop_pct >= safety_threshold:
            logging.error(f"⚠️ LIQUIDATION RISK REJECTED: SL Drop {drop_pct*100:.2f}% too large for {FUTURES_LEVERAGE}x Leverage!")
            return False
        return True

    def buy_market(self, current_price: float) -> dict:
        """
        Executes a Market Buy on Binance Futures using strictly calculated boundaries.
        Safety Feature 1: Feste Maximalgröße (TRADE_SIZE_USDT)
        """
        if not LIVE_TRADING_ENABLED:
            print(f"🛑 [SIMULATION] Would place FUTURES MARKET BUY for ~${TRADE_SIZE_USDT} USDT")
            return {'status': 'simulated', 'filled': TRADE_SIZE_USDT / current_price, 'price': current_price}
            
        if self.check_active_position():
            print("🚫 Double Order Protection: Existing position found. Aborting Buy.")
            return None
            
        # Amount calculation includes leverage
        raw_btc_amount = (TRADE_SIZE_USDT * FUTURES_LEVERAGE) / current_price
        btc_amount_formatted = float(self.exchange.amount_to_precision(self.symbol, raw_btc_amount))
        
        print(f"🚀 LIVE TRADING: Placing Futures Market BUY for {btc_amount_formatted} BTC...")
        
        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side='buy',
                amount=btc_amount_formatted
            )
            print(f"✅ Futures Buy filled! Order ID: {order.get('id')}")
            return order
        except Exception as e:
            logging.error(f"❌ LIVE MARKET BUY FAILED: {e}")
            return None

    def execute_sell(self, btc_amount: float):
        """
        Emergency or Signal-based early exit.
        """
        if not LIVE_TRADING_ENABLED: return
        try:
            btc_amount_formatted = float(self.exchange.amount_to_precision(self.symbol, btc_amount))
            self.exchange.create_order(self.symbol, 'market', 'sell', btc_amount_formatted, params={'reduceOnly': True})
        except Exception as e:
             logging.error(f"❌ MARKET SELL FAILED: {e}")

    def cancel_all_orders(self):
        if not LIVE_TRADING_ENABLED: return
        try:
            self.exchange.cancel_all_orders(self.symbol)
        except Exception as e:
            logging.error(f"⚠️ Order cancellation failed: {e}")

    def place_oco_sell(self, btc_amount: float, entry_price: float, tp_price: float, sl_price: float):
        """
        Safety Feature 2: Hartes Stop-Loss System (Futures Nativ)
        Instead of Spot OCO, we set a TAKE_PROFIT_MARKET and STOP_MARKET.
        Both will have 'closePosition': True.
        """
        if not LIVE_TRADING_ENABLED:
            print(f"🛑 [SIMULATION] Would set Futures Defensive SL @ ${sl_price:.2f} | TP @ ${tp_price:.2f}")
            return
            
        if not self.verify_liquidation_distance(entry_price, sl_price):
            print("🚫 Risk Engine decoupled trade due to liquidation distances.")
            self.execute_sell(btc_amount) # Exit safely
            return
            
        tp_price_formatted = float(self.exchange.price_to_precision(self.symbol, tp_price))
        sl_price_formatted = float(self.exchange.price_to_precision(self.symbol, sl_price))
        
        print(f"🚀 LIVE TRADING: Setting Futures SL @ ${sl_price_formatted} & TP @ ${tp_price_formatted}")
        
        # Place Stop-Market (SL)
        try:
            self.exchange.create_order(
                symbol=self.symbol, type='STOP_MARKET', side='sell', amount=None,
                params={'stopPrice': sl_price_formatted, 'closePosition': True}
            )
        except Exception as e:
            logging.error(f"❌ STOP LOSS PLACEMENT FAILED: {e}. Executing emergency close!")
            self.execute_sell(btc_amount)
            return
            
        # Place Take-Profit-Market (TP)
        try:
            self.exchange.create_order(
                symbol=self.symbol, type='TAKE_PROFIT_MARKET', side='sell', amount=None,
                params={'stopPrice': tp_price_formatted, 'closePosition': True}
            )
        except Exception as e:
            logging.error(f"⚠️ TAKE PROFIT PLACEMENT FAILED: {e}")

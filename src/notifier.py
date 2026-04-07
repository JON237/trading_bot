"""
notifier.py
Handles sending notifications and alerts via the Telegram Bot API.
"""

import os
import requests
from dotenv import load_dotenv

def send_telegram(message: str, bot_token: str = None, chat_id: str = None):
    """
    Sends a message via the Telegram Bot API.
    Loads credentials from the .env file if not provided explicitly.
    """
    load_dotenv()
    
    token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
    chat = chat_id or os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat:
        print("⚠️ Telegram credentials not found in .env. Notification skipped.")
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": message
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")
        return False

def format_usd(val):
    if val >= 0:
         return f"+${val:,.0f}"
    return f"-${abs(val):,.0f}"

def send_trade_open_alert(price, pos_size, invest_amt, stop_price, tp_price, conf, score, trend="BULLISH"):
    stop_pct = ((stop_price - price) / price) * 100
    tp_pct = ((tp_price - price) / price) * 100
    msg = f"""🟢 TRADE OPEN
Pair: BTC/USDT
Action: BUY
Price: ${price:,.0f}
Position size: {pos_size:.4f} BTC (${invest_amt:,.0f})
Stop-Loss: ${stop_price:,.0f} ({stop_pct:+.1f}%)
Take-Profit: ${tp_price:,.0f} ({tp_pct:+.1f}%)
Confidence: {conf:.0f}% | Score: {score}/10
4H Trend: {trend}"""
    return send_telegram(msg)

def send_trade_closed_alert(reason, entry, exit_price, pnl_usd, pnl_pct, duration_hrs, portfolio_usd, port_pct):
    action = f"SELL ({reason})"
    msg = f"""⚪ TRADE CLOSED
Action: {action}
Entry: ${entry:,.0f} -> Exit: ${exit_price:,.0f}
PnL: {format_usd(pnl_usd)} ({pnl_pct:+.1f}%)
Duration: {duration_hrs:.1f} hours
Portfolio: ${portfolio_usd:,.0f} ({port_pct:+.1f}% total)"""
    return send_telegram(msg)

def send_daily_summary(trades, wins, losses, daily_pnl_usd, daily_pnl_pct, total_pnl_usd, total_pnl_pct, best_pct, worst_pct, open_pos):
    msg = f"""📊 DAILY SUMMARY
Trades today: {trades}
Wins: {wins} | Losses: {losses}
Daily PnL: {format_usd(daily_pnl_usd)} ({daily_pnl_pct:+.1f}%)
Total PnL: {format_usd(total_pnl_usd)} ({total_pnl_pct:+.1f}%)
Best trade: {best_pct:+.1f}% | Worst: {worst_pct:+.1f}%
Open position: {open_pos}"""
    return send_telegram(msg)

def send_risk_alert(loss_usd, loss_pct, reason="Price dropped below ATR stop"):
    msg = f"""🔴 RISK ALERT: Stop-loss triggered
Loss: {format_usd(loss_usd)} ({loss_pct:+.1f}%)
Reason: {reason}"""
    return send_telegram(msg)

if __name__ == "__main__":
    # Test notification
    success = send_telegram("Test message from notifier.py!")
    if success:
        print("✅ Test message sent successfully!")

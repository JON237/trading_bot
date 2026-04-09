"""
notifier.py
Handles sending notifications and alerts via the Telegram Bot API.
"""

import os
import socket
import platform
import requests
from dotenv import load_dotenv


def get_env_label() -> str:
    """
    Detects whether the bot is running locally (Mac) or on the Oracle server.
    Returns a short label used in Telegram messages.
    """
    hostname = socket.gethostname().lower()
    system = platform.system().lower()

    # Oracle hostnames typically contain 'instance' or are plain Linux
    if 'oracle' in hostname or 'instance' in hostname:
        return "☁️ Oracle Server"
    elif system == 'darwin':
        return "💻 Lokal (Mac)"
    else:
        # Generic Linux — likely the server
        return f"☁️ Server ({hostname})"


def send_telegram(message: str, bot_token: str = None, chat_id: str = None):
    """
    Sends a message via the Telegram Bot API.
    Loads credentials from the .env file if not provided explicitly.
    """
    load_dotenv(override=True)
    
    token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
    chat = chat_id or os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat:
        print("⚠️ Telegram credentials not found in .env. Notification skipped.")
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")
        return False


def send_bot_started(mode: str = "Paper Trading", strategy: str = "BollingerBounce"):
    """Sends a start notification with env label."""
    env = get_env_label()
    msg = (
        f"🟢 <b>Bot gestartet</b>\n"
        f"Umgebung: {env}\n"
        f"Modus: {mode}\n"
        f"Strategie: {strategy}\n"
        f"Zeitrahmen: 15 Minuten"
    )
    return send_telegram(msg)


def send_bot_stopped(reason: str = "Unbekannt"):
    """Sends a stop notification with env label."""
    env = get_env_label()
    msg = (
        f"🔴 <b>Bot gestoppt!</b>\n"
        f"Umgebung: {env}\n"
        f"Grund: {reason}"
    )
    return send_telegram(msg)


def format_usd(val):
    if val >= 0:
         return f"+${val:,.0f}"
    return f"-${abs(val):,.0f}"

def send_trade_open_alert(price, pos_size, invest_amt, stop_price, tp_price, conf, score, trend="BULLISH"):
    stop_pct = ((stop_price - price) / price) * 100
    tp_pct = ((tp_price - price) / price) * 100
    env = get_env_label()
    msg = f"""🟢 TRADE OPEN ({env})
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
    env = get_env_label()
    msg = f"""⚪ TRADE CLOSED ({env})
Action: {action}
Entry: ${entry:,.0f} -> Exit: ${exit_price:,.0f}
PnL: {format_usd(pnl_usd)} ({pnl_pct:+.1f}%)
Duration: {duration_hrs:.1f} hours
Portfolio: ${portfolio_usd:,.0f} ({port_pct:+.1f}% total)"""
    return send_telegram(msg)

def send_daily_summary(trades, wins, losses, daily_pnl_usd, daily_pnl_pct, total_pnl_usd, total_pnl_pct, best_pct, worst_pct, open_pos):
    env = get_env_label()
    msg = f"""📊 DAILY SUMMARY ({env})
Trades today: {trades}
Wins: {wins} | Losses: {losses}
Daily PnL: {format_usd(daily_pnl_usd)} ({daily_pnl_pct:+.1f}%)
Total PnL: {format_usd(total_pnl_usd)} ({total_pnl_pct:+.1f}%)
Best trade: {best_pct:+.1f}% | Worst: {worst_pct:+.1f}%
Open position: {open_pos}"""
    return send_telegram(msg)

def send_risk_alert(loss_usd, loss_pct, reason="Price dropped below ATR stop"):
    env = get_env_label()
    msg = f"""🔴 RISK ALERT: Stop-loss triggered ({env})
Loss: {format_usd(loss_usd)} ({loss_pct:+.1f}%)
Reason: {reason}"""
    return send_telegram(msg)

if __name__ == "__main__":
    # Test notification
    success = send_telegram(f"🧪 Test: {get_env_label()}")
    if success:
        print("✅ Test message sent successfully!")

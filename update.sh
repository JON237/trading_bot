#!/bin/bash
# update.sh
# Pulls the latest code from GitHub and forcefully restarts the bot service

REMOTE_DIR="/opt/tradingbot"

echo "🔄 Pulling latest changes from GitHub..."
cd $REMOTE_DIR
sudo -u $USER git pull origin main

echo "📦 Installing new requirements (if any)..."
source venv/bin/activate
pip install -r requirements.txt

# In case new indicators/model scripts require migrations
# python src/ml_model.py # Un-comment to auto-retrain model during updates

echo "⚙️ Restarting trading-bot service..."
sudo systemctl restart trading-bot

echo "📊 Verification:"
sudo systemctl status trading-bot --no-pager
echo "✅ Update Complete!"

#!/bin/bash
# deploy.sh
# Deploys the trading bot from your local Mac to an Ubuntu 22.04 server (Oracle Cloud)

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./deploy.sh <user@server_ip> <path_to_ssh_key>"
    echo "Example: ./deploy.sh ubuntu@123.45.67.89 ~/.ssh/id_rsa"
    exit 1
fi

SERVER=$1
SSH_KEY=$2
REMOTE_DIR="/opt/tradingbot"

echo "🚀 Starting Deployment to $SERVER..."

# 1. Create remote directory and set ownership so SCP works without sudo for file transfer
echo "📂 Formatting server directories..."
ssh -i "$SSH_KEY" "$SERVER" "sudo mkdir -p $REMOTE_DIR && sudo chown -R \$USER:\$USER $REMOTE_DIR"

# 2. Package and SCP local files (excluding venv, .git, and cache to save time/bandwidth)
echo "📦 Uploading project files via SCP..."
tar -cf - --exclude='./venv' --exclude='./.git' --exclude='./__pycache__' --exclude='./.pytest_cache' . | ssh -i "$SSH_KEY" "$SERVER" "tar -xf - -C $REMOTE_DIR"

# 3. Execute setup operations remotely
echo "🔧 Running remote server setup (Python, Packages, Systemd, Logrotate)..."
ssh -i "$SSH_KEY" "$SERVER" "bash -s" << 'EOF'
    REMOTE_DIR="/opt/tradingbot"
    
    # System Updates & Dependencies
    sudo apt update
    sudo apt install -y software-properties-common tar git
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

    # Setup Python Virtual Environment
    cd $REMOTE_DIR
    python3.11 -m venv venv
    source venv/bin/activate
    
    # Upgrading pip and installing bot requirements
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Make scripts executable
    chmod +x update.sh deploy.sh
    
    # Configure Systemd Service
    SERVICE_FILE="/etc/systemd/system/trading-bot.service"
    sudo bash -c "cat > $SERVICE_FILE" << EOL
[Unit]
Description=Multi-Timeframe 1H Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REMOTE_DIR
ExecStart=$REMOTE_DIR/venv/bin/python $REMOTE_DIR/src/bot.py
Restart=always
RestartSec=10
Environment="PATH=$REMOTE_DIR/venv/bin"
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
EOL

    sudo systemctl daemon-reload
    sudo systemctl enable trading-bot
    sudo systemctl restart trading-bot

    # Configure Logrotate for auto-cleanup of CSV trade logs & system logs
    LOGROTATE_FILE="/etc/logrotate.d/trading-bot"
    sudo bash -c "cat > $LOGROTATE_FILE" << EOL
$REMOTE_DIR/logs/*.csv {
    daily
    missingok
    rotate 14
    compress
    notifempty
    copytruncate
}
EOL

    # Deployment Verification
    echo "----------------------------------------"
    echo "✅ Setup Complete. Checking Service Status:"
    sudo systemctl status trading-bot --no-pager
EOF

echo "🎉 Deployment executed successfully!"

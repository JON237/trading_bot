# 🚀 KI-Kryptowährungs Trading-Bot

Willkommen zum KI-gestützten Kryptowährungs Trading-Bot! Dieses Projekt ist ein vollständiges automatisiertes Handelssystem, das historische Marktdaten analysiert, Maschinelles Lernen (Künstliche Intelligenz) nutzt, um Preisbewegungen vorherzusagen, und virtuelle Trades (Paper Trading) durchführt – komplett mit Telegram-Benachrichtigungen und einem interaktiven Live-Dashboard.

---

## ✨ Was kann dieser Bot? (Funktionen)

Dieser Bot wurde entwickelt, um den gesamten Handelsprozess für dich abzubilden, ohne dass du selbst vor den Charts sitzen musst:

1. **📊 Marktdaten laden (`fetcher.py`):** Verbindet sich mit der Krypto-Börse (Binance Testnet) und lädt aktuelle Preisdatenpunkte im Hintergrund herunter.
2. **📈 Technische Analyse (`indicators.py`):** Berechnet automatisch bekannte Börsen-Indikatoren wie Gleitende Durchschnitte (SMA/EMA), RSI, MACD und Bollinger Bänder, um den Markt zu "lesen".
3. **🧠 Künstliche Intelligenz (`ml_features.py` & `ml_model.py`):** Eine trainierte Random-Forest-KI analysiert dutzende Indikatoren gleichzeitig und berechnet ein Vertrauens-Level (z.B. 75 %), ob der Kurs in den nächsten Tagen steigen oder fallen wird.
4. **🛡️ Risikomanagement (`backtest.py` & `bot.py`):** Der Bot simuliert Trades mit virtuellem Geld und schützt dein Kapital automatisch durch eingebaute Stop-Loss- (automatischer Notverkauf bei einem Kursrutsch von z.B. 3%) und Take-Profit-Grenzen (Gewinnmitnahmen).
5. **📲 Telegram-Alarme (`notifier.py`):** Bei jedem Kauf, jedem Verkauf oder zum täglichen Zusammenfassen deines Kontostands um Mitternacht schickt der Bot dir automatisch eine Nachricht direkt aufs Handy über die Telegram-App!
6. **🖥️ Live-Dashboard (`dashboard.py`):** Eine professionelle Web-Oberfläche zeigt dir live alle prozentualen Gewinne, die aktuelle Wachstumskurve deines Kontos (Equity-Curve) und signalisiert dir exakt, zu welchen Preisen zuletzt gehandelt wurde.

---

## 🛠️ Einrichtung (Schritt für Schritt)

Auch wenn du nicht fließend programmieren kannst, lässt sich der Bot in wenigen einfachen Schritten starten:

### 1. Voraussetzungen installieren
Du benötigst Python (Version 3.11 oder neuer). Öffne dein Terminal (die Kommandozeile deines PCs) im Hauptordner dieses Projekts und installiere die benötigten Pakete:
```bash
pip install -r requirements.txt
```

*(Hinweis: Für Benutzer der neuesten Python-Versionen ist Pandas auf eine kompatible Version beschränkt, damit die Indikator-Bibliotheken reibungslos funktionieren!)*

### 2. Zugangsdaten eintragen (Die .env Datei)
Der Bot benötigt private Zugangsdaten, um z.B. Telegram-Nachrichten auf dein Handy zu senden.
1. Finde die Datei namens `.env.example` in deinem Ordner.
2. Kopiere die Datei und benenne die Kopie einfach nur `.env`.
3. Öffne die `.env`-Datei mit einem Texteditor.
4. Trage dort deinen **Telegram Bot Token** und deine **Telegram Chat ID** ein (Das Binance API-Feld kannst du ignorieren, solange du nur öffentliche Daten testest).

---

## 🚀 Wie bediene ich den Bot?

Das System besteht aus mehreren Modulen, die du je nach Bedarf unabhängig voneinander starten kannst. Öffne dein Terminal im Projektordner (`tradingbot/`) und führe die gewünschten Befehle aus:

### 🤖 1. Den Haupt-Bot starten (Live Paper-Trading)
Das ist das Herzstück des Systems! Der Bot startet, weckt die K.I. auf, analysiert stündlich den Krypto-Markt (Multi-Timeframe 1H & 4H Strategie), wickelt virtuelle Trades ab und schickt dir strukturierte Telegram-Alerts.
```bash
python src/bot.py
```
> **Tipp:** Drücke `STRG + C` im Terminal, um den Bot jederzeit sicher zu stoppen.

### 🖥️ 2. Das Live-Dashboard öffnen
Möchtest du alles auf einen Blick sehen? Starte die neuste Web-Oberfläche für den KI-Bot:
```bash
streamlit run dashboard_1h.py
```
> Es öffnet sich automatisch dein Internet-Browser. Hier siehst du die PnL-Daten, dein Echtzeit-Portfolio in USD, interaktive Candlesticks (Plotly) mit allen Buy/Sell Positions, sowie die Live ML-Confidences in bunten Dashboards visuell aufbereitet.

### 🧪 3. Historische 1H-Strategien & Stresstests (Backtesting)
Du willst nicht Tage warten, um zu sehen ob die KI gut ist, sondern sofort einen Hardcore Stresstest simulieren? 
```bash
python src/backtest.py
```
> Hier ist inzwischen ein Walkforward-Test sowie ein Regime-Stresstest eingebaut (Crash, Bullenmarkt, Bärenmarkt). Er zerreist die Strategie in Trümmer, um zu sehen ob sie reale Marktbedingungen – inkl. Slippage & Handelsgebühren – überlebt und zeichnet Charts für dich auf.

### 🧠 4. Die K.I. Pipeline trainieren (Machine Learning)
Der Kryptomarkt ändert sich ständig. Wenn du die Künstliche Intelligenz mit den allerneuesten Marktdaten füttern und vergleichen möchtest (Random Forest vs XGBoost):
```bash
python src/ml_model.py
```
> Das System lernt aus den neuesten Mustern, findet die lukrativsten Features automatisch dank `sklearn` Parameter-Tuning, speichert ihr frisch getuntes "Gehirn" im Ordner `models/best_model_1h.pkl` ab und erzeugt SHAP-Interaktions-Charts (`charts/shap_summary.png`) zur menschlichen Erklärbarkeit!

---

## ☁️ Server Deployment (Ubuntu / Oracle Cloud)

Du möchtest deinen Laptop zuklappen und den Bot 24/7 in der Cloud laufen lassen? Wir haben fertige Skripte geschrieben, um den Bot mit einem einzigen Befehl auf einen Ubuntu 22.04 Server (z.B. Oracle Cloud Free Tier) zu katapultieren.

### 1. Initiales Deployment
Das `deploy.sh` Script verbindet sich per SSH, installiert Python 3.11, kopiert den Code, installiert Requirements isoliert und registriert den Bot als unzerstörbaren `systemd`-Service in Linux.
```bash
# Ausführung lokal am Mac (Ersetze die IP mit deiner Server IP)
./deploy.sh ubuntu@123.45.67.89 ~/.ssh/id_rsa
```

### 2. Status & Logs auf dem Server checken
Wenn du dich später auf dem Server einloggst (SSH), kannst du den Bot mit Linux-Boardmitteln steuern:
```bash
sudo systemctl status trading-bot        # Zeigt ob der Bot läuft
sudo journalctl -u trading-bot -f        # Zeigt das Live-Terminal des Bots an
```

### 3. Updates nachschieben
Du hast am Mac etwas am Code geändert und auf GitHub gepusht? Logge dich per SSH auf deinem Wolkenserver ein und führe einfach das Update-Script aus. Es zieht alle Änderungen und bootet den Bot nahtlos neu:
```bash
/opt/tradingbot/update.sh
```

---

## 📂 Die Ordnerstruktur im Überblick
- `data/`: Hier werden die heruntergeladenen historischen Preisdaten zwischengespeichert (z.B. `BTC_USDT_1h.csv`).
- `src/`: Der tatsächliche Base-Code (Logik für ML-Features, MultiTimeframe-Strategie, Fetcher, Indikatoren, Bot, etc.).
- `models/`: Das "Gehirn" des Bots (die fertig trainierten K.I.-Modelle, z.B. `best_model_1h.pkl`).
- `logs/`: Hier landet die Datei `trades_1h.csv` – deine vollautomatisierte Buchhaltung aller vergangenen Transaktionen.
- `charts/`: Sämtliche Output-Bilder (Backtest-Equity-Curves, SHAP Feature-Importance-Grafiken).

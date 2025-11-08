#!/bin/bash
# Запуск COMBO Trading Bot с RL Position Advisor

echo "🤖 Starting COMBO Bot with RL Position Advisor"
echo "==============================================="
echo ""
echo "📊 Features enabled:"
echo "   ✅ IMBA Strategy (opens positions with TP1/TP2/TP3)"
echo "   ✅ RL Position Advisor (manages positions after TP2)"
echo "   ✅ Intelligent Trailing Stop (3% from peak)"
echo "   ✅ Early close on reversal detection (≥75% confidence)"
echo "   ✅ Emergency close on strong reversal (≥95% confidence)"
echo ""
echo "💎 Trading pairs (best models):"
echo "   • BNBUSDT  (Sharpe: 8.36) 🏆"
echo "   • ETHUSDT  (Sharpe: 5.58) 🏆"
echo "   • BTCUSDT  (Sharpe: 1.53) ✅"
echo ""
echo "⚙️  Settings:"
echo "   • Timeframe: 30m"
echo "   • Leverage: 5x"
echo "   • Risk per trade: 0.5%"
echo "   • RL close confidence: 75%"
echo "   • RL emergency confidence: 95%"
echo "   • Trailing distance: 3%"
echo ""

# Выбор режима
read -p "Select mode: [1] Testnet (safe) [2] Mainnet (real money): " mode

if [ "$mode" == "2" ]; then
    echo ""
    echo "⚠️  WARNING: Running on MAINNET with real money!"
    read -p "Are you sure? Type 'YES' to continue: " confirm

    if [ "$confirm" != "YES" ]; then
        echo "Cancelled."
        exit 0
    fi

    echo ""
    echo "🚀 Starting on MAINNET..."
    sleep 2

    python cli.py live \
        --timeframe 30m \
        --use-combo \
        --verbose \
        --symbols BNBUSDT,ETHUSDT,BTCUSDT \
        --leverage 5 \
        --risk-per-trade 0.3
else
    echo ""
    echo "🧪 Starting on TESTNET (safe testing)..."
    sleep 2

    python cli.py live \
        --timeframe 30m \
        --testnet \
        --use-combo \
        --verbose \
        --symbols BNBUSDT,ETHUSDT,BTCUSDT \
        --leverage 5 \
        --risk-per-trade 0.5
fi

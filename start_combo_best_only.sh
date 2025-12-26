#!/bin/bash
# Запуск COMBO бота только с лучшими обученными моделями

echo "🚀 Запуск COMBO Trading Bot с лучшими моделями"
echo "=============================================="
echo ""
echo "📊 Используемые пары:"
echo "   • BNBUSDT  (Win Rate: 56%, Sharpe: 8.36) 🏆"
echo "   • ETHUSDT  (Win Rate: 53%, Sharpe: 5.58) 🏆"
echo "   • BTCUSDT  (Win Rate: 56%, Sharpe: 1.53) ✅"
echo ""
echo "⏳ Запуск через 3 секунды..."
sleep 3

python cli.py live \
    --use-combo \
    --symbols BNBUSDT,ETHUSDT,BTCUSDT \
    --timeframe 30m \
    --leverage 5 \
    --risk-per-trade 0.5

# Альтернатива: только топ-2 пары
# python cli.py live --use-combo --symbols BNBUSDT,ETHUSDT

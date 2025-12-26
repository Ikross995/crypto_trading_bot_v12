#!/usr/bin/env python3
"""
Создаём тестовые позиции для проверки дашборда
"""

import json
from pathlib import Path
from datetime import datetime

def create_test_positions():
    """Создать тестовые открытые позиции."""

    # Загружаем текущий state
    state_file = Path('data/dashboard_state.json')

    if state_file.exists():
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
    else:
        print("❌ dashboard_state.json не найден! Сначала запусти fix_dashboard_data.py")
        return

    # Создаём тестовые позиции с TP и SL
    test_positions = [
        {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'entryPrice': 95000.0,
            'currentPrice': 96500.0,
            'quantity': 0.05,
            'pnl': 75.0,
            'pnlPct': 1.58,
            'leverage': 5,
            'stopLoss': 93500.0,  # -1.58% от входа
            'takeProfit': [
                {'price': 96900.0, 'amount': 0.02},  # TP1: +2%
                {'price': 98000.0, 'amount': 0.02},  # TP2: +3.16%
                {'price': 99500.0, 'amount': 0.01}   # TP3: +4.74%
            ]
        },
        {
            'symbol': 'ETHUSDT',
            'side': 'SHORT',
            'entryPrice': 3600.0,
            'currentPrice': 3550.0,
            'quantity': 1.5,
            'pnl': 75.0,
            'pnlPct': 1.39,
            'leverage': 3,
            'stopLoss': 3650.0,  # +1.39% (для SHORT)
            'takeProfit': [
                {'price': 3520.0, 'amount': 0.6},  # TP1: -2.22%
                {'price': 3480.0, 'amount': 0.5},  # TP2: -3.33%
                {'price': 3420.0, 'amount': 0.4}   # TP3: -5%
            ]
        },
        {
            'symbol': 'SOLUSDT',
            'side': 'LONG',
            'entryPrice': 145.0,
            'currentPrice': 143.5,
            'quantity': 10.0,
            'pnl': -15.0,
            'pnlPct': -1.03,
            'leverage': 2,
            'stopLoss': 142.0,  # -2.07%
            'takeProfit': [
                {'price': 147.5, 'amount': 4.0},   # TP1: +1.72%
                {'price': 150.0, 'amount': 3.5},   # TP2: +3.45%
                {'price': 153.0, 'amount': 2.5}    # TP3: +5.52%
            ]
        }
    ]

    # Обновляем state
    state['positions'] = test_positions
    state['openPositions'] = len(test_positions)
    state['lastUpdate'] = datetime.now().isoformat()

    # Обновляем equity с учётом unrealized PnL
    total_unrealized_pnl = sum(pos['pnl'] for pos in test_positions)
    state['equity'] = state['balance'] + total_unrealized_pnl

    # Сохраняем
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"✅ Создано {len(test_positions)} тестовых позиций:")
    for pos in test_positions:
        print(f"   {pos['symbol']} {pos['side']}: {pos['pnl']:+.2f} ({pos['pnlPct']:+.2f}%)")
    print(f"\n💰 Balance: ${state['balance']:.2f}")
    print(f"💎 Equity: ${state['equity']:.2f}")
    print(f"📊 Unrealized P&L: ${total_unrealized_pnl:+.2f}")
    print(f"\n🔄 Перезагрузи страницу в Telegram!")

if __name__ == '__main__':
    create_test_positions()

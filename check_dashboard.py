#!/usr/bin/env python3
"""
Проверка текущего состояния дашборда
"""

import json
from pathlib import Path
from datetime import datetime


def check_dashboard():
    """Проверить текущее состояние dashboard_state.json."""

    state_file = Path('data/dashboard_state.json')

    if not state_file.exists():
        print("❌ dashboard_state.json не найден!")
        return

    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)

    print("=" * 60)
    print("📊 ТЕКУЩЕЕ СОСТОЯНИЕ ДАШБОРДА")
    print("=" * 60)

    print(f"\n💰 Финансовые показатели:")
    print(f"   Balance:       ${state.get('balance', 0):,.2f}")
    print(f"   Equity:        ${state.get('equity', 0):,.2f}")
    print(f"   Initial:       ${state.get('initialEquity', 0):,.2f}")
    print(f"   Total P&L:     ${state.get('totalPnl', 0):+,.2f}")
    print(f"   ROI:           {state.get('roiPct', 0):+.2f}%")

    print(f"\n📈 Статистика торговли:")
    print(f"   Total Trades:  {state.get('totalTrades', 0)}")
    print(f"   Win Rate:      {state.get('winRate', 0):.1f}%")
    print(f"   Profit Factor: {state.get('profitFactor', 0):.2f}")
    print(f"   Sharpe Ratio:  {state.get('sharpeRatio', 0):.2f}")

    print(f"\n📍 Позиции:")
    positions = state.get('positions', [])
    print(f"   Открытых позиций: {len(positions)}")

    if positions:
        for i, pos in enumerate(positions, 1):
            print(f"\n   {i}. {pos.get('symbol')} {pos.get('side')}")
            print(f"      Entry:    ${pos.get('entryPrice', 0):,.4f}")
            print(f"      Current:  ${pos.get('currentPrice', 0):,.4f}")
            print(f"      Quantity: {pos.get('quantity', 0):.6f}")
            print(f"      P&L:      ${pos.get('pnl', 0):+,.2f} ({pos.get('pnlPct', 0):+.2f}%)")
            print(f"      Leverage: {pos.get('leverage', 1)}x")

            # Проверяем наличие TP/SL
            if 'stopLoss' in pos and pos['stopLoss']:
                print(f"      Stop Loss: ${pos['stopLoss']:,.4f}")
            else:
                print(f"      Stop Loss: ❌ НЕТ")

            if 'takeProfit' in pos and pos['takeProfit']:
                print(f"      Take Profit: ✅ {len(pos['takeProfit'])} уровней")
                for j, tp in enumerate(pos['takeProfit'], 1):
                    print(f"         TP{j}: ${tp.get('price', 0):,.4f} ({tp.get('amount', 0):.6f})")
            else:
                print(f"      Take Profit: ❌ НЕТ")
    else:
        print("   ℹ️  Нет открытых позиций")

    print(f"\n📊 История equity:")
    history = state.get('equityHistory', {})
    labels = history.get('labels', [])
    values = history.get('values', [])
    print(f"   Точек данных: {len(values)}")
    if values:
        print(f"   Первая:  ${values[0]:,.2f} ({labels[0] if labels else 'N/A'})")
        print(f"   Последняя: ${values[-1]:,.2f} ({labels[-1] if labels else 'N/A'})")

    print(f"\n🕐 Обновление:")
    last_update = state.get('lastUpdate', 'N/A')
    if last_update != 'N/A':
        try:
            dt = datetime.fromisoformat(last_update)
            print(f"   {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            print(f"   {last_update}")
    else:
        print(f"   {last_update}")

    print("\n" + "=" * 60)

    # Проверка реальности позиций
    if positions:
        print("\n⚠️  ВНИМАНИЕ:")
        has_tp_sl = all(pos.get('stopLoss') and pos.get('takeProfit') for pos in positions)
        if has_tp_sl:
            print("   ✅ Все позиции имеют TP/SL - скорее всего РЕАЛЬНЫЕ от бота")
        else:
            print("   ⚠️  Не все позиции имеют TP/SL - возможно ТЕСТОВЫЕ")

    print("\n")


if __name__ == '__main__':
    check_dashboard()

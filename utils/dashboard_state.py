"""
Dashboard State Manager
Сохраняет и загружает состояние дашборда для Web App API с WebSocket поддержкой
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable


class DashboardStateManager:
    """Управление состоянием дашборда для Web App с WebSocket."""

    def __init__(self, state_file: str = "data/dashboard_state.json"):
        """
        Инициализация менеджера состояния.

        Args:
            state_file: Путь к файлу состояния
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # WebSocket callback functions (set by webapp_server)
        self.on_trade_update: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        self.on_price_update: Optional[Callable] = None
        self.on_dashboard_update: Optional[Callable] = None

    def save_state(self, trading_engine) -> None:
        """
        Сохранить текущее состояние дашборда из торгового бота.

        Args:
            trading_engine: Объект LiveTradingEngine
        """
        try:
            # Собираем данные из торгового бота
            balance = getattr(trading_engine, 'equity_usdt', 0.0)
            initial = getattr(trading_engine, 'initial_equity', None)
            if initial is None:
                initial = getattr(trading_engine.config, 'paper_equity', 1000.0)

            # Рассчитываем метрики
            total_pnl = balance - initial
            roi_pct = ((balance - initial) / initial * 100) if initial > 0 else 0.0

            # Получаем позиции
            positions = getattr(trading_engine, 'active_positions', {})
            positions_list = []

            for symbol, pos_data in positions.items():
                try:
                    entry_price = pos_data.get('entry_price', 0)
                    current_price = pos_data.get('current_price', entry_price)
                    quantity = pos_data.get('quantity', 0)
                    side = pos_data.get('side', 'LONG')
                    leverage = pos_data.get('leverage', 1)

                    # Расчет P&L
                    if side == 'LONG':
                        pnl = (current_price - entry_price) * quantity * leverage
                    else:
                        pnl = (entry_price - current_price) * quantity * leverage

                    pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                    positions_list.append({
                        'symbol': symbol,
                        'side': side,
                        'pnl': round(pnl, 2),
                        'pnlPct': round(pnl_pct, 2),
                        'entry': round(entry_price, 2),
                        'current': round(current_price, 2),
                        'leverage': leverage
                    })
                except Exception as e:
                    print(f"Warning: Error processing position {symbol}: {e}")

            # Получаем статистику (если есть portfolio_tracker)
            total_trades = 0
            win_rate = 0.0
            profit_factor = 0.0

            if hasattr(trading_engine, 'portfolio_tracker') and trading_engine.portfolio_tracker:
                try:
                    stats = trading_engine.portfolio_tracker.get_stats()
                    if stats:
                        total_trades = stats.get('total_trades', 0)
                        winning_trades = stats.get('winning_trades', 0)
                        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
                        # Можно добавить больше метрик если есть
                except Exception:
                    pass

            # Equity history (последние 10 точек)
            equity_history_labels = []
            equity_history_values = []

            # Здесь можно загрузить историю из файла equity.csv или portfolio_history.json
            # Для примера создаем простую историю
            try:
                import pandas as pd
                equity_file = Path('data/equity_15m.csv')
                if equity_file.exists():
                    df = pd.read_csv(equity_file)
                    if not df.empty:
                        # Последние 20 точек
                        df_recent = df.tail(20)
                        equity_history_labels = df_recent['timestamp'].astype(str).tolist()
                        equity_history_values = df_recent['equity'].tolist()
            except Exception:
                pass

            # Если нет истории, создаем fake для демо
            if not equity_history_values:
                from datetime import timedelta
                now = datetime.now()
                for i in range(10):
                    time = now - timedelta(hours=10-i)
                    equity_history_labels.append(time.strftime('%H:%M'))
                    # Линейная интерполяция от initial до balance
                    progress = i / 9
                    value = initial + (balance - initial) * progress
                    equity_history_values.append(round(value, 2))

            # Формируем состояние
            state = {
                'balance': round(balance, 2),
                'equity': round(balance, 2),
                'totalPnl': round(total_pnl, 2),
                'roiPct': round(roi_pct, 2),
                'openPositions': len(positions),
                'totalTrades': total_trades,
                'winRate': round(win_rate, 1),
                'profitFactor': round(profit_factor, 2),
                'positions': positions_list,
                'equityHistory': {
                    'labels': equity_history_labels[-20:],  # Последние 20 точек
                    'values': equity_history_values[-20:]
                },
                'lastUpdate': datetime.now().isoformat()
            }

            # Сохраняем в файл
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            # Отправляем обновление через WebSocket если callback установлен
            if self.on_dashboard_update:
                try:
                    self.on_dashboard_update(state)
                except Exception as ws_error:
                    print(f"Warning: WebSocket dashboard update failed: {ws_error}")

        except Exception as e:
            print(f"Error saving dashboard state: {e}")

    def emit_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Отправить обновление о сделке через WebSocket.

        Args:
            trade_data: Данные сделки (symbol, side, price, quantity, pnl, etc.)
        """
        if self.on_trade_update:
            try:
                self.on_trade_update(trade_data)
            except Exception as e:
                print(f"Warning: WebSocket trade update failed: {e}")

    def emit_position(self, position_data: Dict[str, Any]) -> None:
        """
        Отправить обновление позиций через WebSocket.

        Args:
            position_data: Данные позиций
        """
        if self.on_position_update:
            try:
                self.on_position_update(position_data)
            except Exception as e:
                print(f"Warning: WebSocket position update failed: {e}")

    def emit_price(self, price_data: Dict[str, Any]) -> None:
        """
        Отправить обновление цены через WebSocket.

        Args:
            price_data: Данные о ценах (symbol, price, change, etc.)
        """
        if self.on_price_update:
            try:
                self.on_price_update(price_data)
            except Exception as e:
                print(f"Warning: WebSocket price update failed: {e}")

    def load_state(self) -> Dict[str, Any]:
        """
        Загрузить сохраненное состояние дашборда.

        Returns:
            Словарь с данными дашборда
        """
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading dashboard state: {e}")

        # Возвращаем пустое состояние
        return {
            'balance': 0.0,
            'equity': 0.0,
            'totalPnl': 0.0,
            'roiPct': 0.0,
            'openPositions': 0,
            'totalTrades': 0,
            'winRate': 0.0,
            'profitFactor': 0.0,
            'positions': [],
            'equityHistory': {'labels': [], 'values': []},
            'lastUpdate': None
        }

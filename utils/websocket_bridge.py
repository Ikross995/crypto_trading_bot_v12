"""
WebSocket Bridge
Связывает торговый движок с WebSocket сервером для real-time обновлений
"""

from typing import Optional
import threading


class WebSocketBridge:
    """Мост между торговым движком и WebSocket сервером."""

    _instance: Optional['WebSocketBridge'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern для глобального доступа."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Инициализация моста."""
        if not hasattr(self, 'initialized'):
            self.dashboard_state_manager = None
            self.initialized = True

    def set_dashboard_manager(self, manager):
        """
        Установить менеджер состояния дашборда.

        Args:
            manager: DashboardStateManager instance
        """
        self.dashboard_state_manager = manager
        print("✅ WebSocket Bridge: Dashboard manager connected")

    def setup_callbacks(self, webapp_module):
        """
        Настроить callback функции для WebSocket.

        Args:
            webapp_module: Модуль webapp_server с функциями emit_*
        """
        if self.dashboard_state_manager:
            self.dashboard_state_manager.on_dashboard_update = webapp_module.emit_dashboard_update
            self.dashboard_state_manager.on_trade_update = webapp_module.emit_trade_update
            self.dashboard_state_manager.on_position_update = webapp_module.emit_position_update
            self.dashboard_state_manager.on_price_update = webapp_module.emit_price_update
            print("✅ WebSocket Bridge: Callbacks connected")

    def emit_trade(self, trade_data: dict):
        """Отправить обновление о сделке."""
        if self.dashboard_state_manager:
            self.dashboard_state_manager.emit_trade(trade_data)

    def emit_position(self, position_data: dict):
        """Отправить обновление позиций."""
        if self.dashboard_state_manager:
            self.dashboard_state_manager.emit_position(position_data)

    def emit_price(self, price_data: dict):
        """Отправить обновление цены."""
        if self.dashboard_state_manager:
            self.dashboard_state_manager.emit_price(price_data)


# Global instance
ws_bridge = WebSocketBridge()

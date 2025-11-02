#!/usr/bin/env python3
"""
🚀 Enhanced Dashboard Launcher
Запуск улучшенного дашборда для AI торгового бота

Использование:
    python run_dashboard.py              # Запуск автономного дашборда
    python run_dashboard.py --demo       # Демо режим с тестовыми данными
    python run_dashboard.py --port 8080  # Запуск на порту 8080
"""

import asyncio
import argparse
import os
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

# Добавляем путь к проекту
import sys
sys.path.append(str(Path(__file__).parent))

from strategy.enhanced_dashboard import EnhancedDashboardGenerator, DashboardData


class DashboardLauncher:
    """Лаунчер для автономного запуска дашборда."""
    
    def __init__(self, port: int = 8000, demo_mode: bool = False):
        self.port = port
        self.demo_mode = demo_mode
        self.dashboard = EnhancedDashboardGenerator()
        
    async def run_standalone_dashboard(self):
        """Запуск автономного дашборда."""
        logger.info("🚀 [DASHBOARD] Starting Enhanced Trading Dashboard...")
        logger.info(f"📊 [DASHBOARD] Mode: {'Demo' if self.demo_mode else 'Live'}")
        logger.info(f"🌐 [DASHBOARD] Port: {self.port}")
        
        if self.demo_mode:
            await self._run_demo_mode()
        else:
            await self._run_live_mode()
    
    async def _run_demo_mode(self):
        """Демо режим с тестовыми данными."""
        logger.info("🎮 [DEMO] Generating demo data for dashboard...")
        
        # Создаем тестовые данные
        demo_data = self._generate_demo_data()
        
        # Добавляем в историю
        self.dashboard.data_history = [demo_data]
        
        # Генерируем дашборд
        dashboard_path = await self.dashboard.update_dashboard()
        
        if dashboard_path:
            logger.info(f"📊 [DEMO] Dashboard generated: {dashboard_path}")
            self._open_dashboard(dashboard_path)
        else:
            logger.error("❌ [DEMO] Failed to generate dashboard")
    
    async def _run_live_mode(self):
        """Живой режим - подключается к реальным данным."""
        logger.info("🔴 [LIVE] Attempting to connect to trading system...")
        
        # Пытаемся найти активный торговый движок
        try:
            # Здесь можно добавить подключение к реальным данным
            # из файлов логов, базы данных или API
            
            # Для примера, создаем базовые данные
            basic_data = self._generate_basic_data()
            self.dashboard.data_history = [basic_data]
            
            # Генерируем дашборд
            dashboard_path = await self.dashboard.update_dashboard()
            
            if dashboard_path:
                logger.info(f"📊 [LIVE] Dashboard generated: {dashboard_path}")
                logger.info("🔄 [LIVE] Dashboard will auto-refresh every 30 seconds")
                self._open_dashboard(dashboard_path)
                
                # Цикл обновления каждые 30 секунд
                while True:
                    await asyncio.sleep(30)
                    await self.dashboard.update_dashboard()
                    logger.debug("🔄 [LIVE] Dashboard refreshed")
            else:
                logger.error("❌ [LIVE] Failed to generate dashboard")
                
        except KeyboardInterrupt:
            logger.info("🛑 [LIVE] Dashboard stopped by user")
        except Exception as e:
            logger.error(f"❌ [LIVE] Error in live mode: {e}")
    
    def _generate_demo_data(self) -> DashboardData:
        """Генерирует демо данные для показа возможностей дашборда."""
        return DashboardData(
            timestamp=datetime.now(timezone.utc),
            
            # Impressive demo trading performance
            total_trades=247,
            winning_trades=156,
            losing_trades=91,
            win_rate=0.632,
            profit_factor=1.85,
            total_pnl=2847.65,
            best_trade=145.32,
            worst_trade=-89.45,
            avg_trade=11.54,
            max_drawdown=234.56,
            
            # Demo account info
            account_balance=12847.65,
            unrealized_pnl=89.23,
            margin_used=1250.00,
            available_balance=11597.65,
            equity=12936.88,
            margin_ratio=9.7,
            
            # Demo positions
            open_positions=3,
            total_position_value=3450.00,
            largest_position=1500.00,
            
            # Demo AI parameters
            confidence_threshold=1.247,
            position_size_multiplier=1.15,
            adaptations_count=23,
            learning_confidence=0.78,
            
            # Demo market data
            market_volatility=0.024,
            market_trend="bullish",
            price_change_24h=3.45,
            volume_24h=1250000000,
            
            # Demo system stats
            iteration=15247,
            uptime_hours=72.5,
            signals_generated=3421,
            signals_executed=247,
            execution_rate=0.072
        )
    
    def _generate_basic_data(self) -> DashboardData:
        """Генерирует базовые данные для live режима."""
        return DashboardData(
            timestamp=datetime.now(timezone.utc),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=1.0,
            total_pnl=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            avg_trade=0.0,
            max_drawdown=0.0,
            account_balance=1000.0,
            unrealized_pnl=0.0,
            margin_used=0.0,
            available_balance=1000.0,
            equity=1000.0,
            margin_ratio=0.0,
            open_positions=0,
            total_position_value=0.0,
            largest_position=0.0,
            confidence_threshold=1.2,
            position_size_multiplier=1.0,
            adaptations_count=0,
            learning_confidence=0.0,
            market_volatility=0.0,
            market_trend="neutral",
            price_change_24h=0.0,
            volume_24h=0.0,
            iteration=0,
            uptime_hours=0.0,
            signals_generated=0,
            signals_executed=0,
            execution_rate=0.0
        )
    
    def _open_dashboard(self, dashboard_path: str):
        """Открывает дашборд в браузере."""
        try:
            abs_path = Path(dashboard_path).resolve()
            file_url = f"file://{abs_path}"
            
            logger.info(f"🌐 [BROWSER] Opening dashboard: {file_url}")
            webbrowser.open(file_url)
            
            logger.info("💡 [INFO] Dashboard opened in browser")
            logger.info("💡 [INFO] Press Ctrl+C to stop")
            
        except Exception as e:
            logger.error(f"❌ [BROWSER] Failed to open dashboard: {e}")
            logger.info(f"📂 [MANUAL] Please open manually: {dashboard_path}")


async def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Enhanced Trading Dashboard Launcher")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with test data")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server (default: 8000)")
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Запуск дашборда
    launcher = DashboardLauncher(port=args.port, demo_mode=args.demo)
    await launcher.run_standalone_dashboard()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
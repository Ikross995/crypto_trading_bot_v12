#!/usr/bin/env python3
"""
Enhanced Trading Dashboard - Improved Real-time Analytics
Создает мощный интерактивный дашборд с расширенной статистикой
"""

# Отключаем TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключаем oneDNN

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import asyncio
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from loguru import logger


@dataclass
class DashboardData:
    """Расширенные данные для дашборда."""
    timestamp: datetime

    # Trading Performance
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    best_trade: float
    worst_trade: float
    avg_trade: float
    max_drawdown: float

    # Account Info
    account_balance: float
    unrealized_pnl: float
    margin_used: float
    available_balance: float
    equity: float
    margin_ratio: float

    # Positions
    open_positions: int
    total_position_value: float
    largest_position: float

    # AI Learning
    confidence_threshold: float
    position_size_multiplier: float
    adaptations_count: int
    learning_confidence: float

    # Market Data
    market_volatility: float
    market_trend: str
    price_change_24h: float
    volume_24h: float

    # System Stats
    iteration: int
    uptime_hours: float
    signals_generated: int
    signals_executed: int
    execution_rate: float

    # GRU Predictions
    gru_prediction: Optional[float] = None
    gru_direction: Optional[str] = None
    gru_confidence: Optional[float] = None
    gru_current_price: Optional[float] = None

    # ML Learning System (Enhanced AI)
    ml_samples_collected: int = 0
    ml_samples_needed: int = 50
    ml_prediction_accuracy: float = 0.0
    ml_avg_pnl_prediction: float = 0.0
    ml_win_probability: float = 0.0

    # Extended Stats
    recent_trades: List[Dict] = None  # Last 10 trades
    open_positions_details: List[Dict] = None  # Detailed position info
    sharpe_ratio: float = 0.0
    avg_hold_time: float = 0.0  # In hours
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0

    # Risk Metrics
    total_margin_used: float = 0.0
    margin_usage_pct: float = 0.0
    free_margin: float = 0.0
    largest_position_margin: float = 0.0

    # Performance Metrics
    roi_pct: float = 0.0  # Return on Investment %
    initial_balance: float = 1000.0  # Starting balance
    win_streak: int = 0  # Current winning streak
    loss_streak: int = 0  # Current losing streak
    max_win_streak: int = 0  # Best winning streak
    max_loss_streak: int = 0  # Worst losing streak
    hourly_pnl: float = 0.0  # P&L per hour
    risk_score: float = 0.0  # Risk score 0-100 (lower is better)

    def __post_init__(self):
        """Initialize list fields after creation."""
        if self.recent_trades is None:
            self.recent_trades = []
        if self.open_positions_details is None:
            self.open_positions_details = []


class EnhancedDashboardGenerator:
    """Генератор улучшенного дашборда с богатой статистикой."""
    
    def __init__(self, output_dir: str = "data/learning_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_file = self.output_dir / "enhanced_dashboard.html"
        self.history_file = self.output_dir / "dashboard_history.json"

        # История данных для графиков
        self.data_history: List[DashboardData] = []

        # Загружаем историю при старте
        self._load_history()

        logger.info(f"📊 [ENHANCED_DASHBOARD] Initialized: {self.dashboard_file}")
        logger.info(f"📊 [DASHBOARD_HISTORY] Loaded {len(self.data_history)} historical data points")
    
    async def update_dashboard(self, trading_engine=None, adaptive_learning=None, enhanced_ai=None) -> str:
        """Обновляет дашборд с новыми данными."""
        try:
            # Собираем данные
            dashboard_data = await self._collect_dashboard_data(trading_engine, adaptive_learning, enhanced_ai)

            # Добавляем в историю
            self.data_history.append(dashboard_data)

            # Ограничиваем историю последними 500 точками
            if len(self.data_history) > 500:
                self.data_history = self.data_history[-500:]

            # Сохраняем историю в JSON
            self._save_history()

            # Генерируем HTML
            html_content = self._generate_enhanced_html()

            # Сохраняем
            with open(self.dashboard_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"📊 [ENHANCED_DASHBOARD] Updated: {self.dashboard_file}")
            return str(self.dashboard_file)

        except Exception as e:
            logger.error(f"❌ [ENHANCED_DASHBOARD] Update failed: {e}")
            return ""

    async def _collect_dashboard_data(self, trading_engine=None, adaptive_learning=None, enhanced_ai=None) -> DashboardData:
        """Собирает данные для дашборда."""
        now = datetime.now(timezone.utc)
        
        # Инициализация значений по умолчанию
        data = DashboardData(
            timestamp=now,
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
            iteration=getattr(trading_engine, 'iteration', 0) if trading_engine else 0,
            uptime_hours=0.0,
            signals_generated=0,
            signals_executed=0,
            execution_rate=0.0
        )
        
        # Данные торгового движка
        if trading_engine:
            data = await self._get_trading_engine_data(trading_engine, data)

        # Используем Portfolio Tracker если доступен (приоритет!)
        if trading_engine and hasattr(trading_engine, 'portfolio_tracker') and trading_engine.portfolio_tracker:
            data = await self._get_portfolio_tracker_data(trading_engine.portfolio_tracker, data)

        # Данные системы обучения
        if adaptive_learning:
            data = await self._get_learning_data(adaptive_learning, data)

        # Данные Enhanced AI (ML система)
        if enhanced_ai:
            data = await self._get_enhanced_ai_data(enhanced_ai, data)

        # GRU предсказания из trading_engine
        if trading_engine and hasattr(trading_engine, 'last_gru_prediction'):
            data = await self._get_gru_data(trading_engine, data)

        return data
    
    async def _get_trading_engine_data(self, engine, data: DashboardData) -> DashboardData:
        """Получает данные от торгового движка."""
        try:
            # Основные параметры
            data.iteration = getattr(engine, 'iteration', 0)
            
            # Позиции
            active_positions = getattr(engine, 'active_positions', {})
            data.open_positions = len(active_positions)
            
            if active_positions:
                position_values = []
                for symbol, pos in active_positions.items():
                    value = pos.get('quantity', 0) * pos.get('entry_price', 0)
                    position_values.append(value)
                
                data.total_position_value = sum(position_values)
                data.largest_position = max(position_values) if position_values else 0.0
            
            # Данные аккаунта через клиент
            if hasattr(engine, 'client') and engine.client:
                try:
                    balance = float(engine.client.get_balance())
                    data.account_balance = balance
                    data.available_balance = balance
                    data.equity = balance
                    
                    # Позиции с биржи
                    positions = engine.client.get_positions()
                    total_unrealized = 0.0
                    total_margin = 0.0
                    
                    for pos in positions:
                        pos_amt = float(pos.get('positionAmt', 0))
                        if abs(pos_amt) > 0:
                            total_unrealized += float(pos.get('unRealizedPnl', 0))
                            total_margin += float(pos.get('initialMargin', 0))
                    
                    data.unrealized_pnl = total_unrealized
                    data.margin_used = total_margin
                    data.equity = balance + total_unrealized
                    
                    if balance > 0:
                        data.margin_ratio = (total_margin / balance) * 100
                    
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Failed to get account data: {e}")
            
            # Рыночные данные
            if hasattr(engine, 'client') and engine.client:
                try:
                    ticker = engine.client.get_24hr_ticker(symbol="BTCUSDT")
                    data.price_change_24h = float(ticker.get('priceChangePercent', 0))
                    data.volume_24h = float(ticker.get('volume', 0))
                    data.market_volatility = abs(data.price_change_24h) / 100
                    
                    if data.price_change_24h > 2:
                        data.market_trend = "bullish"
                    elif data.price_change_24h < -2:
                        data.market_trend = "bearish"
                    else:
                        data.market_trend = "neutral"
                        
                except Exception as e:
                    logger.debug(f"[DASHBOARD] Failed to get market data: {e}")
            
        except Exception as e:
            logger.debug(f"[DASHBOARD] Error getting trading engine data: {e}")
        
        return data
    
    async def _get_learning_data(self, adaptive_learning, data: DashboardData) -> DashboardData:
        """Получает данные от системы обучения."""
        try:
            # История сделок
            if hasattr(adaptive_learning, 'trades_history'):
                trades = adaptive_learning.trades_history
                data.total_trades = len(trades)
                
                if trades:
                    winning_trades = [t for t in trades if t.pnl > 0]
                    losing_trades = [t for t in trades if t.pnl < 0]
                    
                    data.winning_trades = len(winning_trades)
                    data.losing_trades = len(losing_trades)
                    
                    if data.total_trades > 0:
                        data.win_rate = len(winning_trades) / data.total_trades
                    
                    # PnL статистика
                    pnls = [t.pnl for t in trades]
                    data.total_pnl = sum(pnls)
                    data.best_trade = max(pnls) if pnls else 0.0
                    data.worst_trade = min(pnls) if pnls else 0.0
                    data.avg_trade = sum(pnls) / len(pnls) if pnls else 0.0
                    
                    # Profit factor
                    gross_profit = sum([p for p in pnls if p > 0])
                    gross_loss = abs(sum([p for p in pnls if p < 0]))
                    
                    if gross_loss > 0:
                        data.profit_factor = gross_profit / gross_loss
                    else:
                        data.profit_factor = float('inf') if gross_profit > 0 else 1.0
                    
                    # Максимальная просадка
                    cumulative_pnl = np.cumsum(pnls)
                    running_max = np.maximum.accumulate(cumulative_pnl)
                    drawdowns = cumulative_pnl - running_max
                    data.max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # Параметры обучения
            if hasattr(adaptive_learning, 'current_params'):
                params = adaptive_learning.current_params
                data.confidence_threshold = params.get('confidence_threshold', 1.2)
                data.position_size_multiplier = params.get('position_size_multiplier', 1.0)
            
            # Адаптации
            if hasattr(adaptive_learning, 'adaptation_count'):
                data.adaptations_count = adaptive_learning.adaptation_count
            
            # Уверенность обучения
            if hasattr(adaptive_learning, 'learning_confidence'):
                data.learning_confidence = adaptive_learning.learning_confidence
            
        except Exception as e:
            logger.debug(f"[DASHBOARD] Error getting learning data: {e}")

        return data

    async def _get_portfolio_tracker_data(self, portfolio_tracker, data: DashboardData) -> DashboardData:
        """Получает данные от Portfolio Tracker (приоритет над другими источниками!)."""
        try:
            # Получаем полную статистику портфеля
            stats = portfolio_tracker.get_portfolio_stats()

            # Баланс и капитал
            data.account_balance = stats.total_balance
            data.available_balance = stats.available_balance
            data.margin_used = stats.margin_balance
            data.unrealized_pnl = stats.total_unrealized_pnl
            data.equity = stats.total_balance + stats.total_unrealized_pnl
            data.margin_ratio = (stats.margin_balance / stats.total_balance * 100) if stats.total_balance > 0 else 0.0

            # Позиции
            data.open_positions = len(stats.open_positions)
            if stats.open_positions:
                position_values = [pos.notional_value for pos in stats.open_positions]
                data.total_position_value = sum(position_values)
                data.largest_position = max(position_values) if position_values else 0.0

            # Trading Performance (ГЛАВНОЕ!)
            data.total_trades = stats.total_trades
            data.winning_trades = stats.winning_trades
            data.losing_trades = stats.losing_trades
            data.win_rate = stats.win_rate

            # PnL статистика
            data.total_pnl = stats.daily_pnl  # Или можно взять cumulative
            data.best_trade = getattr(stats, 'best_trade', 0.0)
            data.worst_trade = getattr(stats, 'worst_trade', 0.0)
            data.avg_trade = getattr(stats, 'avg_trade', 0.0)

            # Risk metrics
            data.max_drawdown = stats.max_drawdown
            data.profit_factor = getattr(stats, 'profit_factor', 1.0)
            data.sharpe_ratio = stats.sharpe_ratio

            # Performance periods
            data.daily_pnl = stats.daily_pnl
            data.weekly_pnl = stats.weekly_pnl
            data.monthly_pnl = stats.monthly_pnl

            # Детали открытых позиций
            data.open_positions_details = []
            for pos in stats.open_positions[:10]:  # Последние 10
                # Рассчитываем реальные значения
                notional_value = pos.notional_value if pos.notional_value else (pos.quantity * pos.current_price)
                leverage = pos.leverage if pos.leverage and pos.leverage > 0 else 1.0
                margin_used = notional_value / leverage  # Реальный залог

                data.open_positions_details.append({
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'quantity': pos.quantity,
                    'notional': notional_value,
                    'margin_used': margin_used,
                    'pnl': pos.unrealized_pnl,
                    'pnl_pct': pos.unrealized_pnl_pct,
                    'leverage': leverage,
                    'liquidation_price': getattr(pos, 'liquidation_price', None)
                })

            # Рассчитываем risk metrics
            if data.open_positions_details:
                data.total_margin_used = sum(pos['margin_used'] for pos in data.open_positions_details)
                data.largest_position_margin = max((pos['margin_used'] for pos in data.open_positions_details), default=0.0)

                # Margin usage % от account balance
                if data.account_balance > 0:
                    data.margin_usage_pct = (data.total_margin_used / data.account_balance) * 100
                    data.free_margin = data.account_balance - data.total_margin_used
                else:
                    data.margin_usage_pct = 0.0
                    data.free_margin = 0.0

            # Performance Metrics
            # ROI % = (Current Balance - Initial Balance) / Initial Balance * 100
            if data.account_balance > 0:
                data.roi_pct = ((data.account_balance - data.initial_balance) / data.initial_balance) * 100

            # Win/Loss Streaks
            if hasattr(portfolio_tracker, 'trade_history') and portfolio_tracker.trade_history:
                trades = list(portfolio_tracker.trade_history)
                if trades:
                    # Calculate current streak
                    current_streak = 0
                    for trade in reversed(trades):
                        pnl = getattr(trade, 'pnl', 0.0)
                        if pnl > 0:
                            if data.win_streak >= 0:
                                data.win_streak += 1
                                data.loss_streak = 0
                            else:
                                break
                        elif pnl < 0:
                            if data.loss_streak >= 0:
                                data.loss_streak += 1
                                data.win_streak = 0
                            else:
                                break
                        else:
                            break

                    # Calculate max streaks
                    win_streak = 0
                    loss_streak = 0
                    for trade in trades:
                        pnl = getattr(trade, 'pnl', 0.0)
                        if pnl > 0:
                            win_streak += 1
                            loss_streak = 0
                            data.max_win_streak = max(data.max_win_streak, win_streak)
                        elif pnl < 0:
                            loss_streak += 1
                            win_streak = 0
                            data.max_loss_streak = max(data.max_loss_streak, loss_streak)

            # Hourly PnL (based on uptime)
            if data.uptime_hours > 0:
                data.hourly_pnl = data.total_pnl / data.uptime_hours

            # Risk Score (0-100, lower is better)
            # Factors: margin usage, open positions, max drawdown, volatility
            risk_score = 0
            risk_score += min(data.margin_usage_pct, 50)  # Max 50 points from margin
            risk_score += min(data.open_positions * 5, 20)  # Max 20 points from positions
            risk_score += min(abs(data.max_drawdown) / 10, 20)  # Max 20 points from drawdown
            risk_score += min(data.market_volatility * 100, 10)  # Max 10 points from volatility
            data.risk_score = min(risk_score, 100)

            # Последние сделки (из portfolio_tracker history если есть)
            if hasattr(portfolio_tracker, 'trade_history') and portfolio_tracker.trade_history:
                data.recent_trades = []
                for trade in list(portfolio_tracker.trade_history)[-10:]:  # Последние 10
                    data.recent_trades.append({
                        'symbol': getattr(trade, 'symbol', 'N/A'),
                        'side': getattr(trade, 'side', 'N/A'),
                        'pnl': getattr(trade, 'pnl', 0.0),
                        'pnl_pct': getattr(trade, 'pnl_pct', 0.0),
                        'timestamp': getattr(trade, 'timestamp', datetime.now()).strftime('%H:%M:%S')
                    })

            logger.debug(f"[DASHBOARD_PORTFOLIO] Loaded: {data.total_trades} trades, "
                        f"{data.winning_trades}W/{data.losing_trades}L, "
                        f"Win Rate: {data.win_rate*100:.1f}%, "
                        f"Positions: {len(data.open_positions_details)}")

        except Exception as e:
            logger.debug(f"[DASHBOARD] Error getting portfolio tracker data: {e}")
            import traceback
            logger.debug(f"[DASHBOARD] Traceback: {traceback.format_exc()}")

        return data

    async def _get_enhanced_ai_data(self, enhanced_ai, data: DashboardData) -> DashboardData:
        """Получает данные от Enhanced AI (ML система)."""
        try:
            # Количество собранных samples для ML обучения
            if hasattr(enhanced_ai, 'ml_trainer') and hasattr(enhanced_ai.ml_trainer, 'training_data'):
                training_data = enhanced_ai.ml_trainer.training_data
                data.ml_samples_collected = len(training_data) if training_data else 0

            # Можно также проверить trades_history
            if hasattr(enhanced_ai, 'trades_history'):
                completed_trades = [t for t in enhanced_ai.trades_history if t.exit_reason != "pending"]
                data.ml_samples_collected = len(completed_trades)

            # ML метрики
            if hasattr(enhanced_ai, 'enhanced_metrics'):
                metrics = enhanced_ai.enhanced_metrics
                data.ml_prediction_accuracy = metrics.get('prediction_accuracy', 0.0)
                data.ml_avg_pnl_prediction = metrics.get('avg_pnl_prediction', 0.0)

            logger.debug(f"[DASHBOARD_ML] Collected {data.ml_samples_collected}/{data.ml_samples_needed} ML samples")

        except Exception as e:
            logger.debug(f"[DASHBOARD] Error getting Enhanced AI data: {e}")

        return data

    async def _get_gru_data(self, trading_engine, data: DashboardData) -> DashboardData:
        """Получает данные GRU предсказаний."""
        try:
            if hasattr(trading_engine, 'last_gru_prediction') and trading_engine.last_gru_prediction:
                gru_pred = trading_engine.last_gru_prediction

                # GRU предсказание может быть dict с ключами: predicted_price, direction, confidence, current_price
                if isinstance(gru_pred, dict):
                    data.gru_prediction = gru_pred.get('predicted_price')
                    data.gru_direction = gru_pred.get('direction')
                    data.gru_confidence = gru_pred.get('confidence')
                    data.gru_current_price = gru_pred.get('current_price')

                    logger.debug(f"[DASHBOARD_GRU] Prediction: ${data.gru_prediction:.2f}, "
                                f"Direction: {data.gru_direction}, Confidence: {data.gru_confidence:.1f}%")

        except Exception as e:
            logger.debug(f"[DASHBOARD] Error getting GRU data: {e}")

        return data

    def _generate_enhanced_html(self) -> str:
        """Генерирует улучшенный HTML дашборд."""
        if not self.data_history:
            return self._generate_empty_dashboard()
        
        latest = self.data_history[-1]
        
        # Подготовка данных для графиков
        timestamps = [d.timestamp.strftime('%H:%M:%S') for d in self.data_history[-100:]]
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Enhanced Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            color: #ffffff; 
            min-height: 100vh;
        }}
        
        .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
        
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; 
            border-radius: 15px; 
            margin-bottom: 30px; 
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
            text-align: center;
        }}
        
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
        .header p {{ font-size: 1.2em; opacity: 0.9; }}
        
        .status-bar {{ 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            background: rgba(255,255,255,0.1); 
            padding: 15px 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            -webkit-backdrop-filter: blur(10px);
            backdrop-filter: blur(10px);
        }}
        
        .status-item {{ text-align: center; }}
        .status-value {{ font-size: 1.5em; font-weight: bold; color: #00ff88; }}
        .status-label {{ font-size: 0.9em; opacity: 0.8; }}
        
        .main-grid {{ 
            display: grid; 
            grid-template-columns: 1fr 1fr 1fr; 
            gap: 25px; 
            margin-bottom: 30px; 
        }}
        
        .performance-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px; 
        }}
        
        .card {{ 
            background: rgba(255,255,255,0.05); 
            border: 1px solid rgba(255,255,255,0.1);
            padding: 25px;
            border-radius: 15px;
            -webkit-backdrop-filter: blur(10px);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{ 
            transform: translateY(-5px); 
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }}
        
        .card h3 {{ 
            margin-bottom: 20px; 
            color: #667eea; 
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .metric {{ 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 15px; 
            padding: 10px 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }}
        
        .metric-label {{ color: #cccccc; }}
        .metric-value {{ 
            font-weight: bold; 
            font-size: 1.2em;
        }}
        
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4757; }}
        .neutral {{ color: #ffa502; }}
        
        .chart-container {{ 
            background: rgba(255,255,255,0.05); 
            border: 1px solid rgba(255,255,255,0.1);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            -webkit-backdrop-filter: blur(10px);
            backdrop-filter: blur(10px);
        }}
        
        .chart-grid {{ 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 25px; 
            margin-bottom: 30px; 
        }}
        
        .progress-ring {{ 
            transform: rotate(-90deg); 
            margin: 0 auto; 
            display: block;
        }}
        
        .progress-ring-circle {{ 
            transition: stroke-dasharray 0.35s; 
            fill: transparent; 
            stroke: #667eea; 
            stroke-width: 4;
        }}
        
        .footer {{ 
            text-align: center; 
            padding: 20px; 
            opacity: 0.7; 
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }}
        
        .alert {{ 
            padding: 15px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
            border-left: 4px solid;
        }}
        
        .alert-success {{ background: rgba(0,255,136,0.1); border-color: #00ff88; }}
        .alert-warning {{ background: rgba(255,165,2,0.1); border-color: #ffa502; }}
        .alert-danger {{ background: rgba(255,71,87,0.1); border-color: #ff4757; }}
        
        .live-indicator {{ 
            display: inline-block; 
            width: 8px; 
            height: 8px; 
            background: #00ff88; 
            border-radius: 50%; 
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.3; }}
            100% {{ opacity: 1; }}
        }}
        
        .trade-history {{ 
            max-height: 300px; 
            overflow-y: auto; 
            padding: 10px;
        }}
        
        .trade-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            margin-bottom: 5px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
        }}

        /* Table Styles */
        .table-container {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            overflow-x: auto;
        }}

        .table-container h3 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.4em;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }}

        .data-table thead {{
            background: rgba(102,126,234,0.2);
        }}

        .data-table th {{
            padding: 12px 15px;
            text-align: left;
            color: #fff;
            font-weight: 600;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }}

        .data-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            color: #ccc;
        }}

        .data-table tbody tr {{
            transition: background 0.2s ease;
        }}

        .data-table tbody tr:hover {{
            background: rgba(255,255,255,0.05);
        }}

        .badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-long {{
            background: rgba(0,255,136,0.2);
            color: #00ff88;
            border: 1px solid #00ff88;
        }}

        .badge-short {{
            background: rgba(255,71,87,0.2);
            color: #ff4757;
            border: 1px solid #ff4757;
        }}

        /* ==================== Mobile Responsive Styles ==================== */
        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            .header {{ padding: 15px; margin-bottom: 15px; }}
            .header h1 {{ font-size: 1.5em; }}
            .header p {{ font-size: 0.9em; }}
            .status-bar {{ flex-wrap: wrap; padding: 10px 15px; gap: 10px; }}
            .status-item {{ flex: 1 1 45%; min-width: 100px; }}
            .status-value {{ font-size: 1.2em; }}
            .status-label {{ font-size: 0.75em; }}
            .main-grid {{ grid-template-columns: 1fr; gap: 15px; }}
            .chart-grid {{ grid-template-columns: 1fr; gap: 15px; }}
            .performance-grid {{ grid-template-columns: 1fr; gap: 15px; }}
            .card {{ padding: 15px; }}
            .card h3 {{ font-size: 1.1em; }}
            .metric {{ padding: 8px 10px; margin-bottom: 10px; }}
            .metric-label {{ font-size: 0.85em; }}
            .metric-value {{ font-size: 1em; }}
            .chart-container {{ padding: 15px; margin-bottom: 15px; }}
            .card:hover {{ transform: none; }}
            .table-container {{ padding: 15px; }}
            .data-table {{ font-size: 0.8em; }}
        }}

        @media (max-width: 480px) {{
            .container {{ padding: 5px; }}
            .header {{ padding: 12px; }}
            .header h1 {{ font-size: 1.2em; }}
            .header p {{ font-size: 0.75em; }}
            .status-bar {{ flex-direction: column; padding: 10px; }}
            .status-item {{ flex: 1 1 100%; padding: 5px 0; }}
            .card {{ padding: 12px; }}
            .card h3 {{ font-size: 1em; }}
            .table-container {{ padding: 10px; overflow-x: scroll; }}
            .data-table {{ font-size: 0.7em; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 AI Trading Bot - Enhanced Dashboard</h1>
            <p>Advanced Real-time Analytics & Performance Monitoring</p>
            <p><span class="live-indicator"></span> Live • Last Update: {latest.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} • Iteration: {latest.iteration:,}</p>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-value">{'Running' if latest.iteration > 0 else 'Standby'}</div>
                <div class="status-label">Bot Status</div>
            </div>
            <div class="status-item">
                <div class="status-value">{latest.uptime_hours:.1f}h</div>
                <div class="status-label">Uptime</div>
            </div>
            <div class="status-item">
                <div class="status-value">{latest.signals_generated:,}</div>
                <div class="status-label">Signals Generated</div>
            </div>
            <div class="status-item">
                <div class="status-value">{latest.execution_rate:.1%}</div>
                <div class="status-label">Execution Rate</div>
            </div>
            <div class="status-item">
                <div class="status-value">{latest.market_trend.title()}</div>
                <div class="status-label">Market Trend</div>
            </div>
        </div>
        
        <div class="main-grid">
            <div class="card">
                <h3>💰 Account Overview</h3>
                <div class="metric">
                    <span class="metric-label">💵 Account Balance</span>
                    <span class="metric-value">${latest.account_balance:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">⚡ Unrealized PnL</span>
                    <span class="metric-value {'positive' if latest.unrealized_pnl >= 0 else 'negative'}">${latest.unrealized_pnl:+,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">💎 Total Equity</span>
                    <span class="metric-value">${latest.equity:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">🛡️ Margin Used</span>
                    <span class="metric-value">${latest.margin_used:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">📊 Margin Ratio</span>
                    <span class="metric-value {'neutral' if latest.margin_ratio < 50 else 'negative'}">{latest.margin_ratio:.1f}%</span>
                </div>
            </div>
            
            <div class="card">
                <h3>📈 Trading Performance</h3>
                <div class="metric">
                    <span class="metric-label">🎯 Total Trades</span>
                    <span class="metric-value">{latest.total_trades:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">🏆 Win Rate</span>
                    <span class="metric-value {'positive' if latest.win_rate >= 0.5 else 'negative'}">{latest.win_rate:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">💰 Total PnL</span>
                    <span class="metric-value {'positive' if latest.total_pnl >= 0 else 'negative'}">${latest.total_pnl:+,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">⚡ Profit Factor</span>
                    <span class="metric-value {'positive' if latest.profit_factor >= 1.0 else 'negative'}">{latest.profit_factor:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">📉 Max Drawdown</span>
                    <span class="metric-value negative">${latest.max_drawdown:,.2f}</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🤖 AI Learning Status</h3>
                <div class="metric">
                    <span class="metric-label">🎛️ Confidence Threshold</span>
                    <span class="metric-value">{latest.confidence_threshold:.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">📏 Position Multiplier</span>
                    <span class="metric-value">{latest.position_size_multiplier:.2f}x</span>
                </div>
                <div class="metric">
                    <span class="metric-label">🔧 Adaptations</span>
                    <span class="metric-value">{latest.adaptations_count:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">🧠 Learning Confidence</span>
                    <span class="metric-value">{latest.learning_confidence:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">📊 Market Volatility</span>
                    <span class="metric-value">{latest.market_volatility:.1%}</span>
                </div>
            </div>
        </div>

        <!-- 📊 Extended Trading Stats -->
        <div class="main-grid">
            <div class="card">
                <h3>💎 ROI Performance</h3>
                <div class="metric">
                    <span class="metric-label">📈 Return on Investment</span>
                    <span class="metric-value {'positive' if latest.roi_pct >= 0 else 'negative'}" style="font-size: 1.8em;">{latest.roi_pct:+.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">💵 Initial → Current</span>
                    <span class="metric-value">${latest.initial_balance:,.2f} → ${latest.account_balance:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">⚡ Hourly P&L</span>
                    <span class="metric-value {'positive' if latest.hourly_pnl >= 0 else 'negative'}">${latest.hourly_pnl:+,.2f}/hr</span>
                </div>
            </div>

            <div class="card">
                <h3>🔥 Win/Loss Streaks</h3>
                <div class="metric">
                    <span class="metric-label">{'🟢 Current Win Streak' if latest.win_streak > 0 else '🔴 Current Loss Streak' if latest.loss_streak > 0 else '⚪ No Active Streak'}</span>
                    <span class="metric-value {'positive' if latest.win_streak > 0 else 'negative' if latest.loss_streak > 0 else ''}" style="font-size: 2em;">{max(latest.win_streak, latest.loss_streak)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">🏆 Best Win Streak</span>
                    <span class="metric-value">{latest.max_win_streak}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">💔 Worst Loss Streak</span>
                    <span class="metric-value">{latest.max_loss_streak}</span>
                </div>
            </div>

            <div class="card">
                <h3>⚠️ Risk Score</h3>
                <div class="metric">
                    <span class="metric-label">🎯 Overall Risk</span>
                    <span class="metric-value {'positive' if latest.risk_score < 30 else 'negative' if latest.risk_score > 70 else ''}" style="font-size: 2em;">{latest.risk_score:.0f}/100</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <div style="background: rgba(255,255,255,0.1); height: 20px; border-radius: 10px; overflow: hidden; margin-top: 5px;">
                        <div style="background: {'linear-gradient(90deg, #00ff88 0%, #00d9ff 100%)' if latest.risk_score < 30 else 'linear-gradient(90deg, #ffa502 0%, #ff4757 100%)' if latest.risk_score > 70 else 'linear-gradient(90deg, #ffa502 0%, #00ff88 100%)'}; height: 100%; width: {latest.risk_score:.1f}%; transition: width 0.3s;"></div>
                    </div>
                    <span class="metric-value {'positive' if latest.risk_score < 30 else 'negative' if latest.risk_score > 70 else ''}">{'🟢 LOW RISK' if latest.risk_score < 30 else '🔴 HIGH RISK' if latest.risk_score > 70 else '🟡 MEDIUM RISK'}</span>
                </div>
            </div>
        </div>

        <!-- 📈 Trade Stats -->
        <div class="main-grid">
            <div class="card">
                <h3>🏆 Best Trade</h3>
                <div class="metric">
                    <span class="metric-label">💰 Profit</span>
                    <span class="metric-value positive">${latest.best_trade:,.2f}</span>
                </div>
            </div>
            <div class="card">
                <h3>📉 Worst Trade</h3>
                <div class="metric">
                    <span class="metric-label">💸 Loss</span>
                    <span class="metric-value negative">${latest.worst_trade:,.2f}</span>
                </div>
            </div>
            <div class="card">
                <h3>⚖️ Sharpe Ratio</h3>
                <div class="metric">
                    <span class="metric-label">📊 Risk-Adjusted Return</span>
                    <span class="metric-value {'positive' if latest.sharpe_ratio > 1 else 'negative'}">{latest.sharpe_ratio:.2f}</span>
                </div>
            </div>
        </div>

        <!-- 📈 Performance by Period -->
        <div class="main-grid">
            <div class="card">
                <h3>📅 Daily Performance</h3>
                <div class="metric">
                    <span class="metric-label">Today's P&L</span>
                    <span class="metric-value {'positive' if latest.daily_pnl >= 0 else 'negative'}">${latest.daily_pnl:+,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Daily %</span>
                    <span class="metric-value {'positive' if latest.daily_pnl >= 0 else 'negative'}">{(latest.daily_pnl/latest.account_balance*100 if latest.account_balance > 0 else 0):+.2f}%</span>
                </div>
            </div>
            <div class="card">
                <h3>📅 Weekly Performance</h3>
                <div class="metric">
                    <span class="metric-label">Week P&L</span>
                    <span class="metric-value {'positive' if latest.weekly_pnl >= 0 else 'negative'}">${latest.weekly_pnl:+,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Weekly %</span>
                    <span class="metric-value {'positive' if latest.weekly_pnl >= 0 else 'negative'}">{(latest.weekly_pnl/latest.account_balance*100 if latest.account_balance > 0 else 0):+.2f}%</span>
                </div>
            </div>
            <div class="card">
                <h3>📅 Monthly Performance</h3>
                <div class="metric">
                    <span class="metric-label">Month P&L</span>
                    <span class="metric-value {'positive' if latest.monthly_pnl >= 0 else 'negative'}">${latest.monthly_pnl:+,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Monthly %</span>
                    <span class="metric-value {'positive' if latest.monthly_pnl >= 0 else 'negative'}">{(latest.monthly_pnl/latest.account_balance*100 if latest.account_balance > 0 else 0):+.2f}%</span>
                </div>
            </div>
        </div>

        <!-- ⚠️ Risk Metrics -->
        <div class="main-grid">
            <div class="card">
                <h3>💰 Total Margin Used</h3>
                <div class="metric">
                    <span class="metric-label">Used Margin</span>
                    <span class="metric-value">${latest.total_margin_used:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Free Margin</span>
                    <span class="metric-value positive">${latest.free_margin:,.2f}</span>
                </div>
            </div>
            <div class="card">
                <h3>📊 Margin Usage</h3>
                <div class="metric">
                    <span class="metric-label">Usage %</span>
                    <div style="background: rgba(255,255,255,0.1); height: 20px; border-radius: 10px; overflow: hidden; margin-top: 5px;">
                        <div style="background: {'linear-gradient(90deg, #ff4757 0%, #ffa502 100%)' if latest.margin_usage_pct > 80 else 'linear-gradient(90deg, #ffa502 0%, #00ff88 100%)' if latest.margin_usage_pct > 50 else 'linear-gradient(90deg, #00ff88 0%, #00d9ff 100%)'}; height: 100%; width: {min(latest.margin_usage_pct, 100):.1f}%; transition: width 0.3s;"></div>
                    </div>
                    <span class="metric-value {'negative' if latest.margin_usage_pct > 80 else 'positive' if latest.margin_usage_pct < 50 else ''}">{latest.margin_usage_pct:.1f}%</span>
                </div>
            </div>
            <div class="card">
                <h3>🎯 Largest Position</h3>
                <div class="metric">
                    <span class="metric-label">Margin Required</span>
                    <span class="metric-value">${latest.largest_position_margin:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">% of Balance</span>
                    <span class="metric-value">{(latest.largest_position_margin/latest.account_balance*100 if latest.account_balance > 0 else 0):.1f}%</span>
                </div>
            </div>
        </div>

        <!-- 📊 Open Positions Table -->
        {self._generate_positions_table(latest)}

        <!-- 📊 Recent Trades Table -->
        {self._generate_trades_table(latest)}

        <div class="chart-grid">
            <div class="chart-container">
                <h3>📊 PnL Evolution</h3>
                <div id="pnl-chart" style="height: 400px;"></div>
            </div>
            
            <div class="chart-container">
                <h3>🎯 Parameter Evolution</h3>
                <div id="parameters-chart" style="height: 400px;"></div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>📈 Performance Metrics Dashboard</h3>
            <div id="performance-chart" style="height: 500px;"></div>
        </div>
        
        <div class="footer">
            <p>🔄 Dashboard auto-refreshes every 30 seconds | 📁 Reports: {self.output_dir}</p>
            <p>🤖 Enhanced Trading Dashboard v2.0 • Real-time 1s polling enabled</p>
        </div>
    </div>
    
    <script>
        // PnL Chart with gradient fill
        var pnlValues = {[d.total_pnl for d in self.data_history[-100:]]};
        var pnlColors = pnlValues.map(v => v >= 0 ? 'rgba(0,255,136,0.8)' : 'rgba(255,71,87,0.8)');

        var pnlData = [{{
            x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
            y: pnlValues,
            type: 'scatter',
            mode: 'lines',
            name: 'Cumulative PnL',
            line: {{
                color: '#00ff88',
                width: 3,
                shape: 'spline',
                smoothing: 1.3
            }},
            fill: 'tozeroy',
            fillcolor: 'rgba(0,255,136,0.2)',
            hovertemplate: '<b>Time:</b> %{{x}}<br><b>PnL:</b> $%{{y:.2f}}<extra></extra>'
        }},
        {{
            x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
            y: {[d.equity for d in self.data_history[-100:]]},
            type: 'scatter',
            mode: 'lines',
            name: 'Account Equity',
            yaxis: 'y2',
            line: {{
                color: '#667eea',
                width: 2,
                shape: 'spline',
                smoothing: 1.3,
                dash: 'dot'
            }},
            hovertemplate: '<b>Equity:</b> $%{{y:.2f}}<extra></extra>'
        }}];

        var pnlLayout = {{
            title: {{
                text: '💰 PnL & Equity Evolution',
                font: {{size: 18, color: '#ffffff'}}
            }},
            xaxis: {{
                title: 'Time',
                color: '#999',
                gridcolor: 'rgba(255,255,255,0.1)',
                showgrid: true
            }},
            yaxis: {{
                title: 'PnL ($)',
                color: '#00ff88',
                gridcolor: 'rgba(255,255,255,0.1)',
                showgrid: true,
                zeroline: true,
                zerolinecolor: 'rgba(255,255,255,0.3)',
                zerolinewidth: 2
            }},
            yaxis2: {{
                title: 'Equity ($)',
                color: '#667eea',
                overlaying: 'y',
                side: 'right'
            }},
            plot_bgcolor: 'rgba(15,15,35,0.5)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#cccccc', family: 'Segoe UI, Arial'}},
            showlegend: true,
            legend: {{
                bgcolor: 'rgba(0,0,0,0.7)',
                bordercolor: 'rgba(255,255,255,0.2)',
                borderwidth: 1,
                x: 0.02,
                y: 0.98
            }},
            hovermode: 'x unified'
        }};

        Plotly.newPlot('pnl-chart', pnlData, pnlLayout, {{responsive: true, displayModeBar: false}});
        
        // Parameters Chart with area fills
        var paramData = [
            {{
                x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
                y: {[d.confidence_threshold for d in self.data_history[-100:]]},
                type: 'scatter',
                mode: 'lines',
                name: 'Confidence Threshold',
                line: {{
                    color: '#667eea',
                    width: 3,
                    shape: 'spline',
                    smoothing: 1.3
                }},
                fill: 'tozeroy',
                fillcolor: 'rgba(102,126,234,0.2)',
                hovertemplate: '<b>Confidence:</b> %{{y:.3f}}<extra></extra>'
            }},
            {{
                x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
                y: {[d.position_size_multiplier for d in self.data_history[-100:]]},
                type: 'scatter',
                mode: 'lines',
                name: 'Position Multiplier',
                yaxis: 'y2',
                line: {{
                    color: '#764ba2',
                    width: 3,
                    shape: 'spline',
                    smoothing: 1.3
                }},
                fill: 'tozeroy',
                fillcolor: 'rgba(118,75,162,0.2)',
                hovertemplate: '<b>Multiplier:</b> %{{y:.2f}}x<extra></extra>'
            }}
        ];

        var paramLayout = {{
            title: {{
                text: '🎛️ AI Parameter Evolution',
                font: {{size: 18, color: '#ffffff'}}
            }},
            xaxis: {{
                title: 'Time',
                color: '#999',
                gridcolor: 'rgba(255,255,255,0.1)',
                showgrid: true
            }},
            yaxis: {{
                title: 'Confidence Threshold',
                side: 'left',
                color: '#667eea',
                gridcolor: 'rgba(255,255,255,0.05)',
                showgrid: true
            }},
            yaxis2: {{
                title: 'Position Multiplier',
                side: 'right',
                overlaying: 'y',
                color: '#764ba2'
            }},
            plot_bgcolor: 'rgba(15,15,35,0.5)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#cccccc', family: 'Segoe UI, Arial'}},
            showlegend: true,
            legend: {{
                bgcolor: 'rgba(0,0,0,0.7)',
                bordercolor: 'rgba(255,255,255,0.2)',
                borderwidth: 1,
                x: 0.02,
                y: 0.98
            }},
            hovermode: 'x unified'
        }};

        Plotly.newPlot('parameters-chart', paramData, paramLayout, {{responsive: true, displayModeBar: false}});
        
        // Performance Dashboard - Win Rate & Trades Bar Chart
        var performanceData = [
            {{
                x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
                y: {[d.win_rate * 100 for d in self.data_history[-100:]]},
                type: 'scatter',
                mode: 'lines',
                name: 'Win Rate',
                line: {{
                    color: '#00ff88',
                    width: 4,
                    shape: 'spline',
                    smoothing: 1.3
                }},
                fill: 'tozeroy',
                fillcolor: 'rgba(0,255,136,0.15)',
                hovertemplate: '<b>Win Rate:</b> %{{y:.1f}}%<extra></extra>'
            }},
            {{
                x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
                y: {[min(d.profit_factor, 5.0) for d in self.data_history[-100:]]},
                type: 'bar',
                name: 'Profit Factor',
                yaxis: 'y2',
                marker: {{
                    color: {[f"'rgba(255,165,2,{min(d.profit_factor/5, 1)})'" for d in self.data_history[-100:]]},
                    line: {{color: '#ffa502', width: 1}}
                }},
                hovertemplate: '<b>Profit Factor:</b> %{{y:.2f}}<extra></extra>'
            }},
            {{
                x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
                y: {[d.total_trades for d in self.data_history[-100:]]},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Total Trades',
                yaxis: 'y3',
                line: {{
                    color: '#667eea',
                    width: 2,
                    dash: 'dot'
                }},
                marker: {{
                    size: 8,
                    color: '#667eea',
                    line: {{color: '#ffffff', width: 1}}
                }},
                hovertemplate: '<b>Trades:</b> %{{y}}<extra></extra>'
            }}
        ];

        var performanceLayout = {{
            title: {{
                text: '📊 Multi-Metric Performance Dashboard',
                font: {{size: 18, color: '#ffffff'}}
            }},
            xaxis: {{
                title: 'Time',
                color: '#999',
                gridcolor: 'rgba(255,255,255,0.1)',
                showgrid: true
            }},
            yaxis: {{
                title: 'Win Rate (%)',
                side: 'left',
                color: '#00ff88',
                gridcolor: 'rgba(255,255,255,0.05)',
                showgrid: true,
                range: [0, 100]
            }},
            yaxis2: {{
                title: 'Profit Factor',
                side: 'right',
                overlaying: 'y',
                color: '#ffa502',
                position: 0.85,
                range: [0, 5]
            }},
            yaxis3: {{
                title: 'Total Trades',
                side: 'right',
                overlaying: 'y',
                color: '#667eea'
            }},
            plot_bgcolor: 'rgba(15,15,35,0.5)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#cccccc', family: 'Segoe UI, Arial'}},
            showlegend: true,
            legend: {{
                bgcolor: 'rgba(0,0,0,0.7)',
                bordercolor: 'rgba(255,255,255,0.2)',
                borderwidth: 1,
                orientation: 'h',
                x: 0.5,
                y: -0.15,
                xanchor: 'center'
            }},
            hovermode: 'x unified',
            barmode: 'overlay'
        }};

        Plotly.newPlot('performance-chart', performanceData, performanceLayout, {{responsive: true, displayModeBar: false}});
        
        // Auto-refresh every 30 seconds
        setTimeout(function() {{ location.reload(); }}, 30000);
    </script>
</body>
</html>
        """

    def _generate_positions_table(self, data: DashboardData) -> str:
        """Generate table of open positions."""
        if not data.open_positions_details or len(data.open_positions_details) == 0:
            return """
                <div class="table-container">
                    <h3>📊 Open Positions</h3>
                    <p style="text-align: center; opacity: 0.6; padding: 20px;">No open positions</p>
                </div>
            """

        rows = ""
        for pos in data.open_positions_details:
            side_badge = f'<span class="badge badge-long">🟢 LONG</span>' if pos['side'] == 'LONG' else f'<span class="badge badge-short">🔴 SHORT</span>'
            pnl_class = 'positive' if pos['pnl'] >= 0 else 'negative'

            # Liquidation price если доступно
            liq_price = f"${pos['liquidation_price']:,.4f}" if pos.get('liquidation_price') else "N/A"

            rows += f"""
                <tr>
                    <td><strong>{pos['symbol']}</strong></td>
                    <td>{side_badge}</td>
                    <td><strong>{pos['leverage']:.1f}x</strong></td>
                    <td>${pos['entry_price']:,.4f}</td>
                    <td>${pos['current_price']:,.4f}</td>
                    <td>{pos['quantity']:.4f}</td>
                    <td>${pos['notional']:,.2f}</td>
                    <td><strong>${pos['margin_used']:,.2f}</strong></td>
                    <td class="{pnl_class}"><strong>${pos['pnl']:+,.2f}</strong></td>
                    <td class="{pnl_class}"><strong>{pos['pnl_pct']:+.2f}%</strong></td>
                    <td style="opacity: 0.8;">{liq_price}</td>
                </tr>
            """

        return f"""
            <div class="table-container">
                <h3>📊 Open Positions ({len(data.open_positions_details)})</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Leverage</th>
                            <th>Entry</th>
                            <th>Current</th>
                            <th>Quantity</th>
                            <th>Notional</th>
                            <th>Margin Used</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                            <th>Liquidation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
        """

    def _generate_trades_table(self, data: DashboardData) -> str:
        """Generate table of recent trades."""
        if not data.recent_trades or len(data.recent_trades) == 0:
            return """
                <div class="table-container">
                    <h3>📈 Recent Trades</h3>
                    <p style="text-align: center; opacity: 0.6; padding: 20px;">No recent trades</p>
                </div>
            """

        rows = ""
        for trade in reversed(data.recent_trades):  # Latest first
            side_badge = f'<span class="badge badge-long">BUY</span>' if trade['side'] == 'BUY' or trade['side'] == 'LONG' else f'<span class="badge badge-short">SELL</span>'
            pnl_class = 'positive' if trade['pnl'] >= 0 else 'negative'

            rows += f"""
                <tr>
                    <td>{trade['timestamp']}</td>
                    <td><strong>{trade['symbol']}</strong></td>
                    <td>{side_badge}</td>
                    <td class="{pnl_class}"><strong>${trade['pnl']:+,.2f}</strong></td>
                    <td class="{pnl_class}"><strong>{trade['pnl_pct']:+.2f}%</strong></td>
                </tr>
            """

        return f"""
            <div class="table-container">
                <h3>📈 Recent Trades (Last {len(data.recent_trades)})</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
        """

    def _save_history(self):
        """Сохраняет историю дашборда в JSON файл."""
        try:
            if not self.data_history:
                logger.debug("📊 [DASHBOARD_HISTORY] No data to save yet")
                return

            # Конвертируем dataclass объекты в dict
            history_data = []
            for data in self.data_history[-100:]:  # Сохраняем только последние 100 точек
                try:
                    data_dict = asdict(data)
                    # Конвертируем datetime в строку
                    if isinstance(data.timestamp, datetime):
                        data_dict['timestamp'] = data.timestamp.isoformat()
                    history_data.append(data_dict)
                except Exception as e:
                    logger.debug(f"⚠️ [DASHBOARD_HISTORY] Failed to serialize data point: {e}")
                    continue

            if not history_data:
                logger.warning("❌ [DASHBOARD_HISTORY] No valid data points to save")
                return

            # Создаем директорию если не существует
            self.history_file.parent.mkdir(parents=True, exist_ok=True)

            # Сохраняем с pretty print
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 [DASHBOARD_HISTORY] Saved {len(history_data)} data points to {self.history_file}")
        except Exception as e:
            logger.warning(f"❌ [DASHBOARD_HISTORY] Failed to save history: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")

    def _load_history(self):
        """Загружает историю дашборда из JSON файла."""
        try:
            if not self.history_file.exists():
                logger.debug("📊 [DASHBOARD_HISTORY] No history file found, starting fresh")
                return

            with open(self.history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)

            # Получаем список всех полей DashboardData
            valid_fields = {f.name for f in fields(DashboardData)}

            # Конвертируем dict обратно в DashboardData
            loaded_count = 0
            for data_dict in history_data:
                try:
                    # Конвертируем timestamp обратно в datetime
                    if 'timestamp' in data_dict:
                        data_dict['timestamp'] = datetime.fromisoformat(data_dict['timestamp'])

                    # Фильтруем только известные поля (игнорируем устаревшие)
                    filtered_dict = {k: v for k, v in data_dict.items() if k in valid_fields}

                    # Создаем DashboardData объект
                    dashboard_data = DashboardData(**filtered_dict)
                    self.data_history.append(dashboard_data)
                    loaded_count += 1
                except Exception as e:
                    logger.debug(f"⚠️ [DASHBOARD_HISTORY] Skipped invalid data point: {e}")
                    continue

            logger.info(f"📊 [DASHBOARD_HISTORY] Successfully loaded {loaded_count}/{len(history_data)} historical points")
        except Exception as e:
            logger.warning(f"❌ [DASHBOARD_HISTORY] Failed to load history: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            self.data_history = []

    def _generate_empty_dashboard(self) -> str:
        """Генерирует дашборд для случая отсутствия данных."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Trading Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #0f0f23; color: white; }
        .loading { font-size: 24px; margin: 50px 0; }
    </style>
</head>
<body>
    <h1>🚀 Enhanced Trading Dashboard</h1>
    <div class="loading">📊 Collecting data... Please wait for the bot to generate some trading activity.</div>
    <p>This dashboard will auto-refresh with live data every 30 seconds.</p>
    <script>setTimeout(function() { location.reload(); }, 30000);</script>
</body>
</html>
        """
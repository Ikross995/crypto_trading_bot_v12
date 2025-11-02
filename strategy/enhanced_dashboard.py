#!/usr/bin/env python3
"""
Enhanced Trading Dashboard - Improved Real-time Analytics
Создает мощный интерактивный дашборд с расширенной статистикой
"""

import asyncio
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
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


class EnhancedDashboardGenerator:
    """Генератор улучшенного дашборда с богатой статистикой."""
    
    def __init__(self, output_dir: str = "data/learning_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_file = self.output_dir / "enhanced_dashboard.html"
        
        # История данных для графиков
        self.data_history: List[DashboardData] = []
        
        logger.info(f"📊 [ENHANCED_DASHBOARD] Initialized: {self.dashboard_file}")
    
    async def update_dashboard(self, trading_engine=None, adaptive_learning=None) -> str:
        """Обновляет дашборд с новыми данными."""
        try:
            # Собираем данные
            dashboard_data = await self._collect_dashboard_data(trading_engine, adaptive_learning)
            
            # Добавляем в историю
            self.data_history.append(dashboard_data)
            
            # Ограничиваем историю последними 500 точками
            if len(self.data_history) > 500:
                self.data_history = self.data_history[-500:]
            
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
    
    async def _collect_dashboard_data(self, trading_engine=None, adaptive_learning=None) -> DashboardData:
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
        
        # Данные системы обучения
        if adaptive_learning:
            data = await self._get_learning_data(adaptive_learning, data)
        
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
        // PnL Chart
        var pnlData = [{{
            x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
            y: {[d.total_pnl for d in self.data_history[-100:]]},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Cumulative PnL',
            line: {{color: '#00ff88', width: 3}},
            marker: {{size: 6}},
            fill: 'tonexty',
            fillcolor: 'rgba(0,255,136,0.1)'
        }}];
        
        var pnlLayout = {{
            title: 'Cumulative PnL Over Time',
            xaxis: {{title: 'Time', color: '#cccccc'}},
            yaxis: {{title: 'PnL ($)', color: '#cccccc'}},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#cccccc'}},
            showlegend: false
        }};
        
        Plotly.newPlot('pnl-chart', pnlData, pnlLayout, {{responsive: true}});
        
        // Parameters Chart
        var paramData = [
            {{
                x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
                y: {[d.confidence_threshold for d in self.data_history[-100:]]},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Confidence Threshold',
                line: {{color: '#667eea', width: 2}}
            }},
            {{
                x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
                y: {[d.position_size_multiplier for d in self.data_history[-100:]]},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Position Multiplier',
                yaxis: 'y2',
                line: {{color: '#764ba2', width: 2}}
            }}
        ];
        
        var paramLayout = {{
            title: 'AI Parameter Evolution',
            xaxis: {{title: 'Time', color: '#cccccc'}},
            yaxis: {{title: 'Confidence Threshold', side: 'left', color: '#667eea'}},
            yaxis2: {{
                title: 'Position Multiplier',
                side: 'right',
                overlaying: 'y',
                color: '#764ba2'
            }},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#cccccc'}},
            showlegend: true,
            legend: {{bgcolor: 'rgba(0,0,0,0.5)'}}
        }};
        
        Plotly.newPlot('parameters-chart', paramData, paramLayout, {{responsive: true}});
        
        // Performance Dashboard
        var performanceData = [
            {{
                x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
                y: {[d.win_rate * 100 for d in self.data_history[-100:]]},
                type: 'scatter',
                mode: 'lines',
                name: 'Win Rate (%)',
                line: {{color: '#00ff88', width: 2}}
            }},
            {{
                x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
                y: {[min(d.profit_factor, 5.0) for d in self.data_history[-100:]]},  // Cap at 5 for better visualization
                type: 'scatter',
                mode: 'lines',
                name: 'Profit Factor',
                yaxis: 'y2',
                line: {{color: '#ffa502', width: 2}}
            }},
            {{
                x: {[f"'{d.timestamp.strftime('%H:%M:%S')}'" for d in self.data_history[-100:]]},
                y: {[d.equity for d in self.data_history[-100:]]},
                type: 'scatter',
                mode: 'lines',
                name: 'Account Equity',
                yaxis: 'y3',
                line: {{color: '#667eea', width: 2}}
            }}
        ];
        
        var performanceLayout = {{
            title: 'Multi-Metric Performance Dashboard',
            xaxis: {{title: 'Time', color: '#cccccc'}},
            yaxis: {{title: 'Win Rate (%)', side: 'left', color: '#00ff88'}},
            yaxis2: {{
                title: 'Profit Factor',
                side: 'right',
                overlaying: 'y',
                color: '#ffa502',
                position: 0.85
            }},
            yaxis3: {{
                title: 'Equity ($)',
                side: 'right',
                overlaying: 'y',
                color: '#667eea'
            }},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#cccccc'}},
            showlegend: true,
            legend: {{bgcolor: 'rgba(0,0,0,0.5)'}}
        }};
        
        Plotly.newPlot('performance-chart', performanceData, performanceLayout, {{responsive: true}});
        
        // Auto-refresh every 30 seconds
        setTimeout(function() {{ location.reload(); }}, 30000);
    </script>
</body>
</html>
        """
    
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
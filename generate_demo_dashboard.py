#!/usr/bin/env python3
"""Quick demo dashboard generator"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

from strategy.enhanced_dashboard import EnhancedDashboardGenerator, DashboardData

# Create demo data
data = DashboardData(
    timestamp=datetime.now(timezone.utc),
    account_balance=1050.00,
    equity=1075.50,
    total_pnl=75.50,
    unrealized_pnl=25.50,
    realized_pnl=50.00,

    # Trading Stats
    total_trades=15,
    winning_trades=10,
    losing_trades=5,
    win_rate=0.667,
    avg_trade_pnl=5.03,
    best_trade=25.50,
    worst_trade=-8.20,
    profit_factor=2.5,

    # Open Positions
    open_positions=2,
    total_position_value=215.00,
    largest_position=120.00,
    available_balance=860.00,

    # Market Data
    market_volatility=0.025,
    market_trend="BULLISH",
    price_change_24h=0.023,
    volume_24h=125000000.00,

    # System Stats
    iteration=150,
    uptime_hours=2.5,
    signals_generated=45,
    signals_executed=15,
    execution_rate=0.333,

    # Adaptive Learning
    confidence_threshold=0.65,
    position_size_multiplier=1.2,
    adaptations_count=12,
    learning_confidence=0.75,

    # Extended stats
    sharpe_ratio=1.8,
    daily_pnl=25.50,
    weekly_pnl=75.50,
    monthly_pnl=75.50,
    max_drawdown=15.30,

    # Risk Metrics
    total_margin_used=43.00,  # $215 notional / 5x average leverage
    margin_usage_pct=4.1,  # 43/1050 * 100
    free_margin=1007.00,
    largest_position_margin=24.00,  # $120 / 5x
)

# Add open positions details
data.open_positions_details = [
    {
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        'leverage': 5.0,
        'entry_price': 40250.00,
        'current_price': 40780.00,
        'quantity': 0.0025,
        'notional': 101.95,
        'margin_used': 20.39,
        'pnl': 1.33,
        'pnl_pct': 1.32,
        'liquidation_price': 32200.00
    },
    {
        'symbol': 'ETHUSDT',
        'side': 'LONG',
        'leverage': 3.0,
        'entry_price': 2180.00,
        'current_price': 2245.00,
        'quantity': 0.05,
        'notional': 112.25,
        'margin_used': 37.42,
        'pnl': 3.25,
        'pnl_pct': 2.98,
        'liquidation_price': 1453.33
    }
]

# Add recent trades
data.recent_trades = [
    {'symbol': 'BTCUSDT', 'side': 'LONG', 'pnl': 12.50, 'pnl_pct': 2.1, 'timestamp': '14:05:23'},
    {'symbol': 'ETHUSDT', 'side': 'SHORT', 'pnl': -3.20, 'pnl_pct': -0.8, 'timestamp': '14:03:15'},
    {'symbol': 'BTCUSDT', 'side': 'LONG', 'pnl': 8.30, 'pnl_pct': 1.5, 'timestamp': '13:58:42'},
    {'symbol': 'SOLUSDT', 'side': 'SHORT', 'pnl': 5.80, 'pnl_pct': 1.2, 'timestamp': '13:45:10'},
    {'symbol': 'ETHUSDT', 'side': 'LONG', 'pnl': 15.20, 'pnl_pct': 3.1, 'timestamp': '13:32:05'},
]

# Generate dashboard
print("🚀 Generating demo dashboard...")
dashboard_gen = EnhancedDashboardGenerator()

# Add data to history
dashboard_gen.data_history = [data]

# Generate HTML
html_path = dashboard_gen.generate_dashboard_html(data)

print(f"✅ Dashboard generated: {html_path}")
print(f"📊 Open in browser: file://{html_path.absolute()}")
print()
print("📋 Dashboard features:")
print("  ✅ ML Learning & GRU sections removed")
print("  ✅ Correct leverage calculation")
print("  ✅ Margin Used shown separately from Notional")
print("  ✅ Risk Metrics section with margin usage")
print("  ✅ Liquidation prices displayed")
print("  ✅ 11-column positions table")
print()
print("🤖 For Telegram integration:")
print("  - HTML dashboard can be converted to screenshot")
print("  - Or send formatted text summary with emojis")
print("  - Or send as HTML file attachment")

"""
🎯 Complete Integration Example: GRU Model + Adaptive Strategy
===============================================================

This example demonstrates how to integrate:
1. GRU Price Predictor (ML-based price forecasting)
2. Market Regime Detector (identify market conditions)
3. Adaptive Strategy Manager (adjust strategy to regime)

Together, these create a sophisticated trading system that:
- Predicts future prices with ML
- Detects market regime
- Adapts strategy parameters automatically
- Makes informed trading decisions

WORKFLOW:
Market Data → Regime Detection → Strategy Adaptation → GRU Prediction → Trading Decision
"""

# Fix imports for Windows/standalone execution
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from models.gru_predictor import GRUPricePredictor
from strategy.regime_detector import MarketRegimeDetector
from strategy.adaptive_strategy import AdaptiveStrategyManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def prepare_market_data(symbol='BTCUSDT', candles=1500):
    """
    Prepare market data with technical indicators.

    In production, replace this with actual data from your exchange API.
    """
    logger.info(f"📊 Loading market data for {symbol}...")

    # Example: Replace with actual data fetching
    # from data.binance_data import fetch_candles
    # df = await fetch_candles(symbol, interval='15m', limit=candles)

    # For this example, create simulated data
    dates = pd.date_range(
        end=datetime.now(),
        periods=candles,
        freq='15min'
    )

    # Simulated realistic price movement
    np.random.seed(42)
    base_price = 50000
    price_changes = np.random.randn(len(dates)) * 200

    df = pd.DataFrame({
        'timestamp': dates,
        'open': base_price + np.cumsum(price_changes),
        'high': base_price + np.cumsum(price_changes) + np.abs(np.random.randn(len(dates)) * 100),
        'low': base_price + np.cumsum(price_changes) - np.abs(np.random.randn(len(dates)) * 100),
        'close': base_price + np.cumsum(price_changes) + np.random.randn(len(dates)) * 50,
        'volume': np.random.uniform(1000, 10000, len(dates))
    })

    # Add technical indicators (required for regime detection)
    df = add_technical_indicators(df)

    logger.info(f"✅ Loaded {len(df)} candles with indicators")

    return df


def add_technical_indicators(df):
    """Add all required technical indicators"""

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Bollinger Bands
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_middle'] = sma_20

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # ADX (Average Directional Index)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr14 = true_range.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)

    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(14).mean()

    # Moving Averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()

    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Drop NaN values
    df = df.dropna().reset_index(drop=True)

    return df


async def train_gru_on_historical_data(df, symbol='BTCUSDT'):
    """
    Train GRU model on historical data.

    This should be done once initially, then periodically retrained.
    """
    logger.info("🤖 Training GRU model on historical data...")

    # Feature columns for GRU
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'sma_20', 'ema_50',
        'bb_upper', 'bb_lower', 'volume_sma'
    ]

    df_features = df[feature_columns]

    # Initialize GRU predictor
    predictor = GRUPricePredictor(
        sequence_length=60,  # Use last 60 candles (15 hours for 15m timeframe)
        features=len(feature_columns),
        gru_units=100,
        dropout_rate=0.2,
        learning_rate=0.001
    )

    # Prepare training data
    X_train, y_train, X_test, y_test = predictor.prepare_data(
        data=df_features,
        target_column='close',
        train_split=0.8
    )

    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")

    # Train model
    history = await predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=20,
        batch_size=32,
        verbose=1
    )

    # Evaluate
    metrics = await predictor.evaluate(X_test, y_test)

    logger.info("📊 Training Results:")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  R²: {metrics['r2_score']:.4f}")

    # Save model
    model_path = f'models/checkpoints/gru_{symbol.lower()}.keras'
    predictor.save(model_path)
    logger.info(f"✅ Model saved to {model_path}")

    return predictor


async def integrated_trading_decision(
    df: pd.DataFrame,
    gru_predictor: GRUPricePredictor,
    adaptive_manager: AdaptiveStrategyManager,
    symbol: str = 'BTCUSDT'
):
    """
    Make integrated trading decision using GRU + Adaptive Strategy.

    This is the core logic that combines:
    1. Market regime detection
    2. Strategy adaptation
    3. GRU price prediction
    4. Risk-adjusted decision making
    """

    logger.info("=" * 70)
    logger.info(f"🎯 [INTEGRATED_ANALYSIS] {symbol}")
    logger.info("=" * 70)

    current_price = df['close'].iloc[-1]
    current_time = df['timestamp'].iloc[-1]

    logger.info(f"Time: {current_time}")
    logger.info(f"Price: ${current_price:,.2f}")
    logger.info("")

    # === STEP 1: Detect Market Regime ===
    logger.info("📊 Step 1: Market Regime Detection")
    regime_info = await adaptive_manager.update_regime(df)

    logger.info(f"  Regime: {regime_info.regime.value.upper()}")
    logger.info(f"  Confidence: {regime_info.confidence:.2f}")
    logger.info(f"  ADX: {regime_info.metrics.get('adx', 0):.1f}")
    logger.info(f"  Volatility: {regime_info.metrics.get('volatility', 0):.2f}%")
    logger.info("")

    # === STEP 2: Get Adaptive Strategy Parameters ===
    logger.info("🎯 Step 2: Strategy Adaptation")
    params = adaptive_manager.get_current_parameters()

    logger.info(f"  Strategy: {adaptive_manager._get_strategy_description(params)}")
    logger.info(f"  Position Size Multiplier: {params.position_size_multiplier}x")
    logger.info(f"  Stop Loss: {params.stop_loss_multiplier}× ATR")
    logger.info(f"  Take Profit: {params.take_profit_multiplier}× ATR")
    logger.info(f"  Confidence Threshold: {params.confidence_threshold:.2f}")
    logger.info(f"  DCA Enabled: {params.enable_dca}")
    logger.info("")

    # === STEP 3: GRU Price Prediction ===
    logger.info("🤖 Step 3: GRU Price Prediction")

    # Prepare features for prediction
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'sma_20', 'ema_50',
        'bb_upper', 'bb_lower', 'volume_sma'
    ]

    # Get last sequence (60 candles)
    last_sequence = df[feature_columns].iloc[-60:].values

    # Scale and reshape
    last_sequence_scaled = gru_predictor.scaler.transform(last_sequence)
    X = last_sequence_scaled.reshape(1, 60, len(feature_columns))

    # Predict
    predicted_price = (await gru_predictor.predict(X))[0]
    price_change_pct = ((predicted_price - current_price) / current_price) * 100

    logger.info(f"  Current Price: ${current_price:,.2f}")
    logger.info(f"  Predicted Price: ${predicted_price:,.2f}")
    logger.info(f"  Expected Change: {price_change_pct:+.2f}%")

    # Calculate signal confidence based on prediction magnitude
    signal_confidence = min(abs(price_change_pct) / 2.0, 1.0)  # Max confidence at 2% move
    logger.info(f"  Signal Confidence: {signal_confidence:.2f}")
    logger.info("")

    # === STEP 4: Trading Decision ===
    logger.info("💡 Step 4: Trading Decision")

    # Determine direction
    if price_change_pct > 0.3:
        signal_direction = 'BUY'
        signal_strength = price_change_pct
    elif price_change_pct < -0.3:
        signal_direction = 'SELL'
        signal_strength = abs(price_change_pct)
    else:
        signal_direction = 'NEUTRAL'
        signal_strength = 0

    # Check if trade should be taken based on regime
    should_trade, reason = adaptive_manager.should_take_trade(
        signal_confidence=signal_confidence,
        signal_direction=signal_direction,
        last_trade_direction=None  # Would track in real bot
    )

    logger.info(f"  Signal: {signal_direction}")
    logger.info(f"  Signal Strength: {signal_strength:.2f}%")
    logger.info(f"  Should Trade: {should_trade}")
    logger.info(f"  Reason: {reason}")
    logger.info("")

    if should_trade and signal_direction != 'NEUTRAL':
        # === STEP 5: Position Sizing & Risk Management ===
        logger.info("📊 Step 5: Position Sizing & Risk")

        # Base position size (example: $1000)
        base_position_size = 1000
        adjusted_size = adaptive_manager.adjust_position_size(base_position_size)

        # Calculate stop loss and take profit
        atr = df['atr'].iloc[-1]
        stop_loss_distance = adaptive_manager.get_stop_loss_distance(atr)
        take_profit_distance = adaptive_manager.get_take_profit_distance(atr)

        if signal_direction == 'BUY':
            stop_loss_price = current_price - stop_loss_distance
            take_profit_price = current_price + take_profit_distance
        else:  # SELL
            stop_loss_price = current_price + stop_loss_distance
            take_profit_price = current_price - take_profit_distance

        stop_loss_pct = (stop_loss_distance / current_price) * 100
        take_profit_pct = (take_profit_distance / current_price) * 100
        risk_reward = take_profit_distance / stop_loss_distance

        logger.info(f"  Base Position: ${base_position_size:,.2f}")
        logger.info(f"  Adjusted Position: ${adjusted_size:,.2f} ({params.position_size_multiplier}x)")
        logger.info(f"  ATR: ${atr:.2f}")
        logger.info(f"  Stop Loss: ${stop_loss_price:,.2f} ({stop_loss_pct:.2f}%)")
        logger.info(f"  Take Profit: ${take_profit_price:,.2f} ({take_profit_pct:.2f}%)")
        logger.info(f"  Risk/Reward: 1:{risk_reward:.2f}")
        logger.info("")

        # === FINAL DECISION ===
        logger.info("✅ TRADE APPROVED")
        logger.info("=" * 70)

        return {
            'action': signal_direction,
            'regime': regime_info.regime.value,
            'predicted_price': predicted_price,
            'expected_change_pct': price_change_pct,
            'signal_confidence': signal_confidence,
            'position_size': adjusted_size,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'risk_reward': risk_reward
        }
    else:
        logger.info("❌ TRADE REJECTED")
        logger.info("=" * 70)

        return {
            'action': 'HOLD',
            'regime': regime_info.regime.value,
            'reason': reason
        }


async def run_continuous_trading_loop():
    """
    Example: Continuous trading loop with integrated system.

    This simulates a real trading bot that:
    1. Loads market data every interval
    2. Detects regime
    3. Adapts strategy
    4. Gets GRU prediction
    5. Makes trading decision
    """
    logger.info("🚀 Starting Integrated Trading System...")
    logger.info("")

    # Initialize components
    adaptive_manager = AdaptiveStrategyManager(
        enable_regime_logging=True
    )

    # Load or train GRU model
    gru_predictor = GRUPricePredictor(
        sequence_length=60,
        features=12
    )

    try:
        # Try to load existing model
        gru_predictor.load('models/checkpoints/gru_btcusdt.keras')
        logger.info("✅ Loaded existing GRU model")
    except:
        # Train new model if not exists
        logger.info("📚 Training new GRU model (first run)...")
        df_historical = await prepare_market_data(symbol='BTCUSDT', candles=5000)
        gru_predictor = await train_gru_on_historical_data(df_historical)

    logger.info("")
    logger.info("🔄 Entering trading loop...")
    logger.info("")

    # Main trading loop
    for iteration in range(5):  # In production, this runs continuously
        logger.info(f"🔄 Iteration {iteration + 1}")

        # Fetch latest market data
        df = await prepare_market_data(symbol='BTCUSDT', candles=1500)

        # Make integrated trading decision
        decision = await integrated_trading_decision(
            df=df,
            gru_predictor=gru_predictor,
            adaptive_manager=adaptive_manager,
            symbol='BTCUSDT'
        )

        # Execute trade (in production)
        if decision['action'] in ['BUY', 'SELL']:
            logger.info(f"📢 Would execute {decision['action']} order:")
            logger.info(f"  Size: ${decision['position_size']:,.2f}")
            logger.info(f"  Stop Loss: ${decision['stop_loss']:,.2f}")
            logger.info(f"  Take Profit: ${decision['take_profit']:,.2f}")
            # await exchange.create_order(...)

        # Wait for next interval (15 minutes in production)
        await asyncio.sleep(2)  # Short sleep for demo
        logger.info("")

    # Log final statistics
    logger.info("=" * 70)
    logger.info("📊 Session Statistics")
    logger.info("=" * 70)
    adaptive_manager.log_regime_statistics()

    status = adaptive_manager.get_status_summary()
    logger.info(f"Current Regime: {status['current_regime']}")
    logger.info(f"Current Strategy: {status['current_strategy']}")
    logger.info(f"Trading Active: {status['trading_active']}")


async def backtest_integrated_system():
    """
    Backtest the integrated system on historical data.
    """
    logger.info("📈 Starting Integrated System Backtest...")
    logger.info("")

    # Load historical data
    df = await prepare_market_data(symbol='BTCUSDT', candles=5000)

    # Initialize components
    adaptive_manager = AdaptiveStrategyManager()
    gru_predictor = await train_gru_on_historical_data(df)

    # Backtest parameters
    initial_balance = 10000
    balance = initial_balance
    trades = []

    # Walk forward through history
    for i in range(1000, len(df), 10):  # Every 10 candles (2.5 hours for 15m)
        df_slice = df.iloc[:i]

        if len(df_slice) < 100:
            continue

        # Make decision
        decision = await integrated_trading_decision(
            df=df_slice,
            gru_predictor=gru_predictor,
            adaptive_manager=adaptive_manager,
            symbol='BTCUSDT'
        )

        if decision['action'] in ['BUY', 'SELL']:
            # Simulate trade execution and outcome
            entry_price = df_slice['close'].iloc[-1]
            position_size = min(decision['position_size'], balance * 0.1)  # Max 10% per trade

            # Look ahead to see outcome (in real backtest, use proper exit logic)
            if i + 20 < len(df):
                future_prices = df['close'].iloc[i:i+20]

                # Check if stop loss or take profit hit
                if decision['action'] == 'BUY':
                    hit_tp = any(future_prices >= decision['take_profit'])
                    hit_sl = any(future_prices <= decision['stop_loss'])
                else:  # SELL
                    hit_tp = any(future_prices <= decision['take_profit'])
                    hit_sl = any(future_prices >= decision['stop_loss'])

                if hit_tp:
                    pnl = position_size * (decision['risk_reward'] * 0.02)  # 2% risk, RR ratio profit
                    balance += pnl
                    outcome = 'WIN'
                elif hit_sl:
                    pnl = -position_size * 0.02  # 2% loss
                    balance += pnl
                    outcome = 'LOSS'
                else:
                    pnl = 0
                    outcome = 'NEUTRAL'

                trades.append({
                    'entry': entry_price,
                    'action': decision['action'],
                    'regime': decision['regime'],
                    'size': position_size,
                    'pnl': pnl,
                    'outcome': outcome,
                    'balance': balance
                })

    # Calculate statistics
    total_trades = len(trades)
    wins = len([t for t in trades if t['outcome'] == 'WIN'])
    losses = len([t for t in trades if t['outcome'] == 'LOSS'])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = balance - initial_balance
    return_pct = (total_pnl / initial_balance) * 100

    logger.info("=" * 70)
    logger.info("📊 BACKTEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Wins: {wins} ({win_rate:.1f}%)")
    logger.info(f"Losses: {losses}")
    logger.info(f"Initial Balance: ${initial_balance:,.2f}")
    logger.info(f"Final Balance: ${balance:,.2f}")
    logger.info(f"Total P&L: ${total_pnl:+,.2f} ({return_pct:+.2f}%)")
    logger.info("=" * 70)

    # Regime performance breakdown
    regime_stats = {}
    for trade in trades:
        regime = trade['regime']
        if regime not in regime_stats:
            regime_stats[regime] = {'trades': 0, 'wins': 0, 'pnl': 0}
        regime_stats[regime]['trades'] += 1
        if trade['outcome'] == 'WIN':
            regime_stats[regime]['wins'] += 1
        regime_stats[regime]['pnl'] += trade['pnl']

    logger.info("")
    logger.info("📊 Performance by Regime:")
    for regime, stats in regime_stats.items():
        wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        logger.info(f"  {regime}: {stats['trades']} trades, {wr:.1f}% WR, ${stats['pnl']:+,.2f}")

    return trades


if __name__ == '__main__':
    # Choose example to run:

    # 1. Single integrated trading decision
    # asyncio.run(integrated_trading_decision_example())

    # 2. Continuous trading loop (simulated)
    asyncio.run(run_continuous_trading_loop())

    # 3. Backtest integrated system
    # asyncio.run(backtest_integrated_system())

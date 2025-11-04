"""
Example: Train and use GRU model for price prediction
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from models.gru_predictor import GRUPricePredictor
from data.indicators import calculate_technical_indicators  # Your indicators function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def prepare_training_data(symbol='BTCUSDT', days=365):
    """
    Prepare training data with technical indicators.

    In real implementation, fetch from your data source.
    This is a simplified example.
    """
    logger.info(f"📊 Preparing training data for {symbol}...")

    # Example: Load your historical data here
    # df = await fetch_historical_data(symbol, days)

    # For this example, create dummy data
    # In production, replace with actual data fetching
    dates = pd.date_range(
        end=datetime.now(),
        periods=days * 1440,  # 1 minute candles
        freq='1min'
    )

    # Simulated price data (replace with real data!)
    np.random.seed(42)
    base_price = 50000
    df = pd.DataFrame({
        'timestamp': dates,
        'open': base_price + np.cumsum(np.random.randn(len(dates)) * 100),
        'high': base_price + np.cumsum(np.random.randn(len(dates)) * 100) + 50,
        'low': base_price + np.cumsum(np.random.randn(len(dates)) * 100) - 50,
        'close': base_price + np.cumsum(np.random.randn(len(dates)) * 100),
        'volume': np.random.uniform(100, 1000, len(dates))
    })

    # Add technical indicators
    # Replace with your actual indicator calculation
    df['rsi'] = np.random.uniform(30, 70, len(df))
    df['macd'] = np.random.randn(len(df))
    df['bb_upper'] = df['close'] * 1.02
    df['bb_lower'] = df['close'] * 0.98
    df['sma_20'] = df['close'].rolling(20).mean().bfill()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['volume_sma'] = df['volume'].rolling(20).mean().bfill()

    # Drop NaN
    df = df.dropna()

    logger.info(f"✅ Data prepared: {len(df)} samples")

    return df


async def train_gru_model():
    """
    Train GRU model on historical data.
    """
    logger.info("🚀 Starting GRU model training pipeline...")

    # 1. Prepare data
    df = await prepare_training_data(symbol='BTCUSDT', days=365)

    # 2. Select features
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'sma_20', 'ema_50',
        'bb_upper', 'bb_lower', 'volume_sma'
    ]

    # Ensure all features exist
    df_features = df[feature_columns]

    # 3. Initialize GRU model
    predictor = GRUPricePredictor(
        sequence_length=60,  # Use last 60 candles (1 hour for 1m timeframe)
        features=len(feature_columns),
        gru_units=100,
        dropout_rate=0.2,
        learning_rate=0.01
    )

    # 4. Prepare data
    X_train, y_train, X_test, y_test = predictor.prepare_data(
        data=df_features,
        target_column='close',
        train_split=0.8
    )

    logger.info(f"📊 Training set: {len(X_train)} samples")
    logger.info(f"📊 Test set: {len(X_test)} samples")

    # 5. Train model
    history = await predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=20,
        batch_size=32,
        verbose=1
    )

    # 6. Evaluate model
    metrics = await predictor.evaluate(X_test, y_test)

    # 7. Save model
    predictor.save('models/checkpoints/gru_model.keras')

    logger.info("✅ Training pipeline completed!")

    return predictor, metrics


async def use_trained_model():
    """
    Load and use trained model for prediction.
    """
    logger.info("🔮 Loading trained model for prediction...")

    # Initialize predictor
    predictor = GRUPricePredictor()

    # Load trained model
    predictor.load('models/checkpoints/gru_model.keras')

    # Prepare recent data (last 60 candles)
    df = await prepare_training_data(symbol='BTCUSDT', days=7)

    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'sma_20', 'ema_50',
        'bb_upper', 'bb_lower', 'volume_sma'
    ]

    df_features = df[feature_columns]

    # Get last sequence
    last_sequence = df_features.iloc[-60:].values

    # Scale
    last_sequence_scaled = predictor.scaler.transform(last_sequence)

    # Reshape for prediction (add batch dimension)
    X = last_sequence_scaled.reshape(1, 60, len(feature_columns))

    # Predict
    prediction = await predictor.predict(X)

    current_price = df['close'].iloc[-1]
    predicted_price = prediction[0]
    price_change = ((predicted_price - current_price) / current_price) * 100

    logger.info("=" * 70)
    logger.info("🔮 [PREDICTION] GRU Price Forecast")
    logger.info("=" * 70)
    logger.info(f"Current Price:   ${current_price:,.2f}")
    logger.info(f"Predicted Price: ${predicted_price:,.2f}")
    logger.info(f"Expected Change: {price_change:+.2f}%")

    if abs(price_change) > 0.5:
        direction = "📈 UP" if price_change > 0 else "📉 DOWN"
        logger.info(f"Signal: {direction}")
    else:
        logger.info("Signal: ➡️  NEUTRAL")

    logger.info("=" * 70)

    return predicted_price


async def walk_forward_validation():
    """
    Perform walk-forward validation (proper backtesting for time series).
    """
    logger.info("🔄 Starting walk-forward validation...")

    df = await prepare_training_data(symbol='BTCUSDT', days=365)

    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'sma_20', 'ema_50',
        'bb_upper', 'bb_lower', 'volume_sma'
    ]

    # Parameters
    train_window = 30 * 1440  # 30 days of 1m data
    test_window = 7 * 1440    # 7 days test
    step = 7 * 1440          # Step by 1 week

    predictions_all = []
    actuals_all = []

    for start_idx in range(0, len(df) - train_window - test_window, step):
        train_end = start_idx + train_window
        test_end = train_end + test_window

        # Get train and test sets
        train_data = df.iloc[start_idx:train_end][feature_columns]
        test_data = df.iloc[train_end:test_end][feature_columns]

        # Train model
        predictor = GRUPricePredictor(
            sequence_length=60,
            features=len(feature_columns)
        )

        X_train, y_train, _, _ = predictor.prepare_data(
            train_data,
            target_column='close',
            train_split=1.0  # Use all for training
        )

        await predictor.train(
            X_train, y_train,
            epochs=10,  # Fewer epochs for speed
            batch_size=32,
            verbose=0
        )

        # Prepare test data
        X_test, y_test, _, _ = predictor.prepare_data(
            test_data,
            target_column='close',
            train_split=1.0
        )

        # Predict
        predictions = await predictor.predict(X_test[:min(100, len(X_test))])
        actuals = predictor.target_scaler.inverse_transform(
            y_test[:len(predictions)]
        ).flatten()

        predictions_all.extend(predictions)
        actuals_all.extend(actuals)

        logger.info(
            f"✅ Window {start_idx//step + 1}: "
            f"MAPE = {np.mean(np.abs((actuals - predictions) / actuals)) * 100:.2f}%"
        )

    # Overall metrics
    predictions_all = np.array(predictions_all)
    actuals_all = np.array(actuals_all)

    mape = np.mean(np.abs((actuals_all - predictions_all) / actuals_all)) * 100

    logger.info("=" * 70)
    logger.info(f"📊 Walk-Forward Validation MAPE: {mape:.2f}%")
    logger.info("=" * 70)

    return mape


async def integrate_with_trading_bot():
    """
    Example: Integrate GRU predictions into trading bot.
    """
    logger.info("🤖 Integrating GRU with trading bot...")

    # Load model
    predictor = GRUPricePredictor()
    predictor.load('models/checkpoints/gru_model.keras')

    while True:  # In real bot, this would be your main loop
        # Get latest data
        df = await prepare_training_data(symbol='BTCUSDT', days=1)

        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'sma_20', 'ema_50',
            'bb_upper', 'bb_lower', 'volume_sma'
        ]

        # Prepare sequence
        last_sequence = df[feature_columns].iloc[-60:].values
        last_sequence_scaled = predictor.scaler.transform(last_sequence)
        X = last_sequence_scaled.reshape(1, 60, len(feature_columns))

        # Predict
        predicted_price = (await predictor.predict(X))[0]
        current_price = df['close'].iloc[-1]
        price_change_pct = ((predicted_price - current_price) / current_price) * 100

        # Trading decision
        if price_change_pct > 0.5:
            logger.info(f"🟢 BUY signal: Expected +{price_change_pct:.2f}%")
            # await execute_buy_order()
        elif price_change_pct < -0.5:
            logger.info(f"🔴 SELL signal: Expected {price_change_pct:.2f}%")
            # await execute_sell_order()
        else:
            logger.info(f"⚪ NEUTRAL: {price_change_pct:+.2f}%")

        # Sleep (in real bot, wait for next candle)
        await asyncio.sleep(60)  # 1 minute
        break  # Remove in production


if __name__ == '__main__':
    # Choose example to run:

    # 1. Train new model
    asyncio.run(train_gru_model())

    # 2. Use trained model
    # asyncio.run(use_trained_model())

    # 3. Walk-forward validation
    # asyncio.run(walk_forward_validation())

    # 4. Integration example
    # asyncio.run(integrate_with_trading_bot())

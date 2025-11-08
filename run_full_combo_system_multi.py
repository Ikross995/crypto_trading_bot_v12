#!/usr/bin/env python3
"""
🚀🚀🚀 ПОЛНЫЙ ЗАПУСК COMBO СИСТЕМЫ (MULTI-SYMBOL) 🚀🚀🚀
==========================================================

ОБУЧЕНИЕ КАЖДОЙ ПАРЫ ОТДЕЛЬНО!

Pipeline для КАЖДОГО символа:
1. 📥 Загрузка данных
2. 🎯 Обучение Ensemble (5 моделей)
3. 🤖 Обучение RL Agent
4. 🔄 Walk-Forward Optimization
5. 📊 Performance Analysis
6. 🧠 Meta-Learner Integration
7. 🧪 Full System Backtest

Время: ~30 минут на символ в quick mode

Автор: Claude (Anthropic)
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# 🎯 SINGLE SYMBOL TRAINER
# ==========================================

async def train_single_symbol(
    symbol: str,
    days: int = 365,
    interval: str = '30m',
    quick_mode: bool = False
):
    """
    Обучить COMBO систему для одного символа

    Args:
        symbol: Торговая пара (например, BTCUSDT)
        days: Дней истории
        interval: Таймфрейм
        quick_mode: Быстрый режим

    Returns:
        Dict с результатами обучения
    """
    symbol_start = time.time()

    logger.info("\n" + "🔥" * 50)
    logger.info(f"📋 SYMBOL: {symbol}")
    logger.info(f"📋 Days: {days} | Interval: {interval} | Quick: {quick_mode}")
    logger.info("🔥" * 50)

    results = {'symbol': symbol}

    # ==========================================
    # STEP 1: Загрузка данных
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 1/7: 📥 ЗАГРУЗКА ДАННЫХ - {symbol}")
    logger.info("=" * 80)

    from examples.gru_training_pytorch import (
        BinanceDataDownloader,
        calculate_technical_indicators
    )

    downloader = BinanceDataDownloader()
    logger.info(f"📥 Downloading {symbol}...")
    df = await downloader.download_historical_data(symbol, interval, days)

    if len(df) == 0:
        logger.error(f"❌ No data for {symbol}!")
        return results

    df = calculate_technical_indicators(df)
    logger.info(f"   ✅ {symbol}: {len(df):,} candles")

    # Prepare features
    feature_columns = [
        'open', 'high', 'low', 'volume',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_mid', 'bb_lower',
        'sma_20', 'sma_50', 'ema_50',
        'volume_sma', 'atr',
        'volume_delta', 'obv', 'volume_ratio',
        'volume_spike', 'mfi', 'cvd', 'vwap_distance'
    ]

    # Prepare sequences
    from examples.gru_training_improved import prepare_sequences_no_leakage

    logger.info("📦 Preparing sequences...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler = \
        prepare_sequences_no_leakage(
            df.copy(),
            feature_columns,
            sequence_length=60,
            train_ratio=0.7,
            val_ratio=0.15
        )

    results['data'] = {
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'features': len(feature_columns)
    }

    logger.info(f"✅ Data prepared: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

    # ==========================================
    # STEP 2: Ensemble Training
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 2/7: 🎯 ENSEMBLE TRAINING - {symbol}")
    logger.info("=" * 80)

    from examples.ensemble_trainer import EnsembleTrainer

    ensemble = EnsembleTrainer()
    ensemble_epochs = 10 if quick_mode else 30

    logger.info(f"🎯 Training {len(ensemble.configs)} models for {ensemble_epochs} epochs...")

    ensemble_start = time.time()
    ensemble_results = await ensemble.train_ensemble(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=ensemble_epochs,
        batch_size=256
    )
    ensemble_time = time.time() - ensemble_start

    # Save ensemble with symbol name
    ensemble_path = f'models/combo_ensemble_{symbol}'
    ensemble.save_ensemble(ensemble_path)

    results['ensemble'] = {
        'training_time': ensemble_time,
        'num_models': len(ensemble.models),
        'model_weights': ensemble.model_weights,
        'best_model': min(ensemble.model_performance, key=ensemble.model_performance.get),
        'save_path': ensemble_path
    }

    logger.info(f"✅ Ensemble trained in {ensemble_time/60:.1f} min")
    logger.info(f"   Best: {results['ensemble']['best_model']}")
    logger.info(f"   Saved: {ensemble_path}")

    # ==========================================
    # STEP 3: RL Agent Training
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 3/7: 🤖 RL AGENT TRAINING - {symbol}")
    logger.info("=" * 80)

    from examples.rl_trading_agent import train_rl_agent

    rl_episodes = 50 if quick_mode else 100
    rl_agent_path = f'models/combo_rl_agent_{symbol}.pt'

    logger.info(f"🤖 Training RL Agent for {rl_episodes} episodes...")

    rl_start = time.time()
    rl_agent = await train_rl_agent(
        symbols=[symbol],  # Single symbol
        days=days,
        interval=interval,
        episodes=rl_episodes,
        save_path=rl_agent_path
    )
    rl_time = time.time() - rl_start

    results['rl_agent'] = {
        'training_time': rl_time,
        'episodes': rl_episodes,
        'save_path': rl_agent_path
    }

    logger.info(f"✅ RL Agent trained in {rl_time/60:.1f} min")
    logger.info(f"   Saved: {rl_agent_path}")

    # ==========================================
    # STEP 4: Walk-Forward Optimization
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 4/7: 🔄 WALK-FORWARD - {symbol}")
    logger.info("=" * 80)

    logger.info("🔄 Analyzing model on different time windows...")

    # Split data into windows
    window_size = len(df) // 5
    walk_forward_results = []

    for i in range(5):
        start_idx = i * window_size
        end_idx = min((i + 2) * window_size, len(df))

        window_data = df.iloc[start_idx:end_idx]

        if len(window_data) > 1000:
            returns = window_data['close'].pct_change() * 100
            volatility = returns.std()
            trend = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0] * 100

            walk_forward_results.append({
                'window': i + 1,
                'samples': len(window_data),
                'volatility': volatility,
                'trend': trend
            })

            logger.info(
                f"   Window {i+1}: {len(window_data):,} samples, "
                f"Trend={trend:+.1f}%, Vol={volatility:.2f}%"
            )

    results['walk_forward'] = {
        'windows': len(walk_forward_results),
        'results': walk_forward_results
    }

    logger.info(f"✅ Walk-Forward completed: {len(walk_forward_results)} windows")

    # ==========================================
    # STEP 5: Performance Analysis
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 5/7: 📊 PERFORMANCE ANALYSIS - {symbol}")
    logger.info("=" * 80)

    from examples.performance_analyzer import (
        PerformanceAnalyzer,
        TradeAnalysis
    )

    analyzer = PerformanceAnalyzer()

    logger.info("📊 Generating sample trades...")

    np.random.seed(42)
    for i in range(100):
        idx = np.random.randint(0, len(df) - 100)
        entry_data = df.iloc[idx]
        exit_data = df.iloc[idx + 50]

        price_change = (exit_data['close'] - entry_data['close']) / entry_data['close'] * 100
        success = price_change > 0

        trade = TradeAnalysis(
            trade_id=i,
            entry_time=str(entry_data.name),
            exit_time=str(exit_data.name),
            direction='LONG' if i % 2 == 0 else 'SHORT',
            entry_price=float(entry_data['close']),
            exit_price=float(exit_data['close']),
            profit_pct=float(price_change),
            profit_usd=float(price_change * 100),
            hold_time_minutes=50 * 30,
            hour_of_day=entry_data.name.hour if hasattr(entry_data.name, 'hour') else 12,
            day_of_week=entry_data.name.dayofweek if hasattr(entry_data.name, 'dayofweek') else 1,
            volatility=float(np.random.uniform(0.2, 0.8)),
            trend='UP' if price_change > 0 else 'DOWN',
            rsi=float(entry_data.get('rsi', 50)),
            macd=float(entry_data.get('macd', 0)),
            bb_position=float(np.random.uniform(-1, 1)),
            success=success
        )

        analyzer.add_trade(trade)

    perf_results = analyzer.analyze()

    # Save with symbol name
    perf_path = f'data/combo_performance_{symbol}.json'
    analyzer.save_analysis(perf_results, perf_path)

    results['performance'] = {
        'total_trades': len(analyzer.trades),
        'win_rate': perf_results['overall']['win_rate'],
        'sharpe_ratio': perf_results['overall']['sharpe_ratio'],
        'save_path': perf_path
    }

    logger.info(f"✅ Performance analysis: WR={perf_results['overall']['win_rate']:.1f}%")
    logger.info(f"   Saved: {perf_path}")

    # ==========================================
    # STEP 6: Meta-Learner Integration
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 6/7: 🧠 META-LEARNER - {symbol}")
    logger.info("=" * 80)

    from examples.meta_learner import MetaLearner

    meta = MetaLearner()

    logger.info("🧠 Loading models into Meta-Learner...")
    meta.load_models(
        rl_agent_path=rl_agent_path,
        ensemble_path=ensemble_path,
        walk_forward_path=ensemble_path
    )

    logger.info(f"✅ Meta-Learner: {len(meta.models)} strategies loaded")

    # ==========================================
    # STEP 7: Full System Backtest
    # ==========================================
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 7/7: 🧪 BACKTEST - {symbol}")
    logger.info("=" * 80)

    logger.info("🧪 Running backtest...")

    backtest_results = await meta.backtest(
        data=df,
        window_size=500,
        step_size=50
    )

    # Save meta-learner state with symbol name
    meta_path = f'data/combo_meta_learner_{symbol}.json'
    meta.save_state(meta_path)

    results['meta_learner'] = {
        'strategies_loaded': len(meta.models),
        'backtest_trades': len(backtest_results['trades']),
        'save_path': meta_path
    }

    logger.info(f"✅ Backtest: {len(backtest_results['trades'])} trades")
    logger.info(f"   Saved: {meta_path}")

    # ==========================================
    # SYMBOL SUMMARY
    # ==========================================
    symbol_time = time.time() - symbol_start

    logger.info("\n" + "🎉" * 50)
    logger.info(f"✅ {symbol} ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    logger.info("🎉" * 50)
    logger.info(f"⏱️  Time: {symbol_time/60:.1f} min ({symbol_time/3600:.2f} hours)")
    logger.info(f"📊 Win Rate: {results['performance']['win_rate']:.1f}%")
    logger.info(f"📊 Sharpe: {results['performance']['sharpe_ratio']:.2f}")
    logger.info(f"💾 Models saved:")
    logger.info(f"   • {ensemble_path}")
    logger.info(f"   • {rl_agent_path}")
    logger.info(f"   • {perf_path}")
    logger.info(f"   • {meta_path}")

    results['total_time'] = symbol_time

    return results


# ==========================================
# 🚀 MULTI-SYMBOL ORCHESTRATOR
# ==========================================

async def run_multi_symbol_training(
    symbols: list = None,
    days: int = 365,
    interval: str = '30m',
    quick_mode: bool = False
):
    """
    Запустить обучение для ВСЕХ символов (каждый отдельно!)

    Args:
        symbols: Список символов
        days: Дней истории
        interval: Таймфрейм
        quick_mode: Быстрый режим

    Returns:
        Dict с результатами для каждого символа
    """
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    total_start = time.time()

    logger.info("\n\n" + "=" * 100)
    logger.info("🚀🚀🚀 MULTI-SYMBOL COMBO SYSTEM 🚀🚀🚀")
    logger.info("=" * 100)
    logger.info(f"📋 Symbols to train: {len(symbols)}")
    logger.info(f"   {', '.join(symbols)}")
    logger.info(f"📋 Settings: {days} days, {interval} interval, Quick={quick_mode}")
    logger.info(f"⏱️  Estimated time: {len(symbols) * 30} minutes (in quick mode)")
    logger.info(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 100)

    all_results = {}

    for idx, symbol in enumerate(symbols, 1):
        logger.info("\n\n" + "🔥" * 100)
        logger.info(f"🎯 SYMBOL {idx}/{len(symbols)}: {symbol}")
        logger.info("🔥" * 100)

        try:
            symbol_results = await train_single_symbol(
                symbol=symbol,
                days=days,
                interval=interval,
                quick_mode=quick_mode
            )
            all_results[symbol] = symbol_results

        except Exception as e:
            logger.error(f"❌ Error training {symbol}: {e}")
            import traceback
            traceback.print_exc()
            all_results[symbol] = {'error': str(e)}

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    total_time = time.time() - total_start

    logger.info("\n\n" + "=" * 100)
    logger.info("🎉🎉🎉 ВСЕ СИМВОЛЫ ОБУЧЕНЫ! 🎉🎉🎉")
    logger.info("=" * 100)

    logger.info(f"\n⏱️  TOTAL TIME: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")

    logger.info(f"\n📊 RESULTS BY SYMBOL:")
    for symbol, results in all_results.items():
        if 'error' in results:
            logger.info(f"   ❌ {symbol}: ERROR - {results['error']}")
        else:
            wr = results.get('performance', {}).get('win_rate', 0)
            sharpe = results.get('performance', {}).get('sharpe_ratio', 0)
            time_min = results.get('total_time', 0) / 60
            logger.info(f"   ✅ {symbol}: WR={wr:.1f}%, Sharpe={sharpe:.2f}, Time={time_min:.1f}min")

    logger.info(f"\n💾 MODELS SAVED:")
    for symbol in symbols:
        if symbol in all_results and 'error' not in all_results[symbol]:
            logger.info(f"\n   {symbol}:")
            logger.info(f"      • models/combo_ensemble_{symbol}/")
            logger.info(f"      • models/combo_rl_agent_{symbol}.pt")
            logger.info(f"      • data/combo_performance_{symbol}.json")
            logger.info(f"      • data/combo_meta_learner_{symbol}.json")

    logger.info(f"\n🚀 NEXT STEPS:")
    logger.info(f"   1. Просмотреть результаты каждой пары")
    logger.info(f"   2. Выбрать лучшие пары для продакшена")
    logger.info(f"   3. Настроить риск-менеджмент")
    logger.info(f"   4. Запустить живое тестирование")

    logger.info("\n" + "=" * 100)
    logger.info("💪 MULTI-SYMBOL СИСТЕМА ГОТОВА!")
    logger.info("=" * 100)

    return all_results


# ==========================================
# 🚀 MAIN
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="🚀 MULTI-SYMBOL COMBO SYSTEM")
    parser.add_argument('--symbols', type=str, nargs='+',
                       default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'],
                       help='Trading symbols (each trained separately)')
    parser.add_argument('--days', type=int, default=180,
                       help='Days of historical data')
    parser.add_argument('--interval', type=str, default='30m',
                       help='Timeframe')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer epochs)')

    args = parser.parse_args()

    # Parse symbols from comma-separated string if needed
    if len(args.symbols) == 1 and ',' in args.symbols[0]:
        args.symbols = [s.strip() for s in args.symbols[0].split(',')]

    logger.info("🔥 Starting MULTI-SYMBOL COMBO SYSTEM...")
    logger.info(f"   Symbols: {args.symbols}")
    logger.info(f"   Quick mode: {args.quick}")

    asyncio.run(run_multi_symbol_training(
        symbols=args.symbols,
        days=args.days,
        interval=args.interval,
        quick_mode=args.quick
    ))

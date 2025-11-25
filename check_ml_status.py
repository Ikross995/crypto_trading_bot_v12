#!/usr/bin/env python3
"""
ML Learning Status Monitor
Показывает текущий статус ML моделей и прогресс обучения
"""

import json
from pathlib import Path
from datetime import datetime, timezone
import sys

def format_size(bytes):
    """Форматирует размер файла"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

def check_ml_status():
    """Проверяет статус ML системы"""

    print("=" * 80)
    print("🧠 ML LEARNING SYSTEM STATUS")
    print("=" * 80)

    ml_data_dir = Path("ml_learning_data")
    enhanced_data_dir = Path("enhanced_learning_data")

    # Проверяем ml_learning_data
    if not ml_data_dir.exists():
        print("\n❌ ML data directory not found!")
        print("   Models haven't been trained yet. Start the bot to begin learning.")
        return False

    print("\n📁 ML Data Directory:")
    print("-" * 80)

    # Список моделей
    models = ['pnl_predictor', 'win_probability', 'hold_time_predictor', 'risk_estimator']

    total_samples = 0
    models_found = 0

    for model_name in models:
        metadata_file = ml_data_dir / f"{model_name}_metadata.json"
        model_file = ml_data_dir / f"{model_name}_model.pkl"
        scaler_file = ml_data_dir / f"{model_name}_scaler.pkl"

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            samples_seen = metadata.get('samples_seen', 0)
            saved_at = metadata.get('saved_at', 'Unknown')
            is_fitted = metadata.get('is_fitted', False)

            total_samples += samples_seen
            models_found += 1

            model_size = model_file.stat().st_size if model_file.exists() else 0
            scaler_size = scaler_file.stat().st_size if scaler_file.exists() else 0

            status = "✅ Trained" if is_fitted else "❌ Not trained"
            print(f"  {model_name:25s}: {status:15s} | {samples_seen:3d} samples | {format_size(model_size + scaler_size)}")
        else:
            print(f"  {model_name:25s}: ❌ Not found")

    print(f"\n  Total: {models_found}/{len(models)} models loaded | {total_samples} total samples")

    # Проверяем trade outcomes
    outcomes_file = ml_data_dir / "trade_outcomes.json"
    contexts_file = ml_data_dir / "market_contexts.json"

    if outcomes_file.exists():
        with open(outcomes_file, 'r') as f:
            outcomes = json.load(f)
        print(f"  Trade outcomes: {len(outcomes)}")

    if contexts_file.exists():
        with open(contexts_file, 'r') as f:
            contexts = json.load(f)
        print(f"  Market contexts: {len(contexts)}")

    # Показываем прогресс обучения
    print("\n\n📊 LEARNING PROGRESS:")
    print("-" * 80)

    avg_samples = total_samples // len(models) if models_found > 0 else 0

    if avg_samples < 50:
        phase = "COLD START"
        progress = avg_samples / 50 * 100
        description = "Collecting initial data - all IMBA signals allowed"
    elif avg_samples < 200:
        phase = "LEARNING"
        progress = (avg_samples - 50) / 150 * 100
        description = "Gradually incorporating ML predictions"
    else:
        phase = "FULL ML"
        progress = 100.0
        description = "ML fully active and optimizing trades"

    # Progress bar
    bar_length = 50
    filled_length = int(bar_length * progress / 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)

    print(f"  Phase: {phase}")
    print(f"  Progress: [{bar}] {progress:.1f}%")
    print(f"  Samples: {avg_samples}/200 per model")
    print(f"  Status: {description}")

    # Оценка качества
    if avg_samples >= 50:
        print("\n\n🎯 MODEL QUALITY:")
        print("-" * 80)

        confidence = min(1.0, avg_samples / 200)
        quality_stars = '⭐' * int(confidence * 5)

        print(f"  Confidence: {confidence:.1%} {quality_stars}")

        if avg_samples < 100:
            print(f"  Recommendation: Continue training for better predictions")
        elif avg_samples < 200:
            print(f"  Recommendation: Good progress, approaching optimal performance")
        else:
            print(f"  Recommendation: Excellent! ML system is fully trained")

    # Последнее сохранение
    print("\n\n💾 LAST SAVE:")
    print("-" * 80)

    latest_save = None
    for model_name in models:
        metadata_file = ml_data_dir / f"{model_name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            saved_at = metadata.get('saved_at')
            if saved_at:
                try:
                    save_time = datetime.fromisoformat(saved_at.replace('Z', '+00:00'))
                    if latest_save is None or save_time > latest_save:
                        latest_save = save_time
                except:
                    pass

    if latest_save:
        time_ago = datetime.now(timezone.utc) - latest_save
        hours_ago = time_ago.total_seconds() / 3600

        if hours_ago < 1:
            time_str = f"{int(time_ago.total_seconds() / 60)} minutes ago"
        elif hours_ago < 24:
            time_str = f"{int(hours_ago)} hours ago"
        else:
            time_str = f"{int(hours_ago / 24)} days ago"

        print(f"  Last saved: {latest_save.strftime('%Y-%m-%d %H:%M:%S UTC')} ({time_str})")
    else:
        print(f"  Last saved: Unknown")

    # Рекомендации
    print("\n\n💡 RECOMMENDATIONS:")
    print("-" * 80)

    if avg_samples == 0:
        print("  1. Start the trading bot to begin ML training")
        print("  2. ML will learn from each completed trade")
    elif avg_samples < 50:
        print(f"  1. {50 - avg_samples} more trades needed to exit cold start phase")
        print(f"  2. Continue trading to collect baseline data")
    elif avg_samples < 200:
        print(f"  1. {200 - avg_samples} more trades to reach full ML capability")
        print(f"  2. ML is already improving trade quality")
    else:
        print(f"  1. ✅ ML system is fully trained and operational")
        print(f"  2. Continue trading for continuous improvement")

    print("\n" + "=" * 80)

    return True

if __name__ == "__main__":
    try:
        success = check_ml_status()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
